"""
Utilities to fetch the WorldCereal legend + class mapping Excel from SharePoint
and generate mapping JSONs.

This module keeps SharePoint access logic separate from notebooks and lets you
read the latest legend/mappings Excel directly from the source.

Environment variables
---------------------
``WORLDCEREAL_SP_SITE_URL`` (required)
    The SharePoint site URL, e.g. ``https://vitoresearch.sharepoint.com/sites/21717-ccn-world-cereal``.

``WORLDCEREAL_SP_FILE_URL`` (required)
    Server-relative URL, drive-relative path, or a full SharePoint share link
    pointing to ``WorldCereal_LC_CT_legend_v2_class_mappings.xlsx``. Examples:
    ``/sites/21717-ccn-world-cereal/Shared Documents/Research and Development/Legend/WorldCereal_LC_CT_legend_v2_class_mappings.xlsx``
    ``Shared Documents/Research and Development/Legend/WorldCereal_LC_CT_legend_v2_class_mappings.xlsx``
    ``https://vitoresearch.sharepoint.com/:x:/r/sites/21717-ccn-world-cereal/Shared%20Documents/Research%20and%20Development/Legend/WorldCereal_LC_CT_legend_v2_class_mappings.xlsx?...``

Two authentication flows are supported (client credentials are preferred):

1. Client Credentials (Microsoft Graph)
   * ``WORLDCEREAL_SP_CLIENT_ID``
   * ``WORLDCEREAL_SP_CLIENT_SECRET``
   * ``WORLDCEREAL_SP_TENANT_ID``

2. Username / Password (legacy personal accounts)
   * ``WORLDCEREAL_SP_USERNAME``
   * ``WORLDCEREAL_SP_PASSWORD``

Only one method is required. If both are provided, client credentials take priority.

Dependencies
------------
Requires the ``Office365-REST-Python-Client`` package. Install with:
``pip install Office365-REST-Python-Client`` (or add it to ``pyproject.toml``).

Example
-------
>>> import os
>>> from pathlib import Path
>>> from get_mappings_from_legend import (
...     SharePointConfig,
...     get_legend_with_mappings_df,
...     build_class_mappings,
...     write_class_mappings_json,
... )
>>> config = SharePointConfig(
...     site_url="https://vitoresearch.sharepoint.com/sites/21717-ccn-world-cereal",
...     file_server_relative_url=(
...         "/sites/21717-ccn-world-cereal/Shared Documents/Research and Development/Legend/"
...         "WorldCereal_LC_CT_legend_v2_class_mappings.xlsx"
...     ),
...     tenant_id=os.environ.get("WORLDCEREAL_SP_TENANT_ID"),
...     client_id=os.environ.get("WORLDCEREAL_SP_CLIENT_ID"),
...     client_secret=os.environ.get("WORLDCEREAL_SP_CLIENT_SECRET"),
... )
>>> legend_df = get_legend_with_mappings_df(config=config, sheet_name=0)
>>> mappings = build_class_mappings(legend_df)
>>> write_class_mappings_json(mappings, Path("class_mappings.json"))
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd


class SharePointDependencyError(ImportError):
    """Raised when the required Office365 dependency is missing."""


def _ensure_office365_imports() -> None:
    """Import Office365 modules on demand for clearer errors."""
    try:
        from office365.runtime.auth.client_credential import (  # noqa: F401
            ClientCredential,
        )
        from office365.runtime.auth.user_credential import UserCredential  # noqa: F401
        from office365.sharepoint.client_context import ClientContext  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency check
        raise SharePointDependencyError(
            "The 'Office365-REST-Python-Client' package is required to download "
            "files from SharePoint. Install it with "
            "`pip install Office365-REST-Python-Client`."
        ) from exc


def normalize_sharepoint_file_url(file_url: str) -> str:
    """Normalize SharePoint file URLs and share links to a server-relative path."""
    parsed = urlparse(file_url)
    path = parsed.path if parsed.scheme and parsed.netloc else file_url
    path = unquote(path)

    # Handle share links like /:x:/r/sites/<site>/Shared Documents/...
    share_link_match = re.match(r"^/:[^/]+:/r(/.+)$", path)
    if share_link_match:
        path = share_link_match.group(1)

    return path


@dataclass(frozen=True)
class SharePointConfig:
    """Configuration required to access the WorldCereal SharePoint site."""

    site_url: str
    file_server_relative_url: str
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SharePointConfig":
        """Build a configuration object from environment variables."""
        site_url = os.environ.get("WORLDCEREAL_SP_SITE_URL")
        file_url = os.environ.get("WORLDCEREAL_SP_FILE_URL")

        if not site_url or not file_url:
            raise ValueError(
                "Environment variables WORLDCEREAL_SP_SITE_URL and "
                "WORLDCEREAL_SP_FILE_URL must be set to read the legend Excel "
                "from SharePoint."
            )

        return cls(
            site_url=site_url,
            file_server_relative_url=normalize_sharepoint_file_url(file_url),
            tenant_id=os.environ.get("WORLDCEREAL_SP_TENANT_ID"),
            client_id=os.environ.get("WORLDCEREAL_SP_CLIENT_ID"),
            client_secret=os.environ.get("WORLDCEREAL_SP_CLIENT_SECRET"),
            username=os.environ.get("WORLDCEREAL_SP_USERNAME"),
            password=os.environ.get("WORLDCEREAL_SP_PASSWORD"),
        )


def _build_context(config: SharePointConfig):
    """Create an authenticated SharePoint ClientContext instance."""
    _ensure_office365_imports()

    from office365.runtime.auth.client_credential import ClientCredential
    from office365.runtime.auth.user_credential import UserCredential
    from office365.sharepoint.client_context import ClientContext

    if config.client_id and config.client_secret:
        credentials = ClientCredential(config.client_id, config.client_secret)
        return ClientContext(config.site_url).with_credentials(credentials)

    if config.username and config.password:
        credentials = UserCredential(config.username, config.password)
        return ClientContext(config.site_url).with_credentials(credentials)

    raise ValueError(
        "SharePoint credentials are missing. Provide either client credentials "
        "(WORLDCEREAL_SP_CLIENT_ID + WORLDCEREAL_SP_CLIENT_SECRET) or "
        "username/password (WORLDCEREAL_SP_USERNAME + WORLDCEREAL_SP_PASSWORD)."
    )


def _download_file_bytes(config: SharePointConfig) -> bytes:
    """Download the raw Excel file from SharePoint."""
    if config.tenant_id:
        if not (config.client_id and config.client_secret):
            raise ValueError(
                "WORLDCEREAL_SP_TENANT_ID is set, but WORLDCEREAL_SP_CLIENT_ID and "
                "WORLDCEREAL_SP_CLIENT_SECRET are missing. Provide app credentials "
                "to authenticate via Microsoft Graph."
            )
        return _download_file_bytes_via_graph(config)

    return _download_file_bytes_via_sharepoint_sdk(config)


def _download_file_bytes_via_sharepoint_sdk(config: SharePointConfig) -> bytes:
    """Download the Excel file using the Office365 ClientContext SDK."""
    ctx = _build_context(config)
    file = ctx.web.get_file_by_server_relative_url(config.file_server_relative_url)
    buffer = BytesIO()
    file.download(buffer)
    ctx.execute_query()
    return buffer.getvalue()


def _download_file_bytes_via_graph(config: SharePointConfig) -> bytes:
    """Download the Excel file using Microsoft Graph with client credentials."""
    host, site_path = _parse_site_components(config.site_url)
    drive_relative_path = _derive_drive_relative_path(
        config.file_server_relative_url, site_path
    )
    token = _graph_request_access_token(
        config.tenant_id, config.client_id, config.client_secret
    )
    site_id = _graph_resolve_site_id(token, host, site_path)
    return _graph_download_file(token, site_id, drive_relative_path)


def _parse_site_components(site_url: str) -> tuple[str, str]:
    parsed = urlparse(site_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            "WORLDCEREAL_SP_SITE_URL must be a complete URL, "
            "e.g. 'https://vitoresearch.sharepoint.com/sites/worldcereal'."
        )
    site_path = parsed.path.strip("/")
    if not site_path:
        raise ValueError(
            "WORLDCEREAL_SP_SITE_URL is missing the site path segment "
            "(expected something like '/sites/<site-name>')."
        )
    return parsed.netloc, site_path


def _derive_drive_relative_path(file_url: str, site_path: str) -> str:
    relative_path = file_url.lstrip("/")
    if not relative_path:
        raise ValueError(
            "WORLDCEREAL_SP_FILE_URL must contain a path pointing to the Excel file."
        )

    normalized_site = site_path.strip("/")
    site_prefix = f"{normalized_site}/"
    if normalized_site and relative_path.lower().startswith(site_prefix.lower()):
        relative_path = relative_path[len(site_prefix) :]

    return relative_path


def _graph_request_access_token(
    tenant_id: str, client_id: str, client_secret: str
) -> str:
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = urlencode(
        {
            "client_id": client_id,
            "scope": "https://graph.microsoft.com/.default",
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        }
    ).encode("utf-8")
    raw_response = _perform_http_request(
        token_url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    payload = json.loads(raw_response.decode("utf-8"))
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError(
            f"Unable to retrieve access token from Microsoft Graph: {payload}"
        )
    return access_token


def _graph_resolve_site_id(access_token: str, host: str, site_path: str) -> str:
    encoded_site_path = quote(f"/{site_path.strip('/')}", safe="/-_.")
    site_url = f"https://graph.microsoft.com/v1.0/sites/{host}:{encoded_site_path}"
    payload = json.loads(
        _perform_http_request(
            site_url, headers={"Authorization": f"Bearer {access_token}"}
        ).decode("utf-8")
    )
    site_id = payload.get("id")
    if not site_id:
        raise RuntimeError(f"Unable to resolve Site ID from response: {payload}")
    return site_id


def _graph_download_file(
    access_token: str, site_id: str, drive_relative_path: str
) -> bytes:
    clean_path = drive_relative_path.lstrip("/")
    if not clean_path:
        raise ValueError(
            "The resolved drive-relative path is empty. Check WORLDCEREAL_SP_FILE_URL."
        )
    encoded_file_path = quote(clean_path, safe="/-_.()")
    download_url = (
        f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{encoded_file_path}:/content"
    )
    return _perform_http_request(
        download_url, headers={"Authorization": f"Bearer {access_token}"}
    )


def _perform_http_request(
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[dict[str, str]] = None,
    method: Optional[str] = None,
) -> bytes:
    request_headers = headers or {}
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request) as response:
            return response.read()
    except HTTPError as exc:  # pragma: no cover - network failure path
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} error when calling '{url}': {error_body}") from exc


@lru_cache(maxsize=1)
def get_excel_from_sharepoint(
    config: Optional[SharePointConfig] = None, **read_excel_kwargs
) -> pd.DataFrame:
    """Fetch the latest legend/mappings Excel from SharePoint as a DataFrame.

    Parameters
    ----------
    config : SharePointConfig, optional
        Explicit configuration. If omitted, :meth:`SharePointConfig.from_env`
        is used to read the parameters from environment variables.
    **read_excel_kwargs
        Extra keyword arguments forwarded to ``pandas.read_excel``.

    Returns
    -------
    pandas.DataFrame
        The parsed Excel contents.
    """
    cfg = config or SharePointConfig.from_env()
    excel_bytes = _download_file_bytes(cfg)
    read_excel_kwargs.setdefault("sheet_name", 0)
    return pd.read_excel(BytesIO(excel_bytes), **read_excel_kwargs)


def refresh_excel_from_sharepoint_cache(
    config: Optional[SharePointConfig] = None, **read_excel_kwargs
) -> pd.DataFrame:
    """Clear cached DataFrame and fetch a fresh copy from SharePoint."""
    get_excel_from_sharepoint.cache_clear()
    return get_excel_from_sharepoint(config=config, **read_excel_kwargs)


def _find_ewoc_code_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if col.lower() in {"ewoc_code", "ewoc code", "ewoccode"}]
    if candidates:
        return candidates[0]
    for col in df.columns:
        lowered = col.lower()
        if "ewoc" in lowered and "code" in lowered:
            return col
    raise ValueError("Could not locate an EWOC code column in the legend sheet.")


def _normalize_ewoc_code(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        try:
            return str(int(value))
        except (ValueError, TypeError):
            pass
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("-", "").replace(" ", "")
    if text.endswith(".0"):
        text = text[:-2]
    if any(token in text for token in ("e", "E", ".")):
        try:
            text = str(int(float(text)))
        except ValueError:
            pass
    return text or None


def _find_mapping_columns(df: pd.DataFrame) -> List[str]:
    pattern = re.compile(r"^(LANDCOVER|CROPTYPE).*_CLASS$", re.IGNORECASE)
    return [col for col in df.columns if pattern.match(col)]


def build_class_mappings(df: pd.DataFrame, ewoc_code_col: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Build mapping dictionaries for LANDCOVER/CROPTYPE classes from the legend sheet.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame read from the legend/mappings Excel.
    ewoc_code_col : str, optional
        Column name holding EWOC codes. If not provided, it is detected.

    Returns
    -------
    dict
        Mapping name -> {ewoc_code: class_name} dictionary.
    """
    ewoc_code_col = ewoc_code_col or _find_ewoc_code_column(df)
    mapping_columns = _find_mapping_columns(df)
    if not mapping_columns:
        raise ValueError(
            "No LANDCOVER*_CLASS or CROPTYPE*_CLASS columns found in the sheet."
        )

    mappings: Dict[str, Dict[str, str]] = {}
    for column in mapping_columns:
        name = re.sub(r"_CLASS$", "", column, flags=re.IGNORECASE).upper()
        pairs = df[[ewoc_code_col, column]]
        mapping: Dict[str, str] = {}
        for ewoc, cls in pairs.itertuples(index=False):
            key = _normalize_ewoc_code(ewoc)
            if not key or pd.isna(cls):
                continue
            value = str(cls).strip()
            if not value:
                continue
            mapping[key] = value
        if mapping:
            mappings[name] = mapping

    if not mappings:
        raise ValueError("No mappings could be built from the provided sheet.")
    return mappings


def write_class_mappings_json(mappings: Mapping[str, Mapping[str, str]], output_path: Path) -> None:
    """Write the mapping dictionaries to a single JSON file."""
    serialized = {k: dict(v) for k, v in mappings.items()}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, sort_keys=True)
