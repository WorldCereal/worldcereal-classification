"""
SharePoint helpers for downloading Excel files and building class mappings.

Environment variables:
- WORLDCEREAL_SP_TENANT_ID
- WORLDCEREAL_SP_CLIENT_ID
- WORLDCEREAL_SP_CLIENT_SECRET
"""

from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import Dict
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

ENV_TENANT_ID = "WORLDCEREAL_SP_TENANT_ID"
ENV_CLIENT_ID = "WORLDCEREAL_SP_CLIENT_ID"
ENV_CLIENT_SECRET = "WORLDCEREAL_SP_CLIENT_SECRET"


def _get_env(name: str) -> str:
    return os.environ[name]


def _perform_http_request(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    method: str | None = None,
) -> bytes:
    request = Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urlopen(request) as response:
            return response.read()
    except HTTPError as exc:  # pragma: no cover - network failure path
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"HTTP {exc.code} error when calling '{url}': {error_body}"
        ) from exc


def _parse_site_components(site_url: str) -> tuple[str, str]:
    parsed = urlparse(site_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            "site_url must be a full URL like https://<tenant>.sharepoint.com/sites/<site>"
        )
    site_path = parsed.path.strip("/")
    if not site_path:
        raise ValueError("site_url must include the site path, e.g. /sites/<site-name>")
    return parsed.netloc, site_path


def _derive_drive_relative_path(file_server_relative_url: str, site_path: str) -> str:
    relative_path = file_server_relative_url.lstrip("/")
    site_prefix = f"{site_path.strip('/')}/"
    if relative_path.lower().startswith(site_prefix.lower()):
        relative_path = relative_path[len(site_prefix) :]
    if not relative_path:
        raise ValueError("file_server_relative_url must point to a file path.")
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
    encoded_file_path = quote(drive_relative_path.lstrip("/"), safe="/-_.()")
    download_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{encoded_file_path}:/content"
    return _perform_http_request(
        download_url, headers={"Authorization": f"Bearer {access_token}"}
    )


def get_excel_from_sharepoint(
    site_url: str,
    file_server_relative_url: str,
    retries: int = 3,
    **read_excel_kwargs,
) -> pd.DataFrame:
    """Download an Excel file from SharePoint and return it as a DataFrame."""
    tenant_id = _get_env(ENV_TENANT_ID)
    client_id = _get_env(ENV_CLIENT_ID)
    client_secret = _get_env(ENV_CLIENT_SECRET)

    host, site_path = _parse_site_components(site_url)
    drive_relative_path = _derive_drive_relative_path(
        file_server_relative_url, site_path
    )
    last_exc: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            token = _graph_request_access_token(tenant_id, client_id, client_secret)
            site_id = _graph_resolve_site_id(token, host, site_path)
            excel_bytes = _graph_download_file(token, site_id, drive_relative_path)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    else:
        raise RuntimeError(
            "Failed to download SharePoint file after retries."
        ) from last_exc

    read_excel_kwargs.setdefault("sheet_name", 0)
    return pd.read_excel(BytesIO(excel_bytes), **read_excel_kwargs)


def build_class_mappings(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Build mapping dictionaries for LANDCOVER/CROPTYPE classes from the legend sheet."""
    mapping_columns = [
        col
        for col in df.columns
        if re.match(r"^(LANDCOVER|CROPTYPE).*_CLASS$", str(col), flags=re.IGNORECASE)
    ]
    if "ewoc_code" not in df.columns:
        raise ValueError("Missing required column: ewoc_code")
    if not mapping_columns:
        raise ValueError("No LANDCOVER*_CLASS or CROPTYPE*_CLASS columns found.")

    # Ensure ewoc_code is a clean integer by removing dashes and coercing errors to NaN, then dropping them
    df["ewoc_code"] = (
        df["ewoc_code"]
        .astype(str)
        .str.replace("-", "")
        .pipe(pd.to_numeric, errors="coerce")
        .astype(int)
    )

    mappings: Dict[str, Dict[str, str]] = {}
    for column in mapping_columns:
        name = re.sub(r"_CLASS$", "", str(column), flags=re.IGNORECASE).upper()
        mapping: Dict[str, str] = {}
        for ewoc_code, cls in df[["ewoc_code", column]].itertuples(index=False):
            if pd.isna(ewoc_code) or pd.isna(cls):
                continue
            value = str(cls).strip()
            if not value:
                continue
            mapping[str(ewoc_code)] = value
        if mapping:
            mappings[name] = mapping

    if not mappings:
        raise ValueError("No mappings could be built from the provided sheet.")
    return mappings
