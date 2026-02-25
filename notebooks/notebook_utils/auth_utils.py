"""Authentication helpers for notebook apps."""

import os
import time
from typing import Optional

import ipywidgets as widgets
import openeo
from IPython.display import HTML, display
from openeo.rest.auth.oidc import (
    DefaultOidcClientGrant,
    OidcDeviceAuthenticator,
    OidcDeviceCodePollTimeout,
    OidcException,
    VerificationInfo,
    clip,
    create_timer,
)
from openeo.rest.connection import OidcBearerAuth


def trigger_cdse_authentication(
    output: widgets.Output,
    *,
    max_poll_time: Optional[int] = None,
    request_timeout: int = 10,
) -> Optional[openeo.Connection]:
    """Trigger CDSE device authentication and render the link in the given output widget."""

    def append_html(html: str) -> None:
        with output:
            display(HTML(html))

    def append_line(message: str) -> None:
        with output:
            print(message)

    success = False
    try:
        connection = openeo.connect("openeo.dataspace.copernicus.eu")
        _g = DefaultOidcClientGrant
        provider_id, client_info = connection._get_oidc_provider_and_client_info(
            provider_id=None,
            client_id=None,
            client_secret=None,
            default_client_grant_check=lambda grants: (
                _g.REFRESH_TOKEN in grants
                and (_g.DEVICE_CODE in grants or _g.DEVICE_CODE_PKCE in grants)
            ),
        )

        if max_poll_time is None:
            max_poll_time = int(
                os.environ.get("OPENEO_OIDC_DEVICE_CODE_MAX_POLL_TIME") or 60
            )

        authenticator = OidcDeviceAuthenticator(
            client_info=client_info, max_poll_time=max_poll_time
        )

        post_data = {
            "client_id": authenticator.client_id,
            "scope": authenticator._client_info.provider.get_scopes_string(
                request_refresh_token=True
            ),
        }
        if authenticator._pkce:
            post_data["code_challenge"] = authenticator._pkce.code_challenge
            post_data["code_challenge_method"] = (
                authenticator._pkce.code_challenge_method
            )
        resp = authenticator._requests.post(
            url=authenticator._device_code_url,
            data=post_data,
            timeout=(request_timeout, request_timeout),
        )
        if resp.status_code != 200:
            raise OidcException(
                "Failed to get verification URL and user code from {u!r}: {s} {r!r} {t!r}".format(
                    s=resp.status_code,
                    r=resp.reason,
                    u=resp.url,
                    t=resp.text,
                )
            )
        data = resp.json()
        append_line("Device code received.")
        verification_info = VerificationInfo(
            verification_uri=(
                data["verification_uri"]
                if "verification_uri" in data
                else data["verification_url"]
            ),
            verification_uri_complete=data.get("verification_uri_complete"),
            device_code=data["device_code"],
            user_code=data["user_code"],
            interval=data.get("interval", 5),
        )
        append_line("Click the link below to authenticate with your CDSE credentials ⬇️")
        verification_url = (
            verification_info.verification_uri_complete
            or verification_info.verification_uri
        )
        append_line(f"Verification URL: {verification_url}")
        append_line(f"User code: {verification_info.user_code}")
        if verification_info.verification_uri_complete:
            append_html(
                '<p><a href="{url}" target="_blank" rel="noopener">'
                "Open the CDSE login page</a></p>".format(
                    url=verification_info.verification_uri_complete
                )
            )
        else:
            append_html(
                '<p>Open <a href="{url}" target="_blank" rel="noopener">'
                "this link</a> and enter code <b>{code}</b>.</p>".format(
                    url=verification_info.verification_uri,
                    code=verification_info.user_code,
                )
            )

        token_endpoint = authenticator._provider_config["token_endpoint"]
        post_data = {
            "client_id": authenticator.client_id,
            "device_code": verification_info.device_code,
            "grant_type": authenticator.grant_type,
        }
        if authenticator._pkce:
            post_data["code_verifier"] = authenticator._pkce.code_verifier
        else:
            post_data["client_secret"] = authenticator.client_secret

        poll_interval = max(verification_info.interval, 5)
        elapsed = create_timer()
        next_poll = elapsed() + poll_interval
        sleep = clip(authenticator._max_poll_time / 100, min=1, max=5)
        append_line("Waiting for authorization (up to 60s)...")

        while elapsed() <= authenticator._max_poll_time:
            time.sleep(sleep)

            if elapsed() >= next_poll:
                try:
                    resp = authenticator._requests.post(
                        url=token_endpoint,
                        data=post_data,
                        timeout=(request_timeout, request_timeout),
                    )
                except Exception as exc:
                    raise OidcException(f"Token request failed: {exc}") from exc
                if resp.status_code == 200:
                    tokens = authenticator._get_access_token_result(data=resp.json())
                    refresh_token = tokens.refresh_token
                    if refresh_token:
                        connection._get_refresh_token_store().set_refresh_token(
                            issuer=authenticator.provider_info.issuer,
                            client_id=authenticator.client_id,
                            refresh_token=refresh_token,
                        )
                    connection.auth = OidcBearerAuth(
                        provider_id=provider_id,
                        access_token=tokens.access_token,
                    )
                    success = True
                    return connection

                try:
                    error = resp.json()["error"]
                except Exception:
                    error = "unknown"

                if error == "authorization_pending":
                    next_poll = elapsed() + poll_interval
                elif error == "slow_down":
                    poll_interval += 5
                else:
                    raise OidcException(
                        f"Failed to retrieve access token at {token_endpoint!r}: {resp.status_code} {resp.reason!r} {resp.text!r}"
                    )

                next_poll = elapsed() + poll_interval

        raise OidcDeviceCodePollTimeout(
            f"Timeout ({authenticator._max_poll_time:.1f}s) while polling for access token."
        )
    except Exception as exc:
        append_line(f"❌ CDSE authentication failed: {exc}")
        return None
    finally:
        if success:
            append_line("Authorized successfully.")
