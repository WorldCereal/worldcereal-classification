"""Authentication helpers for notebook apps."""

import os
import time
from typing import Optional

import ipywidgets as widgets
import openeo
from IPython import get_ipython
from IPython.display import HTML
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

    def _schedule_output(callback) -> None:
        ip = get_ipython()
        io_loop = getattr(getattr(ip, "kernel", None), "io_loop", None)
        if io_loop is not None:
            io_loop.add_callback(callback)
        else:
            callback()

    def append_html(html: str) -> None:
        _schedule_output(lambda: output.append_display_data(HTML(html)))

    def append_line(message: str) -> None:
        _schedule_output(lambda: output.append_stdout(f"{message}\n"))

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
                os.environ.get("OPENEO_OIDC_DEVICE_CODE_MAX_POLL_TIME") or 30
            )

        authenticator = OidcDeviceAuthenticator(
            client_info=client_info, max_poll_time=max_poll_time
        )

        try:
            append_line("Starting device flow...")
            append_line(f"Requesting device code (timeout {request_timeout}s)...")
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
            attempt = 0
            backoff = 2
            while True:
                attempt += 1
                try:
                    resp = authenticator._requests.post(
                        url=authenticator._device_code_url,
                        data=post_data,
                        timeout=(request_timeout, request_timeout),
                    )
                    break
                except Exception as exc:
                    append_line(
                        "Device code request failed " f"(attempt {attempt}/3): {exc}"
                    )
                    if attempt >= 3:
                        raise
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10)
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
        except Exception as exc:
            append_line(f"❌ Failed to start device flow: {exc}")
            return None
        append_line("Click the link below to authenticate with your CDSE credentials ⬇️")
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
        max_time = int(authenticator._max_poll_time)
        last_heartbeat = -1

        while elapsed() <= authenticator._max_poll_time:
            time.sleep(sleep)

            elapsed_seconds = int(elapsed())
            if elapsed_seconds - last_heartbeat >= 5:
                remaining = max(max_time - elapsed_seconds, 0)
                append_line(
                    f"Waiting for authorization... elapsed {elapsed_seconds}s / {max_time}s (remaining {remaining}s)."
                )
                last_heartbeat = elapsed_seconds

            if elapsed() >= next_poll:
                append_line(
                    f"Polling authorization status... elapsed {elapsed_seconds}s / {max_time}s."
                )
                try:
                    resp = authenticator._requests.post(
                        url=token_endpoint, data=post_data, timeout=request_timeout
                    )
                except Exception as exc:
                    append_line(f"Temporary network error: {exc}. Retrying...")
                    next_poll = elapsed() + poll_interval
                    continue
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
                    append_line("Authorization pending...")
                elif error == "slow_down":
                    append_line("Slowing down...")
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
