import os
from pathlib import Path

from typing_extensions import Literal


def clear_openeo_token_cache(
    verbose=True,
    platform: Literal["linux", "windows"] = "linux",
):
    """Clear the local OpenEO refresh token cache.

    Parameters
    ----------
    verbose : bool, optional
        Whether to print status messages, by default True
    platform : Literal["linux", "windows"], optional
        The operating system platform, by default "linux"
    """
    if platform not in ["linux", "windows"]:
        raise ValueError("platform must be one of 'linux' or 'windows'")

    if platform == "windows":
        token_path = (
            Path.home() / "AppData/Roaming/openeo-python-client/refresh-tokens.json"
        )
    else:
        token_path = (
            Path.home() / ".local/share/openeo-python-client/refresh-tokens.json"
        )

    if token_path.exists():
        try:
            os.remove(token_path)
            if verbose:
                print(f"Deleted cached token at:{token_path}")
        except Exception as e:
            print(f"Failed to delete token:{e}")
    else:
        if verbose:
            print(f"[i] No cached token found at:{token_path}")
