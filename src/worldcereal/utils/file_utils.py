import os
from pathlib import Path
import shutil
from worldcereal.extract.utils import pipeline_log

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def safe_chown_group(path: Path, group_name: str = "vito") -> None:
    if os.name != "posix":
        # Windows: skip chown
        return
    import grp  # only import on POSIX
    try:
        gid = grp.getgrnam(group_name).gr_gid
    except KeyError:
        return
    try:
        shutil.chown(path, group=gid)
    except PermissionError:
        return

def set_file_permissions(path: Path):
    """Set file permissions in a Windows-compatible way."""
    pipeline_log.info(f"Setting file permissions for {path}")
    if os.name == 'posix':
        try:
            os.chmod(path, 0o755)
            safe_chown_group(path)
        except (PermissionError, OSError):
            pass