"""Read-only checks before loading the official SDK in a local worker."""

import platform
import subprocess
import sys
from pathlib import Path


class ConversionEnvironmentUnavailable(ValueError):
    """A shared host fault; trying another object cannot repair this environment."""


def check_conversion_environment():
    if sys.platform != "linux" or "microsoft" not in platform.release().lower():
        return
    if not Path("/run/systemd/system").is_dir():
        return  # WSL distributions without systemd have no init.scope policy.
    try:
        result = subprocess.run(
            ["systemctl", "show", "init.scope", "--property=ActiveState", "--property=Result"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConversionEnvironmentUnavailable("wsl_environment_check_failed") from exc
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not properties.get("ActiveState"):
        raise ConversionEnvironmentUnavailable("wsl_environment_check_failed")
    if properties["ActiveState"] in {"deactivating", "inactive", "failed"}:
        # An old OOM counter can put init.scope into stop-sigterm on each distro
        # boot. Its stop timer then kills WSL's init and every converter in it.
        raise ConversionEnvironmentUnavailable("wsl_init_scope_stopping")
