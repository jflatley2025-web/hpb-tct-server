"""
engine_version.py — Build/deploy version metadata for HPB-TCT server.

Provides version info at startup and via API for dashboard display.
Git commit hash is read once at import time (deploy = full restart on Render).
"""

import os
import subprocess
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# --- Static version ---
ENGINE_VERSION = "5B-live-v3.0.0"
RIG_VERSION = "rig-tct-v2.0"
EXECUTION_VERSION = "exec-v2.1-softgates"
STRATEGY_VERSION = "hpb-tct-5b"

# --- Build timestamp (set at import time = deploy time on Render) ---
BUILD_TIMESTAMP = datetime.now(timezone.utc).isoformat()

# --- Git commit hash ---
def _get_git_commit() -> str:
    """Read current git commit hash. Returns 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return os.getenv("RENDER_GIT_COMMIT", "unknown")[:8]

GIT_COMMIT = _get_git_commit()

# --- Structured version info ---
def get_version_info() -> dict:
    """Return full version metadata dict."""
    return {
        "service": "schematics-5B",
        "engine_version": ENGINE_VERSION,
        "build_timestamp": BUILD_TIMESTAMP,
        "git_commit": GIT_COMMIT,
        "rig_version": RIG_VERSION,
        "execution_version": EXECUTION_VERSION,
        "strategy_version": STRATEGY_VERSION,
        "mode": os.getenv("MOONDEV_PAPER_TRADING", "false").lower() == "true" and "paper" or "live",
    }


def log_startup_version():
    """Log version info at startup."""
    info = get_version_info()
    logger.info(
        "ENGINE STARTUP: version=%s commit=%s build=%s rig=%s exec=%s mode=%s",
        info["engine_version"], info["git_commit"], info["build_timestamp"],
        info["rig_version"], info["execution_version"], info["mode"],
    )
    return info
