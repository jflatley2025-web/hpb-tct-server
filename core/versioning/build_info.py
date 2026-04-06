"""
core/versioning/build_info.py — Single source of truth for all version metadata.

Every runtime surface reads from BUILD_INFO:
  - /api/version endpoint
  - startup log
  - /schematics-5B dashboard panel
  - backtest output stamps

Never duplicate version literals elsewhere.
"""

import os
import subprocess
import time
from datetime import datetime, timezone

# ── Static version constants ────────────────────────────────────────
ENGINE_VERSION = "5B-live-v3.0.1"
STRATEGY_VERSION = "hpb-tct-5b"
RIG_VERSION = "rig-tct-v2.0"
EXECUTION_VERSION = "exec-v2.1-softgates"

# ── Git commit (read once at import time = container start) ─────────
def _get_git_commit() -> str:
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


_GIT_COMMIT = _get_git_commit()
_BUILD_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
_PROCESS_START = time.monotonic()
_PROCESS_START_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ── Environment detection ───────────────────────────────────────────
def _detect_environment() -> str:
    if os.getenv("RENDER"):
        return "production"
    if os.getenv("MOONDEV_PAPER_TRADING", "").lower() == "true":
        return "paper"
    return "development"


# ═══════════════════════════════════════════════════════════════════
# BUILD_INFO — the single source of truth
# ═══════════════════════════════════════════════════════════════════

BUILD_INFO = {
    "engine_version": ENGINE_VERSION,
    "strategy_version": STRATEGY_VERSION,
    "rig_version": RIG_VERSION,
    "execution_version": EXECUTION_VERSION,
    "build_timestamp": _BUILD_TIMESTAMP,
    "git_commit": _GIT_COMMIT,
    "environment": _detect_environment(),
    "changelog": [
        "RIG rewritten for TCT counter-bias range logic",
        "BTC anchor converted from hard block to confidence penalty",
        "Displacement floor reduced from 0.50 to 0.35",
        "Conditional execution instrumentation added",
        "Live deployment version badge added",
    ],
}


def get_runtime_info() -> dict:
    """BUILD_INFO + live runtime fields (uptime, restart check)."""
    uptime_seconds = time.monotonic() - _PROCESS_START
    return {
        **BUILD_INFO,
        "service": "schematics-5B",
        "process_start_time": _PROCESS_START_UTC,
        "uptime_seconds": round(uptime_seconds, 1),
        "uptime_human": _format_uptime(uptime_seconds),
        "restart_required": _restart_required,
        "mode": "paper" if os.getenv("MOONDEV_PAPER_TRADING", "").lower() == "true" else "live",
    }


def _format_uptime(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {int(seconds % 60)}s"


# ── Runtime self-check ──────────────────────────────────────────────
_restart_required = False
_self_check_warnings: list = []


def run_self_check() -> bool:
    """Compare loaded module versions against BUILD_INFO.

    Returns True if all versions match, False if mismatch detected.
    Sets restart_required flag accessible via get_runtime_info().
    """
    import logging
    _logger = logging.getLogger(__name__)
    global _restart_required, _self_check_warnings
    _self_check_warnings = []
    ok = True

    # Check that engine_version.py re-exports match BUILD_INFO
    try:
        import engine_version as ev
        checks = [
            ("ENGINE_VERSION", ev.ENGINE_VERSION, ENGINE_VERSION),
            ("RIG_VERSION", ev.RIG_VERSION, RIG_VERSION),
            ("EXECUTION_VERSION", ev.EXECUTION_VERSION, EXECUTION_VERSION),
        ]
        for name, actual, expected in checks:
            if actual != expected:
                msg = f"VERSION MISMATCH: {name} loaded={actual} expected={expected}"
                _logger.warning(msg)
                _self_check_warnings.append(msg)
                ok = False
    except ImportError:
        _self_check_warnings.append("engine_version module not importable")
        ok = False

    if ok:
        _logger.info("VERSION SELF-CHECK: all modules match BUILD_INFO")
    else:
        _restart_required = True
        _logger.warning("VERSION SELF-CHECK FAILED: restart_required=True")

    return ok
