"""
engine_version.py — Thin facade over core.versioning.build_info.

All version constants live in BUILD_INFO (single source of truth).
This module re-exports for backward compatibility with existing callers.
"""

import logging

from core.versioning.build_info import BUILD_INFO, get_runtime_info, run_self_check

logger = logging.getLogger(__name__)

# Re-exports for callers that import from engine_version directly
ENGINE_VERSION = BUILD_INFO["engine_version"]
RIG_VERSION = BUILD_INFO["rig_version"]
EXECUTION_VERSION = BUILD_INFO["execution_version"]
STRATEGY_VERSION = BUILD_INFO["strategy_version"]
BUILD_TIMESTAMP = BUILD_INFO["build_timestamp"]
GIT_COMMIT = BUILD_INFO["git_commit"]


def get_version_info() -> dict:
    """Return full runtime version metadata."""
    return get_runtime_info()


def log_startup_version() -> dict:
    """Log version info at startup, run self-check, and return the dict."""
    run_self_check()
    info = get_runtime_info()
    logger.info(
        "ENGINE STARTUP: version=%s commit=%s build=%s rig=%s exec=%s env=%s",
        info["engine_version"], info["git_commit"], info["build_timestamp"],
        info["rig_version"], info["execution_version"], info["environment"],
    )
    return info
