"""
GitHub-based persistence for tensor_trade_log.json.

Reads and writes the trade log to the `data` branch of the GitHub repo
so trade history survives Render deployments.

Required env vars:
  GITHUB_TOKEN  — Personal Access Token with `repo` scope
  GITHUB_REPO   — e.g. jflatley2025-web/hpb-tct-server
"""

import base64
import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DATA_BRANCH = "data"
TRADE_LOG_FILENAME = "tensor_trade_log.json"
PHEMEX_LOG_FILENAME = "phemex_trade_log.json"


def _is_configured() -> bool:
    return bool(os.getenv("GITHUB_TOKEN") and os.getenv("GITHUB_REPO"))


def _headers() -> dict:
    return {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3+json",
    }


def _fetch_log(filename: str, local_path: str, label: str) -> bool:
    """
    Pull a trade log file from the `data` branch on GitHub.
    Writes to local_path only if the local file is missing (startup recovery).
    Returns True on success or if the local file already exists.
    """
    if not _is_configured():
        logger.info("[GITHUB] GITHUB_TOKEN/GITHUB_REPO not set — skipping %s fetch", label)
        return False

    if os.path.exists(local_path):
        logger.info("[GITHUB] Local %s exists — skipping fetch", label)
        return True

    repo = os.getenv("GITHUB_REPO")
    url = f"{GITHUB_API}/repos/{repo}/contents/{filename}"

    try:
        resp = requests.get(
            url, headers=_headers(), params={"ref": DATA_BRANCH}, timeout=15
        )
        if resp.status_code == 404:
            logger.info("[GITHUB] No %s on data branch yet — starting fresh", label)
            return False

        resp.raise_for_status()

        content = resp.json().get("content", "")
        decoded = base64.b64decode(content).decode("utf-8")

        with open(local_path, "w") as f:
            f.write(decoded)

        trades = len(json.loads(decoded).get("trade_history", []))
        logger.info("[GITHUB] %s restored — %d trades recovered", label, trades)
        return True

    except Exception as e:
        logger.warning("[GITHUB] %s fetch failed: %s", label, e)
        return False


def _push_log(filename: str, local_path: str, label: str, commit_msg: str) -> bool:
    """
    Push a trade log file from local disk to the `data` branch on GitHub.
    Creates or updates the file. Returns True on success.
    """
    if not _is_configured():
        logger.debug("[GITHUB] GITHUB_TOKEN/GITHUB_REPO not set — skipping %s push", label)
        return False

    if not os.path.exists(local_path):
        logger.warning("[GITHUB] Local %s not found — nothing to push", label)
        return False

    repo = os.getenv("GITHUB_REPO")
    url = f"{GITHUB_API}/repos/{repo}/contents/{filename}"

    try:
        with open(local_path, "r") as f:
            raw = f.read()

        encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")

        # GitHub requires the existing file's SHA to update it
        sha: Optional[str] = None
        get_resp = requests.get(
            url, headers=_headers(), params={"ref": DATA_BRANCH}, timeout=10
        )
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")

        payload: dict = {
            "message": commit_msg,
            "content": encoded,
            "branch": DATA_BRANCH,
        }
        if sha:
            payload["sha"] = sha

        put_resp = requests.put(url, headers=_headers(), json=payload, timeout=15)
        put_resp.raise_for_status()

        logger.info("[GITHUB] %s pushed to data branch", label)
        return True

    except Exception as e:
        logger.warning("[GITHUB] %s push failed: %s", label, e)
        return False


def fetch_trade_log(local_path: str) -> bool:
    """Pull tensor_trade_log.json from the `data` branch. Startup recovery only."""
    return _fetch_log(TRADE_LOG_FILENAME, local_path, "tensor trade log")


def push_trade_log(local_path: str) -> bool:
    """Push tensor_trade_log.json to the `data` branch."""
    return _push_log(
        TRADE_LOG_FILENAME, local_path,
        "tensor trade log", "chore: hourly trade log sync",
    )


# ---------------------------------------------------------------------------
# Phemex TCT trade log — separate file, same data branch
# ---------------------------------------------------------------------------

def fetch_phemex_log(local_path: str) -> bool:
    """Pull phemex_trade_log.json from the `data` branch. Startup recovery only."""
    return _fetch_log(PHEMEX_LOG_FILENAME, local_path, "Phemex trade log")


def push_phemex_log(local_path: str) -> bool:
    """Push phemex_trade_log.json to the `data` branch."""
    return _push_log(
        PHEMEX_LOG_FILENAME, local_path,
        "Phemex trade log", "chore: phemex tct trade log sync",
    )
