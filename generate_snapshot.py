import os
import platform
import subprocess
import json
from datetime import datetime


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "Not available"


snapshot = {}

# ─────────────────────────────
# SYSTEM INFO
# ─────────────────────────────
snapshot["system"] = {
    "os": platform.system(),
    "os_version": platform.version(),
    "architecture": platform.machine(),
    "cpu": platform.processor()
}

# ─────────────────────────────
# PYTHON
# ─────────────────────────────
snapshot["python"] = {
    "version": run_cmd("python --version"),
    "path": run_cmd("where python" if os.name == "nt" else "which python"),
    "packages": run_cmd("pip list")
}

# ─────────────────────────────
# DOCKER
# ─────────────────────────────
docker_version = run_cmd("docker --version")
snapshot["docker"] = {
    "installed": docker_version if "Not available" not in docker_version else "Not installed",
    "containers": run_cmd("docker ps") if "Not installed" not in docker_version else "N/A",
    "images": run_cmd("docker images") if "Not installed" not in docker_version else "N/A"
}

# ─────────────────────────────
# POSTGRES
# ─────────────────────────────
psql_version = run_cmd("psql --version")
snapshot["postgres"] = {
    "version": psql_version if "Not available" not in psql_version else "Not installed",
    "status": run_cmd("pg_isready") if "Not installed" not in psql_version else "N/A"
}

# ─────────────────────────────
# NODE (optional)
# ─────────────────────────────
snapshot["node"] = {
    "node_version": run_cmd("node -v"),
    "npm_version": run_cmd("npm -v")
}

# ─────────────────────────────
# PROJECT STRUCTURE
# ─────────────────────────────
snapshot["project_structure"] = run_cmd("tree /F" if os.name == "nt" else "ls -R")

# ─────────────────────────────
# TIMESTAMP
# ─────────────────────────────
snapshot["generated_at"] = datetime.utcnow().isoformat()

# ─────────────────────────────
# WRITE MARKDOWN (NO EMOJIS)
# ─────────────────────────────
md = f"""
# Development Environment Snapshot

## System
{snapshot['system']}

## Python
{snapshot['python']['version']}

Path:
{snapshot['python']['path']}

Packages:
{snapshot['python']['packages']}

## Docker
{snapshot['docker']['installed']}

Containers:
{snapshot['docker']['containers']}

Images:
{snapshot['docker']['images']}

## PostgreSQL
Version: {snapshot['postgres']['version']}
Status: {snapshot['postgres']['status']}

## Node
Node: {snapshot['node']['node_version']}
NPM: {snapshot['node']['npm_version']}

## Project Structure
{snapshot['project_structure']}

## Generated
{snapshot['generated_at']}
"""

with open("dev_environment.md", "w", encoding="utf-8") as f:
    f.write(md)

# ─────────────────────────────
# AI CONTEXT JSON
# ─────────────────────────────
ai_context = {
    "environment": {
        "os": snapshot["system"]["os"],
        "python_version": snapshot["python"]["version"],
        "uses_docker": "Docker version" in snapshot["docker"]["installed"],
        "uses_postgres": "psql" in snapshot["postgres"]["version"]
    },
    "services": {
        "docker": snapshot["docker"]["installed"],
        "postgres": snapshot["postgres"]["status"]
    },
    "project_type": "HPB_TCT_trading_bot",
    "generated_at": snapshot["generated_at"]
}

with open("ai_context.json", "w", encoding="utf-8") as f:
    json.dump(ai_context, f, indent=4)

print("Snapshot generated: dev_environment.md + ai_context.json")