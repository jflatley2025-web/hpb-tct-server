# Plan: Local Windows 10 Server — Mirror/Staging of Render Production

## Goal

Set up a local development server on Windows 10 that mirrors `https://hpb-tct-server.onrender.com`, enabling rapid iteration and easy deployment of changes to Render.

---

## Architecture Review

### Issue 1 — How to run the server locally on Windows 10?

**Problem**: The production server runs `uvicorn server_mexc:app --host 0.0.0.0 --port $PORT` on Render (Linux). We need an equivalent local setup on Windows 10.

**Options:**

**A (Recommended) — Native Python venv on Windows + uvicorn with `--reload`**
- Install Python 3.11.x from python.org
- Create a virtualenv, install `requirements.txt`
- Run `uvicorn server_mexc:app --host 127.0.0.1 --port 8000 --reload`
- `--reload` gives hot-reload on file save — ideal for dev
- Implementation effort: Low
- Risk: Low — most deps are pure-Python or have Windows wheels
- Maintenance: Low — matches production runtime closely

**B — WSL2 (Windows Subsystem for Linux)**
- More Linux-compatible (matches Render exactly)
- Implementation effort: Medium (WSL2 setup + Ubuntu install)
- Risk: Low — identical OS to production
- Maintenance: Medium — managing two OS layers

**C — Docker Desktop on Windows**
- Full containerized environment matching production
- Implementation effort: High (Dockerfile needed, Docker Desktop install, ~4GB disk)
- Risk: Low — production parity is excellent
- Maintenance: High — Docker Desktop resource usage, Dockerfile upkeep

**Recommendation: Option A.** FastAPI + uvicorn run natively on Windows without issues. All dependencies in `requirements.txt` have Windows wheels. This gives you the fastest feedback loop with `--reload` and no extra infrastructure. If you hit Windows-specific issues later, WSL2 is an easy escalation path.

---

### Issue 2 — How to handle environment variables locally?

**Problem**: Render injects env vars (`TELEGRAM_BOT_TOKEN`, `PHEMEX_API_KEY`, `PHEMEX_API_SECRET`, etc.) via its dashboard. Locally, we need a way to provide these without hardcoding secrets.

**Options:**

**A (Recommended) — `.env.local` file + `python-dotenv` loader**
- Create `.env.local` (already in `.gitignore`) with all required env vars
- Add a small dotenv loader at the top of `server_mexc.py` that only fires when `RENDER` env var is absent
- Implementation effort: Low (3-5 lines of code)
- Risk: None — `.env.local` is gitignored, no secret leakage
- Maintenance: Low — one file to keep in sync with Render env vars

**B — Windows system environment variables**
- Set env vars via System Properties → Environment Variables
- Implementation effort: Low
- Risk: Medium — easy to forget, hard to version-track, pollutes system env
- Maintenance: High — manual per-machine setup

**C — PowerShell script that sets env vars then launches uvicorn**
- `run_local.ps1` exports vars inline then runs the server
- Implementation effort: Low
- Risk: Medium — script contains secrets in plaintext (must be gitignored)
- Maintenance: Low

**Recommendation: Option A.** `.env.local` + `python-dotenv` is the industry standard. It's already gitignored. We'll also add a `run_local.bat` / `run_local.ps1` convenience script that loads dotenv automatically.

---

### Issue 3 — ChromaDB path differences (local vs Render)

**Problem**: Render uses `/opt/render/project/chroma_db/` for persistent ChromaDB storage. This path doesn't exist on Windows.

**Options:**

**A (Recommended) — Environment-aware path with `CHROMA_DB_DIR` env var**
- Default to `./data/chroma_db/` locally
- On Render, `PHEMEX_TRADE_LOG_DIR` already points to `/opt/render/project/chroma_db`
- Add `CHROMA_DB_DIR` to `.env.local` pointing to a local directory
- Implementation effort: Low (one `os.getenv()` with fallback)
- Risk: None
- Maintenance: Low

**B — Hardcode platform detection (`if sys.platform == 'win32'`)**
- Implementation effort: Low
- Risk: Medium — brittle, doesn't work in WSL
- Maintenance: Medium

**Recommendation: Option A.** Env-var-driven paths are explicit and work everywhere.

---

### Issue 4 — Deployment workflow: local changes → Render

**Problem**: You need a smooth workflow to test locally, then deploy to Render. Currently `autoDeploy: true` in `render.yaml` means every push to `master` triggers a Render deploy.

**Options:**

**A (Recommended) — Git branch workflow with PR-based deploys**
- Develop on feature branches (e.g., `feature/xyz`)
- Test locally with `uvicorn --reload`
- Push branch, create PR to `master`
- Merge PR → auto-deploy to Render
- Implementation effort: None (already in place via `autoDeploy: true`)
- Risk: Low — standard Git workflow
- Maintenance: None

**B — Render Preview Environments**
- Render can spin up preview instances per PR (paid feature on some plans)
- Implementation effort: Low (config change in `render.yaml`)
- Risk: Low — but may not be available on free plan
- Maintenance: Low

**C — Manual deploy trigger via Render dashboard**
- Disable `autoDeploy`, manually trigger deploys from Render UI
- Implementation effort: Low
- Risk: Medium — easy to forget to deploy
- Maintenance: Medium

**Recommendation: Option A.** The current `autoDeploy: true` on `master` is already the right setup. We just need a clear local dev workflow documented in a launcher script. Push to `master` = deploy to production.

---

## Implementation Steps

### Step 1 — Create `.env.local` template

Create `.env.local.example` (committed to repo as a reference) with placeholder values:

```env
# Local development environment variables
# Copy this to .env.local and fill in real values
# .env.local is gitignored — never commit secrets

PORT=8000
LOG_LEVEL=debug
TELEGRAM_BOT_TOKEN=your-telegram-bot-token-here
API_URL=http://127.0.0.1:8000
AUTO_TRAIN_DEFAULT_EPISODES=5
PHEMEX_API_KEY=your-phemex-api-key-here
PHEMEX_API_SECRET=your-phemex-api-secret-here
PHEMEX_TRADE_LOG_DIR=./data/chroma_db
CHROMA_DB_DIR=./data/chroma_db

# Optional: set to 'true' to skip heavy model loading during dev
SKIP_ML_MODELS=false
```

### Step 2 — Add dotenv loading to `server_mexc.py`

Add a conditional dotenv loader near the top of `server_mexc.py`:

```python
import os
if not os.getenv("RENDER"):
    try:
        from dotenv import load_dotenv
        load_dotenv(".env.local")
    except ImportError:
        pass
```

This fires only when NOT running on Render (Render sets the `RENDER` env var automatically).

### Step 3 — Create `run_local.bat` (Windows launcher)

```bat
@echo off
echo === HPB-TCT Local Server ===
echo.

REM Activate venv if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the server with hot-reload
uvicorn server_mexc:app --host 127.0.0.1 --port 8000 --reload
```

### Step 4 — Create `run_local.ps1` (PowerShell alternative)

```powershell
Write-Host "=== HPB-TCT Local Server ===" -ForegroundColor Cyan

# Activate venv if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
}

# Run with hot-reload
uvicorn server_mexc:app --host 127.0.0.1 --port 8000 --reload
```

### Step 5 — Create `setup_local.bat` (one-time setup script)

```bat
@echo off
echo === HPB-TCT Local Environment Setup ===
echo.

REM Check Python version
python --version

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

REM Pre-download sentence-transformers model (matches Render build)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

REM Create local data directories
if not exist data\chroma_db mkdir data\chroma_db
if not exist logs mkdir logs

REM Copy env template if .env.local doesn't exist
if not exist .env.local (
    copy .env.local.example .env.local
    echo.
    echo *** IMPORTANT: Edit .env.local with your real API keys ***
)

echo.
echo Setup complete! Run 'run_local.bat' to start the server.
```

### Step 6 — Update `.gitignore`

Add entries for local dev artifacts:

```
# Local development
data/
run_local.bat
run_local.ps1
*.bat
!setup_local.bat
```

Wait — actually, `run_local.bat` and `run_local.ps1` should be committed (they're dev tooling, no secrets). Only `.env.local` needs to stay gitignored (already is). Let me revise:

Add to `.gitignore`:
```
data/
```

### Step 7 — Ensure `python-dotenv` is in `requirements.txt`

Check if `python-dotenv` is already listed. If not, add it.

### Step 8 — Create `LOCALDEV.md` with setup instructions

Short reference doc for the local setup workflow:
- Prerequisites (Python 3.11.x, Git)
- One-time setup steps
- Daily workflow (run server, test, commit, push)
- Deploy-to-Render workflow

---

## Files to Create / Modify

| File | Action | What changes |
|------|--------|-------------|
| `.env.local.example` | **CREATE** | Template with all env var placeholders |
| `setup_local.bat` | **CREATE** | One-time Windows setup script |
| `run_local.bat` | **CREATE** | Windows server launcher |
| `run_local.ps1` | **CREATE** | PowerShell server launcher |
| `server_mexc.py` | **MODIFY** | Add dotenv loader (3 lines near top) |
| `requirements.txt` | **MODIFY** | Add `python-dotenv` if missing |
| `.gitignore` | **MODIFY** | Add `data/` directory |
| `LOCALDEV.md` | **CREATE** | Local development setup guide |

---

## Deployment Workflow Summary

```
┌─────────────────────┐     git push     ┌──────────────────┐
│  Windows 10 Local   │ ──────────────→  │  GitHub (master)  │
│  127.0.0.1:8000     │                  │                   │
│  uvicorn --reload   │                  │  autoDeploy: true │
└─────────────────────┘                  └────────┬─────────┘
                                                  │
                                                  │ auto-deploy
                                                  ▼
                                         ┌──────────────────┐
                                         │  Render.com       │
                                         │  Production       │
                                         │  hpb-tct-server   │
                                         │  .onrender.com    │
                                         └──────────────────┘
```

**Daily workflow:**
1. `run_local.bat` → server starts at `http://127.0.0.1:8000`
2. Edit code → uvicorn hot-reloads automatically
3. Test at `http://127.0.0.1:8000/dashboard`, `/status`, etc.
4. `git add` + `git commit` + `git push origin master`
5. Render auto-deploys within ~2 minutes
6. Verify at `https://hpb-tct-server.onrender.com`

---

## Potential Windows Gotchas

1. **`sentence-transformers` download**: First run downloads ~90MB model. `setup_local.bat` handles this.
2. **ChromaDB on Windows**: Works natively with SQLite backend. No issues expected.
3. **Port conflicts**: If port 8000 is taken, change `PORT` in `.env.local`.
4. **Long path names**: Enable Windows long paths if you hit path-too-long errors: `git config --system core.longpaths true`
5. **`stable-baselines3` / `gymnasium`**: These have Windows wheels — should install cleanly via pip.

---

## Questions Before Proceeding

**Q1 (Issue 1 — Runtime approach):** Are you OK with **Option A** (native Python venv on Windows)? Or would you prefer **Option B** (WSL2) for closer Linux parity?

**Q2 (Issue 2 — Env vars):** Does **Option A** (`.env.local` + `python-dotenv`) work for you? Do you already have your API keys (Telegram, Phemex) available to paste in?

**Q3 (Issue 3 — ChromaDB):** Are you OK with `./data/chroma_db/` as the local storage path, or do you want it somewhere else?

**Q4 (Issue 4 — Deploy flow):** The current `autoDeploy: true` on `master` means every push to master goes live. Do you want to keep this, or would you prefer to deploy only from a specific branch (e.g., `production`) so `master` is your staging branch?
