# Project: HPB–TCT Trading System

This repository contains a Python-based quantitative trading system.

## Core Concepts (do not re-derive)
- **HPB** = High Probability Bias (directional context)
- **TCT** = Three Condition Trigger (entry qualification)
- **FVG** = Fair Value Gap
- **OB** = Order Block
- **RTZ** = Reaction Trading Zone
- **CVD** = Cumulative Volume Delta

Assume these concepts are well-defined and do not ask for basic explanations.

---

## Architecture Overview
- `trade_execution.py` — order placement & execution logic
- `risk_model.py` — position sizing, drawdown control
- `models/` — signal & probability models
- `src/` — scanners, helpers, shared utilities
- `server_*.py` — API / dashboard services

---

## Invariants (never violate without explicit instruction)
- No changes to live trading logic unless explicitly confirmed
- Prefer deterministic rules over ML unless instructed
- No silent behavior changes
- Preserve backward compatibility of signal outputs

---

## Coding Preferences
- Python 3.10+
- Clear, explicit logic > clever abstractions
- Avoid overengineering
- Comment intent, not syntax

---

## How to Assist Efficiently
- Only read files explicitly mentioned unless necessary
- Ask before scanning the entire repo
- Propose changes as diffs
- Default to minimal edits
- Prefer editing existing functions over introducing new abstractions

---

## What NOT to Do
- Do not refactor unrelated code
- Do not rename variables without reason
- Do not introduce new dependencies without approval
- Do not restate project goals unless asked

