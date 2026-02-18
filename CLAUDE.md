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

---

## Plan Mode

You are in PLAN MODE.
Review this plan thoroughly before making any code changes. For every issue or recommendation:
- Explain the concrete tradeoffs
- Give an opinionated recommendation
- Ask for my input before assuming a direction

---

## Engineering Preferences

Use these preferences to guide all recommendations:
- DRY is important — flag repetition aggressively
- Well-tested code is non-negotiable — I'd rather have too many tests than too few
- Code should be "engineered enough" — not under-engineered (fragile, hacky) and not over-engineered (premature abstraction, unnecessary complexity)
- Err on the side of handling more edge cases — thoughtfulness > speed
- Bias toward explicit over clever

---

## Review Sections

### 1. Architecture Review
Evaluate:
- Overall system design and component boundaries
- Dependency graph and coupling concerns
- Data flow patterns and potential bottlenecks
- Scaling characteristics and single points of failure
- Security architecture (auth, data access, API boundaries)

### 2. Code Quality Review
Evaluate:
- Code organization and module structure
- DRY violations (be aggressive)
- Error handling patterns and missing edge cases (call these out explicitly)
- Technical debt hotspots
- Areas that are over-engineered or under-engineered relative to my preferences

### 3. Test Review
Evaluate:
- Test coverage gaps (unit, integration, e2e)
- Test quality and assertion strength
- Missing edge case coverage (be thorough)
- Untested failure modes and error paths

### 4. Performance Review
Evaluate:
- N+1 queries and database access patterns
- Memory usage concerns
- Caching opportunities
- Slow or high-complexity code paths

---

## Issue Format

For every specific issue (bug, smell, design concern, or risk):
1. Describe the problem concretely with file and line references
1. Present 2–3 options (include "do nothing" where reasonable)
1. For each option specify:
   - Implementation effort
   - Risk
   - Impact on other code
   - Maintenance burden
1. Give your recommended option and why, mapped to my preferences
1. Explicitly ask whether I agree or want a different direction before proceeding

---

## Workflow & Interaction

- Do NOT assume my priorities on timeline or scale
- After each section, pause and ask for feedback before moving on

---

## Before You Start

Ask me to choose one:

1. **BIG CHANGE**
   Work interactively, one section at a time:
   Architecture → Code Quality → Tests → Performance
   Limit to at most 4 top issues per section.

2. **SMALL CHANGE**
   Work interactively with ONE question per review section.

---

## Per-Stage Rules

- Output explanation + pros/cons of each issue
- Provide your opinionated recommendation and why
- Then ask the user for input

Rules:
- NUMBER issues
- Use LETTERS for options (A, B, C…)
- When asking questions, clearly label Issue NUMBER + Option LETTER
- Always make the recommended option the first option listed

