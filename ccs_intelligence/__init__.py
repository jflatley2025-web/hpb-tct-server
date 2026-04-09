"""
ccs_intelligence — CCS v2 Read-Only Intelligence Layer
=======================================================
Consumes CCS v1 JSONL events and derives structural metrics.

ZERO execution impact. Read-only. No side effects.
No imports from engine modules. Stdlib only.
"""

import os

CCS_INTELLIGENCE_ENABLED = os.getenv(
    "CCS_INTELLIGENCE_ENABLED", "false"
).lower() == "true"

from ccs_intelligence.orchestrator import compute_ccs_metrics, get_health_summary  # noqa: E402
