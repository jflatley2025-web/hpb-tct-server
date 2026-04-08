"""
reporting/ — HPB Daily Logging System
======================================
Non-blocking, additive, shadow-only reporting pipeline.

Modules:
  task_registry.py        — static source-of-truth for active/completed/queued tasks
  report_exporters.py     — exporter boundary (local JSON + MD; Google Docs later)
  daily_report_builder.py — collects live telemetry and writes daily reports

Note: intentionally NOT named 'logging/' to avoid shadowing stdlib logging.
"""
