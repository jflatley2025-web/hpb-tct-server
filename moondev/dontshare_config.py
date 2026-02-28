"""
Phemex API credentials — loaded from environment variables.
Set PHEMEX_API_KEY and PHEMEX_API_SECRET in your environment before running.
"""
import os

xP_hmv_KEY = os.getenv("PHEMEX_API_KEY", "")
xP_hmv_SECRET = os.getenv("PHEMEX_API_SECRET", "")

if not xP_hmv_KEY or not xP_hmv_SECRET:
    print("[CONFIG] WARNING: PHEMEX_API_KEY or PHEMEX_API_SECRET not set in environment")
