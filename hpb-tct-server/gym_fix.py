# gym_fix.py
# ──────────────────────────────
# Makes TensorTrade use Gymnasium transparently.
import sys, types
import gymnasium as gym

# Expose gymnasium as "gym" to all imports
sys.modules["gym"] = gym

# Optional: warn once so you know it worked
print("[Patch] Gymnasium shim active -> TensorTrade now using gymnasium")
