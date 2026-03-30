import os
import sys
import json
import requests

URL = os.environ.get("VALIDATE_API_URL", "https://hpb-tct-server.onrender.com/api/validate")


def test_once() -> bool:
    """
    Make one request to /api/validate and verify _v2_shadow is present and
    structurally complete.

    Returns True when all checks pass, False on any validation failure.
    Network errors propagate to the caller.
    """
    r = requests.get(URL, timeout=10)
    r.raise_for_status()
    data = r.json()

    print("\n--- RESPONSE CHECK ---")

    # Core checks — API returns "Action" (legacy gate output), not "decision"
    has_action = "Action" in data
    has_shadow = "_v2_shadow" in data

    print(f"Action present:    {has_action}")
    print(f"_v2_shadow present: {has_shadow}")

    if not has_shadow:
        print("❌ ERROR: _v2_shadow missing")
        top_keys = list(data.keys())[:20]
        print(f"   Response keys: {top_keys}")
        return False

    shadow = data["_v2_shadow"]

    # Validate structure
    required_fields = ["status", "score", "model", "timeframe"]
    missing = [f for f in required_fields if f not in shadow]

    assert not missing, f"Shadow missing fields: {missing}"
    assert "decision" in shadow, "Shadow missing 'decision' field"
    assert shadow["decision"] == shadow.get("status"), (
        f"Alias contract broken: decision={shadow['decision']!r} != status={shadow.get('status')!r}"
    )
    print("✅ Shadow structure valid")
    print(f"   status={shadow['status']}  decision={shadow['decision']}  score={shadow.get('score')}")

    print("\nSample shadow output:")
    print(json.dumps(shadow, indent=2, default=str))

    return not missing


if __name__ == "__main__":
    all_passed = True
    for i in range(5):
        print(f"\n=== TEST {i+1} ===")
        try:
            passed = test_once()
            if not passed:
                all_passed = False
        except requests.RequestException as e:
            print(f"Network error: {e}")
            sys.exit(1)
        except AssertionError as e:
            print(f"Validation assertion failed: {e}")
            sys.exit(1)

    sys.exit(0 if all_passed else 1)
