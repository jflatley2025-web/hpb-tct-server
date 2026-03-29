import requests
import json

URL = "https://hpb-tct-server.onrender.com/api/validate"

def test_once():
    r = requests.get(URL, timeout=10)
    r.raise_for_status()
    data = r.json()

    print("\n--- RESPONSE CHECK ---")

    # Core checks
    has_decision = "decision" in data
    has_shadow = "_v2_shadow" in data

    print(f"decision present: {has_decision}")
    print(f"_v2_shadow present: {has_shadow}")

    if not has_shadow:
        print("❌ ERROR: _v2_shadow missing")
        return

    shadow = data["_v2_shadow"]

    # Validate structure
    required_fields = ["status", "score", "model", "timeframe"]
    missing = [f for f in required_fields if f not in shadow]

    if missing:
        print(f"❌ Shadow missing fields: {missing}")
    else:
        print("✅ Shadow structure valid")

    print("\nSample shadow output:")
    print(json.dumps(shadow, indent=2))


if __name__ == "__main__":
    for i in range(5):
        print(f"\n=== TEST {i+1} ===")
        try:
            test_once()
        except Exception as e:
            print(f"Request failed: {e}")
