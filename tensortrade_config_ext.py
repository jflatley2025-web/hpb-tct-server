import json
import traceback
from datetime import datetime
from tensortrade_env import HPB_TensorTrade_Env, HPBContextualReward
from High_Probability_Model_v17.validate_gates import validate_gates


TENSORTRADE_CONFIG = {
    "version": "HPB–TCT v17.3",
    "framework": "TensorTrade 1.0.3",
    "env_class": "HPB_TensorTrade_Env",
    "reward_class": "HPBContextualReward",
    "auto_train": True,
    "episodes": 10,
    "interval": "1H",
    "symbol": "BTC-USDT-SWAP",
    "window_size": 100,
    "plot_interval": 25,
}


def AUTO_INIT():
    try:
        symbol = TENSORTRADE_CONFIG["symbol"]
        interval = TENSORTRADE_CONFIG["interval"]
        window = TENSORTRADE_CONFIG["window_size"]

        env = HPB_TensorTrade_Env(symbol=symbol, interval=interval, window=window)
        env.build_feed()


        if TENSORTRADE_CONFIG["auto_train"]:
            env.auto_train(episodes=TENSORTRADE_CONFIG["episodes"])


        print("[AUTO_INIT] HPB–TCT environment ready.")
        return env

    except Exception as e:
        print(f"[AUTO_INIT ERROR] {e}")
        traceback.print_exc()
        return None


def snapshot_environment(env: HPB_TensorTrade_Env):
    try:
        live_price = env.fetch_live_price()
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": env.symbol,
            "interval": env.interval,
            "window": env.window,
            "price": live_price,
        }
        return json.dumps(snapshot, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
