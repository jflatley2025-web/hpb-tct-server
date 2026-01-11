"""
server.py
────────────────────────────────────────────
HPM–TCT v19 RIG EXTENDED Server (Phase 9)
Integrates: TensorTrade v1.0.3 Env + HPB Gate Logic + Risk Mgmt + P03 Confluence + Trade Execution
────────────────────────────────────────────
"""

import gym_fix  # must be first
import os, json, traceback, asyncio, subprocess, time, threading
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# local imports
from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG, snapshot_environment
from High_Probability_Model_v17_RIG.validate_gates import validate_gates
from risk_model import compute_risk_profile
from p03_model import compute_p03_confluence

import pandas as pd, numpy as np, httpx, plotly.graph_objs as go
from apscheduler.schedulers.background import BackgroundScheduler

# ───────────────────────────────────────────────
#  Init
# ───────────────────────────────────────────────
app = FastAPI(title="HPM–TCT v19 RIG EXTENDED + P03 + EXEC", version="9.0")
ENV = AUTO_INIT()
latest_dashboard_html = "<h3>Initializing dashboard...</h3>"

# ───────────────────────────────────────────────
#  Execution State
# ───────────────────────────────────────────────
trade_state = {
    "position": "FLAT",
    "entry_price": None,
    "pnl": 0.0,
    "trades": [],
}

# ───────────────────────────────────────────────
#  Utility Data Fetchers
# ───────────────────────────────────────────────
async def fetch_algo_context(symbol="BTC-USDT-SWAP", interval="1H"):
    try:
        gates = validate_gates({"symbol": symbol, "interval": interval})
        return gates
    except Exception as e:
        return {"error": str(e)}

async def fetch_ohlc(symbol="BTC-USDT", interval="1H", limit=200):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}-SWAP&bar={interval}&limit={limit}"
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
    data = r.json().get("data", [])
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["ts","o","h","l","c","v"]
    df = df.astype({"o":float,"h":float,"l":float,"c":float,"v":float})
    df["ts"]=pd.to_datetime(pd.to_numeric(df["ts"]),unit="ms")
    return df.sort_values("ts")

# ───────────────────────────────────────────────
#  Risk + Confluence + Execution Core
# ───────────────────────────────────────────────
async def get_signal_bundle():
    algo = await fetch_algo_context("BTC-USDT-SWAP","1H")
    risk = await compute_risk_profile(algo)
    p03  = await compute_p03_confluence(risk)
    return {"algo":algo,"risk":risk,"p03":p03}

# Decision logic
def evaluate_trade_decision(bundle):
    bias = bundle["p03"].get("execution_bias","NEUTRAL")
    rr   = bundle["p03"].get("expected_reward_ratio",1.0)
    conf = bundle["p03"]["confidence_matrix"]["Bullish_Transition"]
    price = bundle["risk"].get("current_price",0)
    vol = bundle["risk"].get("volatility_score",0)

    # thresholds
    if rr < 1.0 or vol>0.015: return "HOLD"
    if bias=="LONG" and conf>0.6: return "BUY"
    if bias=="SHORT" and conf>0.6: return "SELL"
    return "HOLD"

# ───────────────────────────────────────────────
#  Trade Execution Simulation
# ───────────────────────────────────────────────
@app.post("/execute")
async def execute_trade():
    """
    Phase 9 – Simulated Trade Execution Engine
    Executes a synthetic order based on confluence + risk bias.
    """
    global trade_state
    try:
        bundle = await get_signal_bundle()
        decision = evaluate_trade_decision(bundle)
        price = bundle["risk"].get("current_price",0)

        if decision=="BUY" and trade_state["position"]!="LONG":
            trade_state["position"]="LONG"
            trade_state["entry_price"]=price
            trade_state["trades"].append(
                {"side":"BUY","price":price,"time":datetime.utcnow().isoformat()}
            )
        elif decision=="SELL" and trade_state["position"]!="SHORT":
            trade_state["position"]="SHORT"
            trade_state["entry_price"]=price
            trade_state["trades"].append(
                {"side":"SELL","price":price,"time":datetime.utcnow().isoformat()}
            )
        elif decision=="HOLD" and trade_state["position"]!="FLAT":
            # close position
            entry=trade_state["entry_price"] or price
            pnl = (price-entry)*(1 if trade_state["position"]=="LONG" else -1)
            trade_state["pnl"]+=pnl
            trade_state["trades"].append(
                {"side":"CLOSE","price":price,"pnl":round(pnl,2),"time":datetime.utcnow().isoformat()}
            )
            trade_state["position"]="FLAT"
            trade_state["entry_price"]=None

        summary = {
            "decision":decision,
            "position":trade_state["position"],
            "entry_price":trade_state["entry_price"],
            "pnl_total":round(trade_state["pnl"],2),
            "trades":trade_state["trades"][-5:],  # recent
            "risk":bundle["risk"],
            "p03":bundle["p03"],
        }
        print(f"[EXEC] {decision} @ {price} | pos={trade_state['position']} | pnl={trade_state['pnl']:.2f}")
        return JSONResponse(summary)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error":str(e)},status_code=500)

# ───────────────────────────────────────────────
#  Dashboard (includes P03 & Execution)
# ───────────────────────────────────────────────
def plot_range_chart(df,title):
    if df.empty:
        return go.Figure().update_layout(title=f"{title} – No data",template="plotly_dark")
    fig=go.Figure(go.Candlestick(x=df["ts"],open=df["o"],high=df["h"],low=df["l"],close=df["c"]))
    fig.update_layout(title=title,template="plotly_dark",height=600,xaxis_rangeslider_visible=False)
    return fig

async def generate_dashboard():
    global latest_dashboard_html
    try:
        bundle = await get_signal_bundle()
        htf,mtf,ltf = await fetch_ohlc("BTC-USDT","4H"),await fetch_ohlc("BTC-USDT","1H"),await fetch_ohlc("BTC-USDT","15m")
        htf_fig,mtf_fig,ltf_fig = plot_range_chart(htf,"HTF 4H"),plot_range_chart(mtf,"MTF 1H"),plot_range_chart(ltf,"LTF 15m")
        p03=bundle["p03"]; risk=bundle["risk"]

        html=(f"<h2>HPM–TCT v19 RIG EXTENDED Dashboard (Phase 9)</h2>"
              f"<p>Updated @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>"
              f"<p><b>Signal:</b> {risk.get('signal')} | "
              f"<b>Risk:</b> {risk.get('risk_score')} | "
              f"<b>Vol:</b> {risk.get('volatility_score')} | "
              f"<b>Price:</b> {risk.get('current_price')}</p>"
              f"<h3>🧠 P03 Confluence</h3>"
              f"<pre>{json.dumps(p03.get('confidence_matrix',{}),indent=2)}</pre>"
              f"<p><b>Phase:</b> {p03.get('phase')} | <b>Bias:</b> {p03.get('execution_bias')} | "
              f"<b>Expected RR:</b> {p03.get('expected_reward_ratio')}</p>"
              f"<h3>⚙️ Execution State</h3>"
              f"<p><b>Position:</b> {trade_state['position']} | "
              f"<b>PNL:</b> {round(trade_state['pnl'],2)}</p>"
              + htf_fig.to_html(full_html=False,include_plotlyjs="cdn")
              + mtf_fig.to_html(full_html=False,include_plotlyjs=False)
              + ltf_fig.to_html(full_html=False,include_plotlyjs=False))
        latest_dashboard_html=html
    except Exception as e:
        latest_dashboard_html=f"<h3>Error generating dashboard:</h3><pre>{e}</pre>"

@app.get("/dashboard",response_class=HTMLResponse)
async def get_dashboard(): return latest_dashboard_html

# ───────────────────────────────────────────────
#  Keepalive & Startup
# ───────────────────────────────────────────────
def touch_keepalive():
    while True:
        try:
            f="/tmp/render_keepalive.flag"
            subprocess.run(["touch",f],check=True)
            print(f"[KEEPALIVE] Touched {f} @ {datetime.utcnow().isoformat()}")
        except Exception as e: print(f"[KEEPALIVE ERR] {e}")
        time.sleep(600)
threading.Thread(target=touch_keepalive,daemon=True).start()

@app.on_event("startup")
async def startup_event():
    print("[HPB] Initializing dashboard + scheduler (Phase 9)")
    await generate_dashboard()
    sched=BackgroundScheduler()
    sched.add_job(lambda: asyncio.run(generate_dashboard()),"interval",hours=24)
    sched.start()

# ───────────────────────────────────────────────
#  Run
# ───────────────────────────────────────────────
if __name__=="__main__":
    import uvicorn
    uvicorn.run("server:app",host="0.0.0.0",port=int(os.environ.get("PORT",8080)),reload=False)
