# Emulator_main.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Query

from bybit_sim import BybitV5SimHTTP, load_ticks_from_json

app = FastAPI(title="Bybit Emulator (Bybit v5 compatible)")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# === FIXED, SIMPLE: files must exist in ./data/ with exactly these names ===
KLINE_15_FILE = "BTCUSDT_15m_test_data1.json"
KLINE_60_FILE = "BTCUSDT_1h_test_data_2.json"
# ========================================================================

KLINE_PATH_15 = DATA_DIR / KLINE_15_FILE
KLINE_PATH_60 = DATA_DIR / KLINE_60_FILE

print("✅ Loaded Emulator_main.py")
print(f"[EMULATOR] DATA_DIR={DATA_DIR}")
print(f"[EMULATOR] 15m={KLINE_PATH_15}")
print(f"[EMULATOR] 1h ={KLINE_PATH_60}")

if not KLINE_PATH_15.exists():
    raise FileNotFoundError(f"Missing 15m data file: {KLINE_PATH_15}")
if not KLINE_PATH_60.exists():
    raise FileNotFoundError(f"Missing 1h data file: {KLINE_PATH_60}")

candles_15 = load_ticks_from_json(str(KLINE_PATH_15))
candles_60 = load_ticks_from_json(str(KLINE_PATH_60))

# Base interval drives the emulator time (15m)
sim = BybitV5SimHTTP(
    candles_15=candles_15,
    candles_60=candles_60,
    symbol="BTCUSDT",
    base_interval="15",
    initial_balance=10_000.0,
    default_leverage=10.0,
    verbose=True,
)

@app.get("/health")
async def health():
    return {
        "ok": True,
        "data_dir": str(DATA_DIR),
        "file_15m": KLINE_15_FILE,
        "file_1h": KLINE_60_FILE,
        "len_15m": len(candles_15),
        "len_1h": len(candles_60),
        "cursor_forming": sim.cursor_forming,
        "finished": sim.finished,
    }

# Optional helper: reset emulator state
@app.post("/emulator/start")
async def emulator_start(payload: Dict[str, Any] = Body(default={})):
    initial = payload.get("initialBalance", None)
    sim.reset(initial_balance=initial)
    return {"retCode": 0, "retMsg": "OK", "result": {}, "retExtInfo": {}, "time": sim.now()}

@app.post("/emulator/finalize")
async def emulator_finalize(payload: Dict[str, Any] = Body(default={})):
    return sim.finalize()

# ----------------- Bybit v5 endpoints -----------------

@app.get("/v5/market/kline")
async def market_kline(
    category: str = Query("linear"),
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("15"),
    start: Optional[int] = Query(None),
    end: Optional[int] = Query(None),
    limit: int = Query(200),
):
    return sim.get_kline(category=category, symbol=symbol, interval=str(interval), start=start, end=end, limit=int(limit))

@app.get("/v5/account/wallet-balance")
async def wallet_balance(accountType: str = Query("UNIFIED")):
    return sim.get_wallet_balance(accountType=accountType)

@app.get("/v5/position/list")
async def position_list(category: str = Query("linear"), symbol: str = Query("BTCUSDT")):
    return sim.get_position_list(category=category, symbol=symbol)

@app.post("/v5/position/trading-stop")
async def trading_stop(payload: Dict[str, Any] = Body(...)):
    return sim.set_trading_stop(**payload)

@app.post("/v5/position/set-leverage")
async def set_leverage(payload: Dict[str, Any] = Body(...)):
    return sim.set_leverage(**payload)

@app.post("/v5/order/create")
async def order_create(payload: Dict[str, Any] = Body(...)):
    return sim.create_order(**payload)

@app.get("/v5/order/realtime")
async def order_realtime(
    category: str = Query("linear"),
    symbol: str = Query("BTCUSDT"),
    orderFilter: Optional[str] = Query(None),
):
    return sim.order_realtime(category=category, symbol=symbol, orderFilter=orderFilter)

@app.post("/v5/order/cancel")
async def order_cancel(payload: Dict[str, Any] = Body(...)):
    return sim.cancel_order(**payload)

@app.post("/v5/order/amend")
async def order_amend(payload: Dict[str, Any] = Body(...)):
    return sim.amend_order(**payload)
