from fastapi import FastAPI, Query
from bybit_sim import BybitV5SimHTTP, load_ticks_from_json

app = FastAPI()

candles = load_ticks_from_json("data/BTCUSDT_15m_2020-2025.json")
sim = BybitV5SimHTTP(candles)

@app.post("/emulator/start")
def start():
    sim.i = 0
    sim.window_start = 1
    sim.current = sim.candles[0] if sim.candles else None

    sim.trades.clear()
    sim.position = None

    sim.balance = 10_000  # falls du wirklich reset willst, sonst weglassen
    sim.equity_curve = [sim.balance]

    sim.finished = False
    sim._finalized = False
    return {"status": "started"}

@app.get("/v5/market/kline")
async def get_kline(
    category: str = "linear",
    symbol: str = "BTCUSDT",
    interval: str = "15",
    limit: int = Query(200, ge=1, le=1000),
    start: int | None = None,
    end: int | None = None,
):
    return await sim.get_kline(limit=limit, start=start, end=end)

@app.get("/v5/account/wallet-balance")
async def wallet_balance():
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {
                    "totalEquity": str(sim.equity()),          # equity inkl. unrealized
                    "totalWalletBalance": str(sim.balance),   # realized
                }
            ]
        }
    }

@app.post("/v5/order/create")
async def place_order(payload: dict):
    return await sim.place_order(**payload)
