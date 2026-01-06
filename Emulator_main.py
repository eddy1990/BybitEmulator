# Emulator_main.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

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


# ----------------- Live UI / Debug endpoints -----------------

@app.get("/emulator/state")
async def emulator_state(
    bars: int = Query(600, ge=50, le=5000),
    interval: str = Query("15"),
):
    # Snapshot used by the live dashboard (candles + trades + equity)
    return JSONResponse(sim.export_state(bars=bars, interval=interval))

@app.get("/emulator/ui", response_class=HTMLResponse)
async def emulator_ui():
    # Simple live dashboard (no extra dependencies; uses Plotly.js CDN)
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Emulator Live Plot</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ margin:0; font-family: Arial, sans-serif; background:#111; color:#ddd; }}
    .topbar {{ padding:10px 14px; background:#1b1b1b; position:sticky; top:0; z-index:10; }}
    .row {{ display:flex; gap:10px; padding:10px; }}
    .card {{ background:#1b1b1b; border-radius:10px; padding:10px; flex:1; }}
    #price {{ height: 62vh; }}
    #equity {{ height: 28vh; }}
    .kv {{ display:grid; grid-template-columns: 140px 1fr; gap:6px 10px; font-size: 13px; }}
    .muted {{ color:#aaa; }}
    input {{ width:80px; }}
    button {{ padding:6px 10px; border-radius:8px; border:0; background:#2b2b2b; color:#ddd; cursor:pointer; }}
    button:hover {{ background:#3a3a3a; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
      <div><b>Emulator Live</b> <span class="muted">/emulator/ui</span></div>
      <div class="muted">Symbol: <span id="sym">-</span> | Cursor: <span id="cur">-</span> | Version: <span id="ver">-</span></div>
      <div style="margin-left:auto; display:flex; gap:8px; align-items:center;">
        <span class="muted">Bars</span>
        <input id="bars" type="number" value="600" min="50" max="5000"/>
        <button id="btn">Apply</button>
      </div>
    </div>
  </div>

  <div class="row">
    <div class="card" style="flex:3;">
      <div id="price"></div>
      <div id="equity"></div>
    </div>
    <div class="card" style="flex:1; min-width:300px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <b>Status</b>
        <span class="muted" id="ts"></span>
      </div>
      <hr style="border:0; border-top:1px solid #2a2a2a; margin:10px 0;"/>
      <div class="kv">
        <div class="muted">Wallet</div><div id="wallet">-</div>
        <div class="muted">Equity</div><div id="eq">-</div>
        <div class="muted">uPnL</div><div id="upnl">-</div>
        <div class="muted">Position</div><div id="pos">-</div>
        <div class="muted">Open orders</div><div id="ords">-</div>
        <div class="muted">Trades</div><div id="trcnt">-</div>
        <div class="muted">Last trade</div><div id="last">-</div>
      </div>
      <hr style="border:0; border-top:1px solid #2a2a2a; margin:10px 0;"/>
      <div class="muted" style="font-size:12px;">
        Updates every 800ms via <code>/emulator/state</code> (polling).  
        Entry markers: ▲ (long) / ▼ (short).  
        TP marker: ● green. SL marker: ● red.
      </div>
    </div>
  </div>

<script>
let lastVersion = null;
let bars = 600;

function fmt(x) {{
  if (x === null || x === undefined) return "-";
  const n = Number(x);
  if (!isFinite(n)) return String(x);
  return n.toFixed(2);
}}

function buildPriceFigure(state) {{
  const c = state.candles || [];
  const x = c.map(d => new Date(d.ts));
  const open = c.map(d => d.open);
  const high = c.map(d => d.high);
  const low  = c.map(d => d.low);
  const close= c.map(d => d.close);

  const traces = [
    {{
      type: "candlestick",
      x, open, high, low, close,
      name: "Price",
      increasing: {{ line: {{ color: "lime" }}, fillcolor: "lime" }},
      decreasing: {{ line: {{ color: "red" }}, fillcolor: "red" }},
    }}
  ];

  // closed trades
  const trades = state.trades || [];
  const longEntriesX=[], longEntriesY=[], shortEntriesX=[], shortEntriesY=[];
  const tpX=[], tpY=[], slX=[], slY=[], beX=[], beY=[];
  const openT = state.open_trade;

  function pushEntry(t) {{
    const ts = t.entry_ts ?? t.entryTs ?? null;
    if (!ts) return;
    const side = (t.side || t.type || "").toLowerCase();
    if (side.includes("long")) {{ longEntriesX.push(new Date(ts)); longEntriesY.push(t.entry); }}
    else if (side.includes("short")) {{ shortEntriesX.push(new Date(ts)); shortEntriesY.push(t.entry); }}
  }}

  function pushExit(t) {{
    const ts = t.exit_ts ?? t.exitTs ?? null;
    if (!ts) return;
    const r = (t.reason || "").toUpperCase();
    if (r.includes("TP")) {{ tpX.push(new Date(ts)); tpY.push(t.exit); }}
    else if (r.includes("SL")) {{ slX.push(new Date(ts)); slY.push(t.exit); }}
    else if (r.includes("BE")) {{ beX.push(new Date(ts)); beY.push(t.exit); }}
  }}

  for (const t of trades) {{
    pushEntry(t);
    pushExit(t);
  }}
  if (openT) {{
    pushEntry(openT);
  }}

  if (longEntriesX.length) {{
    traces.push({{
      type: "scattergl", mode: "markers",
      x: longEntriesX, y: longEntriesY,
      name: "BUY",
      marker: {{ symbol: "triangle-up", size: 11, line: {{ width: 1, color: "black" }}, color: "green" }},
    }});
  }}
  if (shortEntriesX.length) {{
    traces.push({{
      type: "scattergl", mode: "markers",
      x: shortEntriesX, y: shortEntriesY,
      name: "SELL",
      marker: {{ symbol: "triangle-down", size: 11, line: {{ width: 1, color: "black" }}, color: "orange" }},
    }});
  }}
  if (tpX.length) {{
    traces.push({{
      type: "scattergl", mode: "markers",
      x: tpX, y: tpY,
      name: "TP",
      marker: {{ symbol: "circle", size: 14, line: {{ width: 2, color: "black" }}, color: "green" }},
    }});
  }}
  if (slX.length) {{
    traces.push({{
      type: "scattergl", mode: "markers",
      x: slX, y: slY,
      name: "SL",
      marker: {{ symbol: "circle", size: 14, line: {{ width: 2, color: "black" }}, color: "red" }},
    }});
  }}
  if (beX.length) {{
    traces.push({{
      type: "scattergl", mode: "markers",
      x: beX, y: beY,
      name: "BE",
      marker: {{ symbol: "square", size: 12, line: {{ width: 1, color: "black" }}, color: "black" }},
    }});
  }}

  // show current TP/SL as horizontal lines if position open
  const shapes = [];
  if (state.position && x.length) {{
    const x0 = x[0], x1 = x[x.length-1];
    const tp = state.position.tp;
    const sl = state.position.sl;
    if (tp !== null && tp !== undefined) {{
      shapes.push({{
        type:"line", xref:"x", yref:"y",
        x0, x1, y0: tp, y1: tp,
        line: {{ width: 1, dash: "dot", color: "rgba(0,255,0,0.6)" }}
      }});
    }}
    if (sl !== null && sl !== undefined) {{
      shapes.push({{
        type:"line", xref:"x", yref:"y",
        x0, x1, y0: sl, y1: sl,
        line: {{ width: 1, dash: "dot", color: "rgba(255,0,0,0.6)" }}
      }});
    }}
  }}

  const layout = {{
    paper_bgcolor:"#1b1b1b",
    plot_bgcolor:"#1b1b1b",
    font: {{ color:"#ddd" }},
    margin: {{ l:40, r:20, t:30, b:40 }},
    xaxis: {{ rangeslider: {{ visible:true }}, type:"date" }},
    yaxis: {{ fixedrange:false }},
    legend: {{ orientation:"h" }},
    dragmode: "pan",
    hovermode: "x unified",
    shapes: shapes
  }};
  const config = {{ scrollZoom:true, displaylogo:false, doubleClick:"reset" }};
  Plotly.react("price", traces, layout, config);
}}

function buildEquityFigure(state) {{
  const e = state.equity_curve || [];
  const x = e.map(d => new Date(d.ts));
  const y = e.map(d => d.equity);
  const trace = {{
    type:"scatter", mode:"lines",
    x, y, name:"Equity"
  }};
  const layout = {{
    paper_bgcolor:"#1b1b1b",
    plot_bgcolor:"#1b1b1b",
    font: {{ color:"#ddd" }},
    margin: {{ l:40, r:20, t:10, b:30 }},
    xaxis: {{ type:"date" }},
    yaxis: {{ fixedrange:false }},
    showlegend:false
  }};
  const config = {{ scrollZoom:true, displaylogo:false, doubleClick:"reset" }};
  Plotly.react("equity", [trace], layout, config);
}}

async function tick() {{
  try {{
    const r = await fetch(`/emulator/state?bars=${{bars}}&interval=15`);
    const s = await r.json();
    document.getElementById("sym").textContent = s.symbol ?? "-";
    document.getElementById("cur").textContent = s.cursor_forming ?? "-";
    document.getElementById("ver").textContent = s.version ?? "-";

    document.getElementById("wallet").textContent = fmt(s.wallet_balance);
    document.getElementById("eq").textContent = fmt(s.equity);
    document.getElementById("upnl").textContent = fmt(s.unrealised_pnl);

    const p = s.position;
    document.getElementById("pos").textContent = p ? `${{p.side}} size=${{p.size.toFixed(4)}} @ ${{fmt(p.avg_price)}} (tp=${{p.tp ?? "-"}} sl=${{p.sl ?? "-"}})` : "-";
    document.getElementById("ords").textContent = (s.open_orders || []).length;
    document.getElementById("trcnt").textContent = (s.trades || []).length;

    const trades = (s.trades || []);
    if (trades.length) {{
      const t = trades[trades.length-1];
      document.getElementById("last").textContent = `${{t.side}} ${{fmt(t.entry)}} → ${{fmt(t.exit)}}  ${{t.reason}}  pnl=${{fmt(t.pnl)}}`;
    }} else {{
      document.getElementById("last").textContent = "-";
    }}

    document.getElementById("ts").textContent = new Date().toLocaleTimeString();

    if (lastVersion === null || s.version !== lastVersion) {{
      buildPriceFigure(s);
      buildEquityFigure(s);
      lastVersion = s.version;
    }}
  }} catch (e) {{
    console.error(e);
  }}
}}

document.getElementById("btn").addEventListener("click", () => {{
  const v = Number(document.getElementById("bars").value);
  if (isFinite(v)) bars = Math.max(50, Math.min(v, 5000));
  lastVersion = null;
}});

setInterval(tick, 800);
tick();
</script>
</body>
</html>
"""
    return HTMLResponse(html)

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
