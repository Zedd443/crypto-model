"""
Standalone test: open 2 positions on DEMO using exact stage_08 sizing logic.
Symbols: ETHUSDT + XRPUSDT -- different from Postman tested coins (SOL/BTC/WIF/ASTER)
Usage:
    D:/Workspace/AI/crypto_model/.venv/Scripts/python.exe test_inject_2pos.py
"""
import sys
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from omegaconf import OmegaConf
from src.execution.binance_client import BinanceClient
from src.execution.order_manager import OrderManager

cfg = OmegaConf.load(ROOT / "config" / "base.yaml")


def _get_vol_mult(wallet, cfg):
    tiers = list(cfg.growth_gate.tiers)
    for tier in sorted(tiers, key=lambda t: float(t.max_equity)):
        if wallet <= float(tier.max_equity):
            return float(getattr(tier, "vol_mult", 2.0))
    return 2.0


def _round_qty(qty, step):
    if step <= 0:
        return qty
    decimals = max(0, round(-math.log10(step)))
    return round(math.floor(qty / step) * step, decimals)


def _compute_qty(sym, volume_usd, client, cfg, leverage):
    qty_step     = client.get_qty_step(sym)
    min_notional = client.get_min_notional(sym)
    max_qty      = client.get_max_qty(sym, 0)
    klines       = client.get_klines(sym, "15m", limit=2)
    entry_price  = float(klines["close"].iloc[-1])

    qty_raw  = volume_usd / entry_price
    qty      = _round_qty(min(qty_raw, max_qty), qty_step)
    notional = qty * entry_price

    if notional < min_notional:
        qty      = _round_qty(min((min_notional * 1.05) / entry_price, max_qty), qty_step)
        notional = qty * entry_price

    if cfg.trading.mode == "DEMO" and qty_step <= 0.001 and qty > 1.0 and qty != math.floor(qty):
        qty_int = math.floor(qty)
        if qty_int * entry_price >= min_notional:
            qty      = float(qty_int)
            notional = qty * entry_price

    return {
        "entry_price": entry_price,
        "qty": qty,
        "notional": notional,
        "margin": notional / leverage,
        "qty_step": qty_step,
        "min_notional": min_notional,
        "max_qty": max_qty,
    }


# ── init ─────────────────────────────────────────────────────────────────────

print(f"\nMode    : {cfg.trading.mode}")
print(f"Endpoint: {cfg.trading.endpoints[cfg.trading.mode]}")

client = BinanceClient(cfg)
order_manager = OrderManager(
    client, cfg,
    trade_log_path=ROOT / "results" / "live_trade_log.csv",
    mode=cfg.trading.mode,
)

acct   = client.get_account()
wallet = float(acct.get("totalWalletBalance", 0))
avail  = float(acct.get("availableBalance", 0))
print(f"\nWallet total : ${wallet:.4f}")
print(f"Available    : ${avail:.4f}")

leverage   = int(cfg.trading.leverage)
vol_mult   = _get_vol_mult(wallet, cfg)
volume_usd = wallet * vol_mult
tp_pct     = float(cfg.growth_gate.tp_fixed_pct)
sl_pct     = float(cfg.growth_gate.sl_fixed_pct)

print(f"\nvol_mult     : {vol_mult}x")
print(f"volume/pos   : ${volume_usd:.2f}")
print(f"leverage     : {leverage}x")
print(f"margin/pos   : ${volume_usd / leverage:.2f}")
print(f"TP / SL      : +{tp_pct:.0%} / -{sl_pct:.0%}")

# Pipeline coins not tested in Postman: AVAX (mid-cap decimal) + 1000BONK (micro-price integer)
SYMBOLS = [("AVAXUSDT", "long"), ("1000BONKUSDT", "long")]

# ── set leverage ─────────────────────────────────────────────────────────────

for sym, _ in SYMBOLS:
    try:
        r = client._request("POST", "/fapi/v1/leverage",
                            params={"symbol": sym, "leverage": leverage}, signed=True)
        print(f"\n{sym}: leverage set -> {r.get('leverage')}x")
    except Exception as e:
        print(f"\n{sym}: set leverage FAILED: {e}")

# ── dry-run sizing for all symbols ───────────────────────────────────────────

print("\n" + "="*65)
print(f"{'SYMBOL':<12} {'PRICE':>10} {'QTY':>12} {'NOTIONAL':>10} {'MARGIN':>8} {'STEP':>8} {'MAX_QTY':>8}")
print("-"*65)

results = []
for sym, direction in SYMBOLS:
    info = _compute_qty(sym, volume_usd, client, cfg, leverage)
    tp_price = round(info["entry_price"] * (1 + tp_pct if direction == "long" else 1 - tp_pct), 8)
    sl_price = round(info["entry_price"] * (1 - sl_pct if direction == "long" else 1 + sl_pct), 8)
    print(f"  {sym:<10} {info['entry_price']:>10.4f} {info['qty']:>12.4f} "
          f"${info['notional']:>9.2f} ${info['margin']:>7.2f} "
          f"{info['qty_step']:>8} {info['max_qty']:>8}")
    results.append({**info, "symbol": sym, "direction": direction,
                    "tp_price": tp_price, "sl_price": sl_price})

print("="*65)

# ── place orders ─────────────────────────────────────────────────────────────

print("\nPlacing orders...")
order_results = []
for r in results:
    sym       = r["symbol"]
    direction = r["direction"]
    qty       = r["qty"]
    side      = "BUY" if direction == "long" else "SELL"

    print(f"\n[{sym} {direction.upper()}]")
    print(f"  qty={qty}  notional=${r['notional']:.2f}  TP={r['tp_price']}  SL={r['sl_price']}")

    try:
        resp       = client.place_order(sym, side, qty, order_type="MARKET")
        order_id   = resp.get("orderId")
        avg_price  = float(resp.get("avgPrice") or r["entry_price"])
        filled_qty = float(resp.get("executedQty") or qty)
        status     = resp.get("status")
        print(f"  ORDER: orderId={order_id} status={status} avgPrice={avg_price}")
        order_results.append({"symbol": sym, "order_id": order_id, "status": status,
                              "qty": filled_qty, "avg_price": avg_price,
                              "notional": filled_qty * avg_price,
                              "tp_price": r["tp_price"], "sl_price": r["sl_price"]})
    except Exception as e:
        print(f"  FAILED: {e}")
        order_results.append({"symbol": sym, "error": str(e)})

    time.sleep(1)

# ── verify fills ─────────────────────────────────────────────────────────────

time.sleep(3)
print("\n" + "="*65)
print("POSITION VERIFY (from exchange)")
print("="*65)
for r in order_results:
    sym = r["symbol"]
    try:
        pos = client.get_position(sym)
        amt  = pos["positionAmt"]
        ep   = pos["entryPrice"]
        mark = pos["markPrice"]
        upnl = pos["unrealizedProfit"]
        print(f"  {sym:<12} posAmt={amt}  entry={ep}  mark={mark}  uPnL={upnl}")
    except Exception as e:
        print(f"  {sym}: get_position failed: {e}")

acct2 = client.get_account()
print(f"\nAvailable after: ${float(acct2.get('availableBalance', 0)):.2f}  "
      f"(was ${avail:.2f}, used ${avail - float(acct2.get('availableBalance', 0)):.2f} margin)")

# ── close ────────────────────────────────────────────────────────────────────

print("\nClosing positions...")
for r in order_results:
    if "error" in r:
        continue
    sym = r["symbol"]
    try:
        pos = client.get_position(sym)
        amt = abs(float(pos["positionAmt"]))
        if amt == 0:
            print(f"  {sym}: already closed")
            continue
        qty_step = client.get_qty_step(sym)
        close_qty = _round_qty(amt, qty_step)
        close_side = "SELL" if float(pos["positionAmt"]) > 0 else "BUY"
        cr = client.place_order(sym, close_side, close_qty, order_type="MARKET", reduce_only=True)
        print(f"  {sym}: closed orderId={cr.get('orderId')} status={cr.get('status')}")
    except Exception as e:
        print(f"  {sym}: close FAILED: {e}")
    time.sleep(0.5)

print("\nDone.\n")
