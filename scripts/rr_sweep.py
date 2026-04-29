import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""
R:R exit-geometry sweep. Reuses stage-4 OOF predictions (trained on 2:1 R:R labels)
and re-simulates trade exits under a grid of (tp_atr_mult, sl_atr_mult, max_hold_bars,
prob_floor) combinations.

Model is NOT retrained. Only the exit rule that routes existing signals changes.
Runtime: minutes. Output: results/rr_sweep.csv.

Usage:
    .venv/Scripts/python.exe scripts/rr_sweep.py
    .venv/Scripts/python.exe scripts/rr_sweep.py --tp 1.0,1.5,2.0,2.5,3.0 --sl 1.0 --floor 0.55
    .venv/Scripts/python.exe scripts/rr_sweep.py --symbols BTCUSDT,SOLUSDT

Interpretation:
    - realized_rr ≈ (hit_rate × avg_win) / ((1 - hit_rate) × avg_loss)
      is the actual edge, not the target R:R geometry.
    - Model was trained expecting 2:1 geometry. Deviation degrades hit_rate — this
      sweep quantifies how fast.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from omegaconf import OmegaConf

from src.utils.logger import get_logger

logger = get_logger("rr_sweep")

_TEST_START_DEFAULT = "2026-01-01"


def _load_cfg() -> OmegaConf:
    return OmegaConf.load("config/base.yaml")


def _compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period, adjust=False).mean()


def _simulate_one_symbol(
    ohlcv: pd.DataFrame,
    signal_ts: np.ndarray,       # int64 ns timestamps where signals live
    signal_prob: np.ndarray,     # primary model prob at each signal ts
    tp_mult: float,
    sl_mult: float,
    max_hold_bars: int,
    prob_floor: float,
    tp_min_pct: float,
    tp_max_pct: float,
    sl_min_pct: float,
    sl_max_pct: float,
    fee_rt: float,
) -> list[dict]:
    # Returns list of trade dicts for the given exit-geometry combo.
    if len(ohlcv) == 0 or len(signal_ts) == 0:
        return []

    atr = _compute_atr_series(ohlcv)
    closes = ohlcv["close"].values
    highs = ohlcv["high"].values
    lows = ohlcv["low"].values
    idx_ns = ohlcv.index.view("int64")
    # Map each signal timestamp → bar index (exact match; signals come from training data)
    pos_map = pd.Series(np.arange(len(idx_ns)), index=idx_ns)

    trades: list[dict] = []
    for ts, prob in zip(signal_ts, signal_prob):
        if abs(prob - 0.5) < 0.01:
            continue  # no signal in dead zone
        if not (prob >= prob_floor or prob <= (1.0 - prob_floor)):
            continue  # floor gate — matches live signal_strength logic loosely

        i = pos_map.get(ts)
        if i is None or i + 1 >= len(closes):
            continue
        entry_i = int(i) + 1  # enter at NEXT bar open (= this bar's close for simplicity)
        if entry_i >= len(closes):
            continue
        entry = closes[entry_i - 1]
        atr_i = atr.iloc[entry_i - 1]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        atr_pct = atr_i / entry

        tp_pct = min(max(tp_min_pct, atr_pct * tp_mult), tp_max_pct)
        sl_pct = min(max(sl_min_pct, atr_pct * sl_mult), sl_max_pct)

        direction = 1 if prob >= 0.5 else -1
        if direction == 1:
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)
        else:
            tp_price = entry * (1 - tp_pct)
            sl_price = entry * (1 + sl_pct)

        exit_reason = "TIME"
        exit_price = closes[min(entry_i + max_hold_bars, len(closes) - 1)]
        # Walk forward bar-by-bar; check high/low against barriers (conservative intrabar order)
        for j in range(entry_i, min(entry_i + max_hold_bars, len(closes))):
            hi, lo = highs[j], lows[j]
            if direction == 1:
                sl_hit = lo <= sl_price
                tp_hit = hi >= tp_price
            else:
                sl_hit = hi >= sl_price
                tp_hit = lo <= tp_price
            if sl_hit and tp_hit:
                # Ambiguous — assume SL hits first (conservative, matches labels convention)
                exit_price, exit_reason = sl_price, "SL"
                break
            if sl_hit:
                exit_price, exit_reason = sl_price, "SL"
                break
            if tp_hit:
                exit_price, exit_reason = tp_price, "TP"
                break

        if direction == 1:
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry
        pnl_net = pnl_pct - fee_rt

        trades.append({
            "pnl_pct": pnl_pct,
            "pnl_pct_net": pnl_net,
            "exit_reason": exit_reason,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
        })
    return trades


def _summarize(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0, "hit_rate": 0.0, "sharpe": 0.0, "total_return_net": 0.0,
                "max_dd": 0.0, "realized_rr": 0.0, "exit_tp_pct": 0.0, "exit_sl_pct": 0.0,
                "exit_time_pct": 0.0, "avg_pnl_net": 0.0}
    df = pd.DataFrame(trades)
    wins = df["pnl_pct_net"] > 0
    avg_win = df.loc[wins, "pnl_pct_net"].mean() if wins.any() else 0.0
    avg_loss = -df.loc[~wins, "pnl_pct_net"].mean() if (~wins).any() else 0.0
    realized_rr = (avg_win / avg_loss) if avg_loss > 1e-9 else 0.0
    equity = (1 + df["pnl_pct_net"]).cumprod()
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min())
    mu = df["pnl_pct_net"].mean()
    sd = df["pnl_pct_net"].std()
    sharpe = float(mu / (sd + 1e-9)) * np.sqrt(252)  # per-trade sharpe × sqrt(annualization)
    exit_mix = df["exit_reason"].value_counts(normalize=True)
    return {
        "n_trades":         int(len(df)),
        "hit_rate":         round(float(wins.mean()), 4),
        "avg_pnl_net":      round(float(mu), 6),
        "sharpe":           round(sharpe, 3),
        "total_return_net": round(float(equity.iloc[-1] - 1.0), 4),
        "max_dd":           round(max_dd, 4),
        "realized_rr":      round(float(realized_rr), 3),
        "exit_tp_pct":      round(float(exit_mix.get("TP", 0.0)), 4),
        "exit_sl_pct":      round(float(exit_mix.get("SL", 0.0)), 4),
        "exit_time_pct":    round(float(exit_mix.get("TIME", 0.0)), 4),
    }


def _load_signals(symbol: str, cfg, test_start: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    # Returns (timestamps_ns, primary_prob) filtered to test period.
    # Uses stage-6 signals parquet — has full history including test, with primary_prob + meta_prob.
    sig_path = Path(cfg.data.checkpoints_dir) / "signals" / f"{symbol}_15m_signals.parquet"
    if not sig_path.exists():
        return np.array([], dtype="int64"), np.array([])
    df = pd.read_parquet(sig_path, columns=["primary_prob", "is_signal"])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[df.index >= test_start]
    # Keep only rows flagged as a signal (outside dead zone, etc)
    if "is_signal" in df.columns:
        df = df[df["is_signal"].astype(bool)]
    df = df.dropna(subset=["primary_prob"])
    if df.empty:
        return np.array([], dtype="int64"), np.array([])
    return df.index.view("int64"), df["primary_prob"].values.astype(float)


def _load_ohlcv(symbol: str, cfg) -> pd.DataFrame:
    # Raw OHLCV lives in data/raw/<symbol>_15m.parquet with open_time as index.
    path = Path(cfg.data.raw_dir) / f"{symbol}_15m.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.index, utc=True)
    return df.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", default="1.0,1.5,2.0,2.5,3.0",
                    help="comma-separated tp_atr_mult grid")
    ap.add_argument("--sl", default="0.5,1.0,1.5,2.0",
                    help="comma-separated sl_atr_mult grid")
    ap.add_argument("--hold", default="16,32,64",
                    help="comma-separated max_hold_bars grid")
    ap.add_argument("--floor", default="0.55",
                    help="comma-separated prob_floor grid (long side — short uses 1-floor)")
    ap.add_argument("--symbols", default="",
                    help="comma-separated symbol list; empty = all with OOF files")
    ap.add_argument("--out", default="results/rr_sweep.csv")
    args = ap.parse_args()

    cfg = _load_cfg()
    test_start = pd.Timestamp(getattr(cfg.data, "test_start", _TEST_START_DEFAULT), tz="UTC")
    fee_rt = 2 * float(cfg.backtest.commission_pct) + 2 * float(cfg.backtest.slippage_pct)

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        sig_dir = Path(cfg.data.checkpoints_dir) / "signals"
        symbols = sorted({p.stem.replace("_15m_signals", "")
                          for p in sig_dir.glob("*_15m_signals.parquet")})
    logger.info(f"Sweeping {len(symbols)} symbols from test_start={test_start}")

    tp_grid    = [float(x) for x in args.tp.split(",")]
    sl_grid    = [float(x) for x in args.sl.split(",")]
    hold_grid  = [int(x)   for x in args.hold.split(",")]
    floor_grid = [float(x) for x in args.floor.split(",")]

    tp_min = float(cfg.labels.tp_min_pct); tp_max = float(cfg.labels.tp_max_pct)
    sl_min = float(cfg.labels.sl_min_pct); sl_max = float(cfg.labels.sl_max_pct)

    # Pre-load per-symbol data once
    cache = {}
    for s in symbols:
        sig_ts, sig_p = _load_signals(s, cfg, test_start)
        if len(sig_ts) == 0:
            continue
        ohlcv = _load_ohlcv(s, cfg)
        if ohlcv.empty:
            continue
        cache[s] = (ohlcv, sig_ts, sig_p)
    logger.info(f"Loaded {len(cache)} symbols with signals + OHLCV")

    rows = []
    for tp_m in tp_grid:
        for sl_m in sl_grid:
            for hold in hold_grid:
                for floor in floor_grid:
                    all_trades: list[dict] = []
                    for s, (ohlcv, sig_ts, sig_p) in cache.items():
                        t = _simulate_one_symbol(
                            ohlcv, sig_ts, sig_p,
                            tp_mult=tp_m, sl_mult=sl_m,
                            max_hold_bars=hold, prob_floor=floor,
                            tp_min_pct=tp_min, tp_max_pct=tp_max,
                            sl_min_pct=sl_min, sl_max_pct=sl_max,
                            fee_rt=fee_rt,
                        )
                        all_trades.extend(t)
                    summary = _summarize(all_trades)
                    summary.update({
                        "tp_mult": tp_m, "sl_mult": sl_m,
                        "max_hold_bars": hold, "prob_floor": floor,
                        "target_rr": round(tp_m / sl_m, 3) if sl_m > 0 else 0.0,
                    })
                    rows.append(summary)
                    logger.info(
                        f"tp={tp_m} sl={sl_m} hold={hold} floor={floor} → "
                        f"n={summary['n_trades']} hit={summary['hit_rate']:.3f} "
                        f"sharpe={summary['sharpe']:.2f} ret={summary['total_return_net']:.2%} "
                        f"rr={summary['realized_rr']:.2f} tp/sl/time={summary['exit_tp_pct']:.0%}/"
                        f"{summary['exit_sl_pct']:.0%}/{summary['exit_time_pct']:.0%}"
                    )

    df = pd.DataFrame(rows)
    cols = ["tp_mult", "sl_mult", "target_rr", "max_hold_bars", "prob_floor",
            "n_trades", "hit_rate", "realized_rr", "sharpe", "total_return_net",
            "avg_pnl_net", "max_dd", "exit_tp_pct", "exit_sl_pct", "exit_time_pct"]
    df = df[cols].sort_values("sharpe", ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df)} combos to {out}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
