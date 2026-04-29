"""
Live-vs-training drift rollup. Refreshed every bar by stage_08.

Compares live trade outcomes against what the model was trained to expect,
using the OOF-simulated backtest as the reference distribution. Claude reads
the output JSON at session start (via project_state.token_hints.live_drift).

Key metrics:
  - Rolling hit_rate (20/50/100 trades) with Wilson lower bound
  - Realized R:R vs trained R:R (labels.tp_atr_mult / labels.sl_atr_mult)
  - Exit reason mix (TIME dominance → signal decays faster than training)
  - Probability calibration residual — live primary_prob vs hit outcome
  - Per-symbol live-vs-training hit_rate gap (flag worst divergences)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from src.utils.logger import get_logger

logger = get_logger("live_vs_training")

_TRADE_LOG = Path("results/live_trade_log.csv")
_OUT = Path("monitoring/live_vs_training.json")
_CFG_PATH = Path("config/base.yaml")


def _wilson_lower(p: float, n: int, z: float = 1.96) -> float:
    # Wilson score lower bound — honest floor for hit_rate with small sample
    if n <= 0:
        return 0.0
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(max(0.0, (centre - margin) / denom))


def _tail_stats(df: pd.DataFrame, n: int) -> dict:
    tail = df.tail(n)
    if tail.empty:
        return {"n": 0}
    wins = (tail["pnl_pct"] > 0)
    hit = float(wins.mean())
    pnl_col = "pnl_pct_net" if "pnl_pct_net" in tail.columns else "pnl_pct"
    avg_win = float(tail.loc[wins, pnl_col].mean()) if wins.any() else 0.0
    avg_loss = -float(tail.loc[~wins, pnl_col].mean()) if (~wins).any() else 0.0
    return {
        "n":                int(len(tail)),
        "hit_rate":         round(hit, 4),
        "hit_rate_lb95":    round(_wilson_lower(hit, len(tail)), 4),
        "avg_pnl_pct":      round(float(tail["pnl_pct"].mean()), 6),
        "avg_pnl_pct_net":  round(float(tail.get("pnl_pct_net", tail["pnl_pct"]).mean()), 6),
        "avg_bars_held":    round(float(tail["bars_held"].mean()), 2)
                             if "bars_held" in tail.columns and tail["bars_held"].notna().any() else None,
        "realized_rr":      round(avg_win / avg_loss, 3) if avg_loss > 1e-9 else None,
    }


def _prob_calibration_residual(df: pd.DataFrame) -> dict | None:
    # Bucket entries by primary_prob_at_entry, compare bucket mean prob vs realized hit rate.
    # Returns mean absolute residual (Brier-style). Low = well calibrated; high = drift.
    if "primary_prob_at_entry" not in df.columns or df.empty:
        return None
    d = df[df["primary_prob_at_entry"].between(0.0, 1.0, inclusive="neither")].copy()
    if len(d) < 20:
        return None
    d["hit"] = (d["pnl_pct"] > 0).astype(int)
    bins = np.linspace(0.5, 1.0, 6)  # 0.5–0.6, …, 0.9–1.0 (long side) — use |prob-0.5|+0.5 for shorts
    d["conf"] = np.where(d["primary_prob_at_entry"] >= 0.5,
                         d["primary_prob_at_entry"],
                         1 - d["primary_prob_at_entry"])
    d["bucket"] = pd.cut(d["conf"], bins=bins, include_lowest=True)
    g = d.groupby("bucket", observed=True).agg(
        bucket_prob=("conf", "mean"),
        bucket_hit=("hit", "mean"),
        n=("hit", "size"),
    ).dropna()
    if g.empty:
        return None
    g["residual"] = (g["bucket_prob"] - g["bucket_hit"]).abs()
    weighted = float((g["residual"] * g["n"]).sum() / g["n"].sum())
    return {
        "mean_abs_residual": round(weighted, 4),
        "by_bucket": [
            {"conf_bin": str(idx), "prob": round(float(r["bucket_prob"]), 3),
             "hit": round(float(r["bucket_hit"]), 3), "n": int(r["n"])}
            for idx, r in g.iterrows()
        ],
    }


def _psi(live: np.ndarray, ref: np.ndarray, bins: int = 10) -> float:
    if len(live) < 20 or len(ref) < 20:
        return float("nan")
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r_hist, _ = np.histogram(ref,  bins=edges)
    l_hist, _ = np.histogram(live, bins=edges)
    r_p = np.clip(r_hist / max(r_hist.sum(), 1), 1e-6, None)
    l_p = np.clip(l_hist / max(l_hist.sum(), 1), 1e-6, None)
    return float(((l_p - r_p) * np.log(l_p / r_p)).sum())


def _oof_reference_hit_rate() -> pd.Series | None:
    # Per-symbol OOF hit_rate computed on-the-fly from stage-4 OOF files.
    # This is the "what we trained for" benchmark, using a simple hit rule: prob>0.5 predicts up.
    # A richer version would load actual labels, but labels + geometry aren't the comparison —
    # we want to know if the primary model's directional calls are converting at the live win rate.
    try:
        cfg = OmegaConf.load(_CFG_PATH)
        ckpt = Path(cfg.data.checkpoints_dir) / "oof"
        labels_dir = Path(cfg.data.labels_dir)
        if not ckpt.exists() or not labels_dir.exists():
            return None
        rows = {}
        for proba_path in ckpt.glob("*_15m_oof_proba.npy"):
            symbol = proba_path.stem.replace("_15m_oof_proba", "")
            idx_path = ckpt / f"{symbol}_15m_oof_index.npy"
            label_path = labels_dir / f"{symbol}_15m_labels.parquet"
            if not idx_path.exists() or not label_path.exists():
                continue
            proba = np.load(proba_path)
            idx_ns = np.load(idx_path)
            lab = pd.read_parquet(label_path, columns=["label"])
            lab.index = pd.to_datetime(lab.index, utc=True)
            lab_idx_ns = lab.index.view("int64")
            common_mask = np.isin(idx_ns, lab_idx_ns)
            if common_mask.sum() < 30:
                continue
            sub_proba = proba[common_mask]
            sub_idx = idx_ns[common_mask]
            lab_aligned = lab.loc[pd.to_datetime(sub_idx, utc=True), "label"].values
            # Hit: model direction matched non-neutral label sign
            pred_long = sub_proba >= 0.5
            hit = ((pred_long & (lab_aligned == 1)) | (~pred_long & (lab_aligned == -1))).sum()
            denom = (lab_aligned != 0).sum()
            if denom > 0:
                rows[symbol] = hit / denom
        return pd.Series(rows, name="train_hr") if rows else None
    except Exception as exc:
        logger.debug(f"OOF reference hit_rate compute failed: {exc}")
        return None


def _rr_compare(df: pd.DataFrame, trained_rr: float) -> dict:
    if not {"tp_pct_used", "sl_pct_used"}.issubset(df.columns):
        return {"trained_rr": trained_rr}
    tp_mean = float(df["tp_pct_used"].replace(0, np.nan).dropna().mean())
    sl_mean = float(df["sl_pct_used"].replace(0, np.nan).dropna().mean())
    geom_rr = (tp_mean / sl_mean) if (sl_mean and sl_mean > 0) else float("nan")
    pnl_col = "pnl_pct_net" if "pnl_pct_net" in df.columns else "pnl_pct"
    wins = df[pnl_col] > 0
    avg_win  = float(df.loc[ wins, pnl_col].mean()) if wins.any() else 0.0
    avg_loss = -float(df.loc[~wins, pnl_col].mean()) if (~wins).any() else 0.0
    realized_rr = (avg_win / avg_loss) if avg_loss > 1e-9 else float("nan")
    return {
        "trained_rr":     round(trained_rr, 3),
        "live_geom_rr":   round(geom_rr, 3) if geom_rr == geom_rr else None,
        "live_realized_rr": round(realized_rr, 3) if realized_rr == realized_rr else None,
        "avg_tp_pct":     round(tp_mean, 6) if tp_mean == tp_mean else None,
        "avg_sl_pct":     round(sl_mean, 6) if sl_mean == sl_mean else None,
    }


def compute_rollup(trade_log: Path = _TRADE_LOG) -> dict:
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_trades": 0,
        "status": "no_trade_log",
    }
    if not trade_log.exists():
        return out
    df = pd.read_csv(trade_log)
    df["n_rows"] = 1
    out["n_trades"] = int(len(df))
    if df.empty:
        out["status"] = "empty_trade_log"
        return out

    out["rolling"] = {
        "last_20":  _tail_stats(df, 20),
        "last_50":  _tail_stats(df, 50),
        "last_100": _tail_stats(df, 100),
        "all":      _tail_stats(df, len(df)),
    }

    # R:R comparison — trained R:R from config
    try:
        cfg = OmegaConf.load(_CFG_PATH)
        trained_rr = float(cfg.labels.tp_atr_mult) / max(float(cfg.labels.sl_atr_mult), 1e-9)
    except Exception:
        trained_rr = 2.0
    out["rr"] = _rr_compare(df, trained_rr)

    # Exit reason mix
    if "exit_reason" in df.columns:
        out["exit_reason_mix"] = {
            k: round(float(v), 4)
            for k, v in df["exit_reason"].value_counts(normalize=True).items()
        }

    # Probability calibration residual — drift signal that doesn't need retraining to detect
    calib = _prob_calibration_residual(df)
    if calib is not None:
        out["prob_calibration"] = calib

    # Per-symbol hit_rate gap vs OOF-derived training baseline
    train_hr = _oof_reference_hit_rate()
    if train_hr is not None and "symbol" in df.columns:
        live_hr = (df.assign(win=(df["pnl_pct"] > 0).astype(int))
                     .groupby("symbol")["win"].agg(["mean", "count"])
                     .rename(columns={"mean": "live_hr", "count": "n"}))
        joined = live_hr.join(train_hr, how="inner")
        joined = joined[joined["n"] >= 5]
        if not joined.empty:
            joined["gap"] = (joined["live_hr"] - joined["train_hr"]).round(4)
            joined["live_hr"] = joined["live_hr"].round(4)
            joined["train_hr"] = joined["train_hr"].round(4)
            worst = joined.sort_values("gap").head(5).reset_index().to_dict("records")
            best  = joined.sort_values("gap", ascending=False).head(5).reset_index().to_dict("records")
            out["per_symbol_hitrate"] = {"worst_5": worst, "best_5": best,
                                          "n_symbols_compared": int(len(joined))}

    # ATR regime PSI: live entry ATR distribution vs recent OOF-period ATR
    if "atr_pct_at_entry" in df.columns and df["atr_pct_at_entry"].notna().any():
        try:
            # Reference: recent 30-day ATR across all symbols' processed data
            cfg = OmegaConf.load(_CFG_PATH)
            proc_dir = Path(cfg.data.processed_dir)
            ref_atr: list[np.ndarray] = []
            for p in list(proc_dir.glob("*_15m.parquet"))[:20]:  # sample 20 symbols — cheap
                try:
                    sub = pd.read_parquet(p, columns=["high", "low", "close"]).tail(2880)  # 30d × 96 bars
                    tr = pd.concat([sub["high"] - sub["low"],
                                    (sub["high"] - sub["close"].shift(1)).abs(),
                                    (sub["low"]  - sub["close"].shift(1)).abs()], axis=1).max(axis=1)
                    atr = tr.ewm(span=14, adjust=False).mean()
                    ref_atr.append((atr / sub["close"]).dropna().values)
                except Exception:
                    continue
            if ref_atr:
                ref_vec = np.concatenate(ref_atr)
                live_vec = df["atr_pct_at_entry"].replace(0, np.nan).dropna().values
                out["atr_regime_psi"] = round(_psi(live_vec, ref_vec), 4)
        except Exception as exc:
            logger.debug(f"ATR PSI compute failed: {exc}")

    # Verdict flags — lets Claude skim without re-reading all metrics
    flags = []
    tail_all = out["rolling"]["all"]
    if tail_all.get("hit_rate_lb95") is not None and tail_all["hit_rate_lb95"] < 0.45 and tail_all["n"] >= 20:
        flags.append("hit_rate_lb95<0.45")
    if out["rr"].get("live_realized_rr") is not None and out["rr"]["live_realized_rr"] < 1.0:
        flags.append("realized_rr<1.0")
    if out.get("exit_reason_mix", {}).get("TIME", 0) > 0.50:
        flags.append("time_exits_dominant")
    if out.get("atr_regime_psi") is not None and out["atr_regime_psi"] > 0.20:
        flags.append("atr_regime_drift_major")
    if calib and calib["mean_abs_residual"] > 0.15:
        flags.append("prob_calibration_drift")
    out["flags"] = flags
    out["status"] = "ok"
    return out


def write_rollup(out_path: Path = _OUT) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = compute_rollup()
    out_path.write_text(json.dumps(data, indent=2, default=str))
    return out_path


if __name__ == "__main__":
    path = write_rollup()
    logger.info(f"Rollup written to {path}")
