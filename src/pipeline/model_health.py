"""
Model health report — run after stage 4/5 to diagnose per-symbol quality.
Also reports artifact coverage (labels, signals, model, meta) across all symbols.
Output: console table + results/model_health.csv

Usage:
    .venv/Scripts/python.exe -m src.pipeline.model_health
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_state() -> dict:
    p = ROOT / "project_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _load_training_summary() -> pd.DataFrame:
    p = ROOT / "results" / "training_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _load_meta_summary() -> pd.DataFrame:
    p = ROOT / "results" / "meta_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _load_backtest_summary() -> pd.DataFrame:
    p = ROOT / "results" / "backtest_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _load_diagnostics() -> pd.DataFrame:
    p = ROOT / "results" / "pipeline_diagnostics.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _model_file_size(symbol: str, tf: str = "15m") -> float:
    p = ROOT / "models" / f"{symbol}_{tf}_model.json"
    return p.stat().st_size / 1024 if p.exists() else 0.0


def _artifact_coverage(symbols: list[str], tf: str = "15m") -> dict:
    """Returns per-symbol artifact presence flags."""
    coverage = {}
    for sym in symbols:
        coverage[sym] = {
            "has_labels":  (ROOT / "data" / "labels"  / f"{sym}_{tf}_labels.parquet").exists(),
            "has_model":   (ROOT / "models"            / f"{sym}_{tf}_model.json").exists(),
            "has_meta":    (ROOT / "models"            / f"{sym}_{tf}_meta.pkl").exists(),
            "has_signals": (ROOT / "data" / "checkpoints" / "signals" / f"{sym}_{tf}_signals.parquet").exists(),
            "has_imputer": (ROOT / "data" / "checkpoints" / "imputers" / f"imputer_{sym}_{tf}.pkl").exists(),
        }
    return coverage


def run():
    state    = _load_state()
    train_df = _load_training_summary()
    meta_df  = _load_meta_summary()
    bt_df    = _load_backtest_summary()
    diag_df  = _load_diagnostics()

    completed = state.get("stages", {}).get("training", {}).get("completed_symbols", [])
    meta_done = state.get("stages", {}).get("meta_labeling", {}).get("completed_symbols", [])
    failed    = state.get("stages", {}).get("training", {}).get("failed_symbols", {})

    all_symbols = sorted(set(completed) | set(failed.keys()))
    coverage = _artifact_coverage(all_symbols)

    # Index lookup helpers
    train_idx = {}
    if not train_df.empty and "symbol" in train_df.columns:
        train_idx = train_df.set_index("symbol").to_dict("index")

    meta_idx = {}
    if not meta_df.empty and "symbol" in meta_df.columns:
        meta_idx = meta_df.set_index("symbol").to_dict("index")

    bt_idx = {}
    if not bt_df.empty and "symbol" in bt_df.columns:
        bt_idx = bt_df.set_index("symbol").to_dict("index")

    diag_idx = {}
    if not diag_df.empty and "symbol" in diag_df.columns:
        diag_idx = diag_df.drop_duplicates("symbol").set_index("symbol").to_dict("index")

    rows = []
    for sym in all_symbols:
        t   = train_idx.get(sym, {})
        m   = meta_idx.get(sym, {})
        bt  = bt_idx.get(sym, {})
        dg  = diag_idx.get(sym, {})

        da_val       = t.get("da", t.get("val_da", t.get("da_val", float("nan"))))
        da_test      = t.get("test_da", t.get("da_test", float("nan")))
        pct_pos      = t.get("pct_positive_val", t.get("pct_positive_train", float("nan")))
        meta_acc     = m.get("meta_accuracy", m.get("accuracy", float("nan")))
        meta_spw     = m.get("scale_pos_weight", float("nan"))
        sharpe       = bt.get("sharpe", float("nan"))
        calmar       = bt.get("calmar", float("nan"))
        n_trades     = bt.get("n_trades", float("nan"))
        model_kb     = _model_file_size(sym)
        is_failed    = sym in failed
        cov          = coverage.get(sym, {})

        # Flags
        flags = []
        if is_failed:
            flags.append("TRAIN_FAILED")
        if not cov.get("has_labels"):
            flags.append("NO_LABELS")
        if not cov.get("has_model"):
            flags.append("NO_MODEL_FILE")
        if not cov.get("has_meta"):
            flags.append("NO_META")
        if not cov.get("has_signals"):
            flags.append("NO_SIGNALS")
        if not math.isnan(da_val) and not math.isnan(pct_pos):
            majority = max(pct_pos, 1 - pct_pos)
            if da_val <= majority + 0.01:
                flags.append("NO_SIGNAL_VAL")
        if not math.isnan(da_test) and not math.isnan(pct_pos):
            majority = max(pct_pos, 1 - pct_pos)
            if da_test <= majority + 0.01:
                flags.append("NO_SIGNAL_TEST")
        if not math.isnan(sharpe) and sharpe < 0.5:
            flags.append("LOW_SHARPE")
        if not math.isnan(n_trades) and n_trades < 10:
            flags.append("FEW_TRADES")

        rows.append({
            "symbol":       sym,
            "da_val":       round(da_val, 4)  if not math.isnan(da_val)   else None,
            "da_test":      round(da_test, 4) if not math.isnan(da_test)  else None,
            "pct_pos":      round(pct_pos, 4) if not math.isnan(pct_pos)  else None,
            "meta_acc":     round(meta_acc, 4) if not math.isnan(meta_acc) else None,
            "meta_spw":     round(meta_spw, 2) if not math.isnan(meta_spw) else None,
            "sharpe":       round(sharpe, 3)  if not math.isnan(sharpe)   else None,
            "calmar":       round(calmar, 3)  if not math.isnan(calmar)   else None,
            "n_trades":     int(n_trades)     if not math.isnan(n_trades) else None,
            "model_kb":     round(model_kb, 1),
            "meta_trained": sym in meta_done,
            "flags":        "|".join(flags) if flags else "OK",
        })

    result = pd.DataFrame(rows)

    # Save
    out = ROOT / "results" / "model_health.csv"
    result.to_csv(out, index=False)

    # Console output
    print("\n" + "="*110)
    print("MODEL HEALTH REPORT")
    print("="*110)

    col_w = {"symbol":16,"da_val":8,"da_test":9,"pct_pos":8,"meta_acc":10,"sharpe":8,"calmar":8,"n_trades":9,"model_kb":10,"flags":30}
    hdr   = "".join(k.ljust(v) for k,v in col_w.items())
    print(hdr)
    print("-"*110)

    ok_count = warn_count = crit_count = 0
    for _, r in result.sort_values("flags").iterrows():
        flag = r["flags"]
        if flag == "OK":
            ok_count += 1
        elif any(x in flag for x in ["NO_SIGNAL","TRAIN_FAILED","NO_MODEL"]):
            crit_count += 1
        else:
            warn_count += 1

        line = (
            str(r["symbol"]).ljust(col_w["symbol"]) +
            str(r["da_val"]).ljust(col_w["da_val"]) +
            str(r["da_test"]).ljust(col_w["da_test"]) +
            str(r["pct_pos"]).ljust(col_w["pct_pos"]) +
            str(r["meta_acc"]).ljust(col_w["meta_acc"]) +
            str(r["sharpe"]).ljust(col_w["sharpe"]) +
            str(r["calmar"]).ljust(col_w["calmar"]) +
            str(r["n_trades"]).ljust(col_w["n_trades"]) +
            str(r["model_kb"]).ljust(col_w["model_kb"]) +
            str(r["flags"])
        )
        print(line)

    print("="*110)
    print("SUMMARY: %d OK | %d WARN | %d CRITICAL | %d total" % (ok_count, warn_count, crit_count, len(result)))
    print("Saved: %s" % out)
    print()

    # Artifact coverage summary
    n = len(all_symbols)
    if n > 0:
        n_labels  = sum(1 for c in coverage.values() if c["has_labels"])
        n_models  = sum(1 for c in coverage.values() if c["has_model"])
        n_meta    = sum(1 for c in coverage.values() if c["has_meta"])
        n_signals = sum(1 for c in coverage.values() if c["has_signals"])
        n_imputer = sum(1 for c in coverage.values() if c["has_imputer"])
        print("="*60)
        print("ARTIFACT COVERAGE  (%d symbols)" % n)
        print("="*60)
        print("  labels    %d / %d" % (n_labels,  n))
        print("  imputers  %d / %d" % (n_imputer, n))
        print("  models    %d / %d" % (n_models,  n))
        print("  meta      %d / %d" % (n_meta,    n))
        print("  signals   %d / %d" % (n_signals, n))
        missing_models = [s for s, c in coverage.items() if not c["has_model"]]
        if missing_models:
            print("  MISSING MODEL: " + ", ".join(missing_models[:10]))
        print("="*60)
        print()

    # What Claude needs to improve the model
    print("="*110)
    print("WHAT CLAUDE NEEDS TO IMPROVE THE MODEL")
    print("="*110)
    print("""
To diagnose and improve model quality, Claude needs:

1. RETRAIN OUTPUT (per symbol, from stage 4 --force):
   - results/training_summary.csv  -> da_val, da_test, pct_positive_train per symbol
   - results/pipeline_diagnostics.csv -> fold Sharpe variance, top SHAP features
   - logs/stage_04_train.txt -> full XGBoost training log (Optuna trials, best params)

2. META-LABEL OUTPUT (from stage 5):
   - results/meta_summary.csv -> meta_accuracy, scale_pos_weight, n_meta0/n_meta1
   - logs/stage_05_meta.txt

3. BACKTEST OUTPUT (from stage 7):
   - results/backtest_summary.csv -> Sharpe, Calmar, n_trades, win_rate per symbol
   - results/trade_log.csv -> all trades with entry/exit prices, pnl
   - results/equity_curve.csv -> equity over time

4. LIVE OUTPUT (from stage 8 after >50 trades):
   - logs/stage_08_live.log -> order fills, DMS events, sync_fills hits
   - results/live_trade_log.csv -> actual live PnL vs backtest PnL

5. FEATURE QUALITY:
   - results/pipeline_diagnostics.csv (stage=features) -> PSI per feature, missing rate
   - Top 10 SHAP features per symbol from training_summary or diagnostics

ACTION ITEMS IDENTIFIED THIS SESSION:
   [P1] fracdiff_threshold: 0.01 -> 0.05 (ADF too strict, d=0 for most features)
   [P1] amihud/ofi/vwap windows: 20 -> 96 bars (too noisy on 15m)
   [P1] conf_width_series=0.20 hardcoded in signal_generator.py (unused placeholder)
   [P1] 30x bare except Exception in execution layer (mask real errors)
   [P2] _train_symbol() 280 lines -> refactor into 4 helpers
   [P2] PBO hardcoded 0.5 -> all models Tier B, never Tier A leverage
""")


if __name__ == "__main__":
    run()
