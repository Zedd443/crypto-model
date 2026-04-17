import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("market_positioning")


def _zscore(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window, min_periods=window // 2)
    return ((series - roll.mean()) / (roll.std() + 1e-9)).rename(f"{series.name}_zscore_{window}")


def _load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df
    except Exception as e:
        logger.warning(f"Could not load {path}: {e}")
        return pd.DataFrame()


def build_market_positioning_features(
    symbol: str,
    target_index: pd.DatetimeIndex,
    raw_dir: Path,
    cfg,
) -> pd.DataFrame:
    """
    Build market positioning features from pre-fetched Binance FAPI data:
    - Open Interest: level, change (1h/4h/24h), zscore, spike flag
    - Global LS ratio: level, zscore, contrarian flags
    - Top trader position LS ratio: level, zscore
    - Taker buy/sell ratio: level, zscore, imbalance
    All features are computed on the raw data then reindexed to target_index via ffill.
    No look-ahead: all computations are backward-looking rolling ops.
    """
    raw_dir = Path(raw_dir)
    oi_window = int(cfg.features.oi_zscore_window)
    oi_change_windows = list(cfg.features.oi_change_windows)
    ls_window = int(cfg.features.ls_zscore_window)
    taker_window = int(cfg.features.taker_zscore_window)
    extreme_z = float(cfg.features.positioning_extreme_zscore)

    parts = []

    # ── Open Interest ──────────────────────────────────────────────────────────
    oi_path = raw_dir / f"{symbol}_oi_15m.parquet"
    oi_df = _load_parquet_safe(oi_path)
    if not oi_df.empty and "sumOpenInterestValue" in oi_df.columns:
        oi_val = oi_df["sumOpenInterestValue"].rename("oi_value")
        oi_features = pd.DataFrame(index=oi_df.index)
        oi_features["oi_value"] = oi_val

        # Rolling zscore of OI level
        oi_features["oi_zscore"] = _zscore(oi_val, oi_window)

        # OI changes over multiple windows (pct change — scale invariant)
        for w in oi_change_windows:
            col = f"oi_change_{w}b"
            oi_features[col] = oi_val.pct_change(w).clip(-1.0, 5.0)

        # Spike flag: OI z-score > 2 (sudden large OI buildup)
        oi_features["oi_spike"] = (oi_features["oi_zscore"].abs() > extreme_z).astype(int)

        # OI contracts (coin-denominated) if available
        if "sumOpenInterest" in oi_df.columns:
            oi_contracts = oi_df["sumOpenInterest"].rename("oi_contracts")
            oi_features["oi_contracts_zscore"] = _zscore(oi_contracts, oi_window)

        # Reindex to target — ffill only (OI updates every bar at 15m)
        oi_aligned = oi_features.reindex(target_index, method="ffill", limit=4)
        parts.append(oi_aligned)
        logger.debug(f"{symbol}: OI features built ({len(oi_features)} rows → {oi_aligned.shape[1]} cols)")
    else:
        logger.debug(f"{symbol}: no OI data, skipping OI features")

    # ── Global Long/Short Ratio ────────────────────────────────────────────────
    gls_path = raw_dir / f"{symbol}_ls_global_15m.parquet"
    gls_df = _load_parquet_safe(gls_path)
    if not gls_df.empty and "longShortRatio" in gls_df.columns:
        gls = gls_df["longShortRatio"].rename("ls_global_ratio")
        gls_features = pd.DataFrame(index=gls_df.index)
        gls_features["ls_global_ratio"] = gls

        # Z-score of the ratio
        gls_features["ls_global_zscore"] = _zscore(gls, ls_window)

        # Contrarian flags: extreme long → potential top, extreme short → potential bottom
        gls_features["ls_global_extreme_long"] = (gls_features["ls_global_zscore"] > extreme_z).astype(int)
        gls_features["ls_global_extreme_short"] = (gls_features["ls_global_zscore"] < -extreme_z).astype(int)

        # Long account pct if available
        if "longAccount" in gls_df.columns:
            long_pct = gls_df["longAccount"].rename("ls_global_long_pct")
            gls_features["ls_global_long_pct"] = long_pct

        gls_aligned = gls_features.reindex(target_index, method="ffill", limit=4)
        parts.append(gls_aligned)
        logger.debug(f"{symbol}: global LS features built")
    else:
        logger.debug(f"{symbol}: no global LS data, skipping")

    # ── Top Trader Position L/S Ratio ──────────────────────────────────────────
    tp_path = raw_dir / f"{symbol}_ls_top_position_15m.parquet"
    tp_df = _load_parquet_safe(tp_path)
    if not tp_df.empty and "longShortRatio" in tp_df.columns:
        tp = tp_df["longShortRatio"].rename("ls_top_position_ratio")
        tp_features = pd.DataFrame(index=tp_df.index)
        tp_features["ls_top_position_ratio"] = tp
        tp_features["ls_top_position_zscore"] = _zscore(tp, ls_window)
        # Top trader divergence from global (smart money vs crowd)
        if not gls_df.empty and "longShortRatio" in gls_df.columns:
            gls_aligned_raw = gls_df["longShortRatio"].reindex(tp_df.index, method="ffill", limit=4)
            tp_features["ls_top_vs_global_div"] = tp - gls_aligned_raw

        tp_aligned = tp_features.reindex(target_index, method="ffill", limit=4)
        parts.append(tp_aligned)
        logger.debug(f"{symbol}: top trader position LS features built")
    else:
        logger.debug(f"{symbol}: no top trader position LS data, skipping")

    # ── Taker Buy/Sell Volume Ratio ────────────────────────────────────────────
    taker_path = raw_dir / f"{symbol}_taker_ratio_15m.parquet"
    taker_df = _load_parquet_safe(taker_path)
    if not taker_df.empty and "buySellRatio" in taker_df.columns:
        taker = taker_df["buySellRatio"].rename("taker_ratio")
        taker_features = pd.DataFrame(index=taker_df.index)
        taker_features["taker_ratio"] = taker

        # Z-score of taker ratio
        taker_features["taker_ratio_zscore"] = _zscore(taker, taker_window)

        # Rolling sum of buy / sell volume for imbalance
        if "buyVol" in taker_df.columns and "sellVol" in taker_df.columns:
            buy_vol = taker_df["buyVol"]
            sell_vol = taker_df["sellVol"]
            net_vol = buy_vol - sell_vol
            total_vol = buy_vol + sell_vol + 1e-9
            taker_features["taker_net_vol_pct"] = (net_vol / total_vol).clip(-1.0, 1.0)
            # Rolling cumulative taker imbalance (24h)
            taker_features["taker_imbalance_24h"] = (
                taker_features["taker_net_vol_pct"].rolling(taker_window, min_periods=taker_window // 2).mean()
            )

        # Extreme taker pressure flags
        taker_features["taker_extreme_buy"] = (taker_features["taker_ratio_zscore"] > extreme_z).astype(int)
        taker_features["taker_extreme_sell"] = (taker_features["taker_ratio_zscore"] < -extreme_z).astype(int)

        taker_aligned = taker_features.reindex(target_index, method="ffill", limit=4)
        parts.append(taker_aligned)
        logger.debug(f"{symbol}: taker ratio features built")
    else:
        logger.debug(f"{symbol}: no taker ratio data, skipping")

    if not parts:
        logger.info(f"{symbol}: no market positioning data available — all features will be NaN")
        return pd.DataFrame(index=target_index)

    result = pd.concat(parts, axis=1)
    result = result[~result.index.duplicated(keep="last")]
    logger.info(f"{symbol}: market positioning — {result.shape[1]} features, {result.notna().mean().mean():.1%} non-null")
    return result
