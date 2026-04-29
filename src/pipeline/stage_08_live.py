import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from src.utils.cli_progress import LiveBarPanel, console as _rich_console

# Load .env before any Binance client is instantiated so API keys are in os.environ
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed — keys must be set in environment manually

from src.dashboard.live_dashboard import LiveDashboard
from src.execution.binance_client import BinanceClient
from src.execution.live_features import compute_live_features, get_lookback_bars_needed
from src.execution.order_manager import OrderManager
from src.pipeline.live_vs_training import write_rollup as _write_drift_rollup
from src.models.model_versioning import get_latest_model
from src.models.primary_model import load_model
from src.models.meta_labeler import load_meta_model, build_meta_features
from src.portfolio.position_sizer import (
    compute_position_size,
    get_growth_gate_limits,
    check_portfolio_capacity,
)
from src.features.technical import compute_atr as _compute_atr_series
from src.utils.logger import get_logger
from src.utils.state_manager import load_state, save_state, update_equity
from src.utils.telegram_notifier import (
    notify_entry as _tg_entry,
    notify_exit as _tg_exit,
    notify_daily_summary as _tg_daily,
    notify_heartbeat as _tg_heartbeat,
    notify_alert as _tg_alert,
    notify_maintenance as _tg_maintenance,
)

logger = get_logger("stage_08_live")

_model_cache: dict = {}

_SYMBOLS_PATH = Path("config/symbols.yaml")
_TRADE_LOG_PATH = Path("results/live_trade_log.csv")
_BAR_CLOSE_BUFFER = 5    # seconds after bar close before sampling
_RETRAIN_FLAG = Path(".retrain_pending")  # touch this file to trigger graceful shutdown


def _parse_timeframe_seconds(tf: str) -> int:
    # Convert timeframe string to seconds: "15m" -> 900, "1h" -> 3600, "4h" -> 14400, "1d" -> 86400
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    raise ValueError(f"Cannot parse timeframe seconds from: {tf!r}")


def _load_symbol_list() -> list[str]:
    with open(_SYMBOLS_PATH) as f:
        raw = yaml.safe_load(f)
    # Return all symbols; tier-based filtering happens at runtime using model registry
    return list(raw.get("symbols", {}).keys())


def _seconds_until_next_bar(bar_seconds: int) -> float:
    now_ts = time.time()
    # Next bar boundary is the next multiple of bar_seconds from epoch
    next_boundary = (int(now_ts / bar_seconds) + 1) * bar_seconds
    return next_boundary - now_ts


def _get_forecast_symbols(all_symbols: list[str], primary_timeframe: str) -> list[str]:
    # Return all symbols that have a registered primary model — these get forecasted every bar
    forecast = [s for s in all_symbols if get_latest_model(s, primary_timeframe, model_type="primary") is not None]
    logger.info(f"Forecast symbols ({len(forecast)} with trained models): {forecast}")
    return forecast


def _get_vol_mult(wallet: float, cfg) -> float:
    tiers = list(cfg.growth_gate.tiers)
    for tier in sorted(tiers, key=lambda t: float(t.max_equity)):
        if wallet <= float(tier.max_equity):
            return float(getattr(tier, "vol_mult", 2.0))
    return 2.0


def _get_trade_limit(cfg, state: dict) -> int:
    # Growth gate: max simultaneous open positions, gated by daily P&L limits
    equity = float(state.get("account", {}).get("current_equity", 0.0))
    max_symbols, _ = get_growth_gate_limits(equity, cfg)

    wallet_day_start = float(state.get("wallet_day_start", equity))
    if wallet_day_start > 0:
        daily_pnl_pct = (equity - wallet_day_start) / wallet_day_start
        profit_cap = float(getattr(cfg.growth_gate, "daily_profit_target_pct", 0.04))
        loss_limit  = float(getattr(cfg.growth_gate, "daily_loss_limit_pct",   0.05))
        if daily_pnl_pct >= profit_cap:
            logger.info(f"Daily profit cap hit ({daily_pnl_pct:.2%} >= +{profit_cap:.0%}) — no new entries today")
            return 0
        if daily_pnl_pct <= -loss_limit:
            logger.info(f"Daily hard stop hit ({daily_pnl_pct:.2%} <= -{loss_limit:.0%}) — no new entries today")
            return 0

    return max_symbols


def _load_primary_model(symbol: str, cfg):
    primary_tf = str(cfg.data.primary_timeframe)
    entry = get_latest_model(symbol, primary_tf, model_type="primary", cfg=cfg)
    if entry is None:
        return None, None
    try:
        model, calibrator = load_model(symbol, primary_tf, entry["version"], cfg.data.models_dir)
        return model, calibrator
    except Exception as exc:
        logger.warning(f"{symbol}: could not load primary model: {exc}")
        return None, None


def _load_meta(symbol: str, cfg):
    primary_tf = str(cfg.data.primary_timeframe)
    entry = get_latest_model(symbol, primary_tf, model_type="meta", cfg=cfg)
    if entry is None:
        return None, None
    try:
        model = load_meta_model(symbol, primary_tf, entry["version"], cfg.data.models_dir)
        return model, entry
    except Exception as exc:
        logger.debug(f"{symbol}: no meta model loaded: {exc}")
        return None, None


def _predict(model, calibrator, meta_model, feature_series: pd.Series, meta_entry=None) -> tuple[float, float, float]:
    # Returns (primary_prob, signal_strength, meta_prob). meta_prob=0.5 when no meta model.
    X_arr = feature_series.values.reshape(1, -1)

    raw_prob = float(model.predict_proba(X_arr)[0][1])
    if calibrator is not None:
        primary_prob = float(calibrator.predict(np.array([raw_prob]))[0])
    else:
        primary_prob = raw_prob

    meta_prob = 0.5
    if meta_model is not None:
        # Meta model trained on build_meta_features() output.
        # Use raw_prob (pre-calibration) to match stage_05 training distribution — Platt
        # scaling changes the probability distribution and would cause feature drift.
        oof_proba_1bar = np.array([[1.0 - raw_prob, raw_prob]])

        # Use raw (pre-scale) values captured before StandardScaler in compute_live_features.
        # stage_05 training reads these from the unscaled features parquet — must match.
        # Exact column names must match stage_05: rv_daily, volume_surprise_20, ofi_20, etc.
        _meta_raw = feature_series.attrs.get("meta_raw", {})
        realized_vol  = pd.Series([_meta_raw.get("rv_daily",           np.nan)])
        volume_zscore = pd.Series([_meta_raw.get("volume_surprise_20", np.nan)])
        ofi           = pd.Series([_meta_raw.get("ofi_20",             np.nan)])
        spread_series = pd.Series([_meta_raw.get("spread_proxy_20",    np.nan)])
        atr_series    = pd.Series([_meta_raw.get("atr_14",             np.nan)])

        # Extract regime_prob_* from feature_series — these are available because
        # compute_live_features loads HMM artifacts and appends regime_prob_* columns
        # before scaling. Passing them to build_meta_features matches the training path
        # in stage_05 where regime probs were also included.
        _regime_cols = [c for c in feature_series.index if c.startswith("regime_prob_")]
        if _regime_cols:
            _regime_vals = pd.DataFrame(
                {c: [float(feature_series[c])] for c in _regime_cols}
            )
        else:
            _regime_vals = None

        meta_X = build_meta_features(
            oof_proba_1bar, _regime_vals, realized_vol, volume_zscore, ofi,
            spread_series=spread_series, atr_series=atr_series,
        )
        # Align to training feature schema to prevent shape mismatch
        if meta_entry is not None and meta_entry.get("feature_names"):
            expected_cols = meta_entry["feature_names"]
            for col in expected_cols:
                if col not in meta_X.columns:
                    meta_X[col] = 0.0
            meta_X = meta_X[expected_cols]
        meta_X_arr = np.nan_to_num(meta_X.values, nan=0.0)
        try:
            meta_prob = float(meta_model.predict_proba(meta_X_arr)[0][1])
        except Exception:
            meta_prob = 0.5  # fallback if feature count mismatch (old model)
        # signal_strength = directional confidence × meta_prob
        # confidence = distance from 0.5 (works symmetrically for long AND short)
        confidence = abs(primary_prob - 0.5) * 2.0  # 0=uncertain, 1=max confident
        signal_strength = confidence * meta_prob
    else:
        signal_strength = abs(primary_prob - 0.5) * 2.0

    return primary_prob, signal_strength, meta_prob


def _compute_daily_pnl_pct(trade_log_path: Path, equity: float) -> float:
    # Read live_trade_log.csv, filter to today UTC, sum realized pnl_pct weighted by size_usd
    if not trade_log_path.exists() or equity <= 0:
        return 0.0
    try:
        df = pd.read_csv(trade_log_path)
        if df.empty or "timestamp_exit" not in df.columns:
            return 0.0
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # timestamp_exit is ISO string — filter rows where date portion matches today UTC
        today_mask = df["timestamp_exit"].astype(str).str.startswith(today_str)
        today_df = df[today_mask]
        if today_df.empty:
            return 0.0
        # pnl in USD = pnl_pct * size_usd; daily_pnl_pct = sum(pnl_usd) / equity
        pnl_usd = (today_df["pnl_pct"] * today_df["size_usd"]).sum()
        return float(pnl_usd / equity)
    except Exception as exc:
        logger.warning(f"Could not compute daily PnL: {exc}")
        return 0.0


def _compute_rolling_sharpe(trade_log_path: Path, days: int = 7) -> float:
    # Compute Sharpe of per-trade pnl_pct over last N days — returns nan if insufficient trades
    if not trade_log_path.exists():
        return float("nan")
    try:
        df = pd.read_csv(trade_log_path)
        if df.empty or "timestamp_exit" not in df.columns or "pnl_pct" not in df.columns:
            return float("nan")
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days)
        df["ts"] = pd.to_datetime(df["timestamp_exit"], utc=True, errors="coerce")
        recent = df[df["ts"] >= cutoff]["pnl_pct"].dropna()
        if len(recent) < 5:  # need at least 5 trades for meaningful Sharpe
            return float("nan")
        mean_r = recent.mean()
        std_r = recent.std()
        return float(mean_r / (std_r + 1e-9)) * (len(recent) ** 0.5)  # information ratio style
    except Exception as exc:
        logger.debug(f"Rolling Sharpe compute failed: {exc}")
        return float("nan")


def _reconcile_positions_from_api(client, order_manager, trading_mode: str) -> None:
    # Sync local positions dict with exchange reality every bar.
    # Orphans (on exchange but not in local dict) are injected so the system treats them as open.
    # Ghosts (in local dict but exchange shows zero) are removed.
    try:
        api_positions = client.get_all_open_positions()
        api_syms = {p["symbol"] for p in api_positions}
        local_syms = set(order_manager.positions.keys())

        # Orphan positions: on exchange but not tracked locally (e.g. prior session, manual trade)
        # Inject them so system won't try to open a second position on top
        orphans = api_syms - local_syms
        if orphans:
            logger.warning(f"RECONCILE: {len(orphans)} orphan position(s) on exchange not in local state: {orphans}")
            for sym in orphans:
                p = next(x for x in api_positions if x["symbol"] == sym)
                amt = float(p["positionAmt"])
                entry = float(p["entryPrice"])
                upnl = float(p["unrealizedProfit"])
                direction = "long" if amt > 0 else "short"
                size_usd = abs(amt) * entry
                logger.warning(
                    f"  {sym}: amt={amt:.4f} entry={entry:.4f} upnl={upnl:.2f} "
                    f"— injecting as {direction} into local state"
                )
                # Inject minimal position record so system knows this slot is occupied
                order_manager.positions[sym] = {
                    "order_id": "RECONCILED",
                    "direction": direction,
                    "entry_price": entry,
                    "size_usd": size_usd,
                    "tp_price": None,
                    "sl_price": None,
                    "tp_order_id": None,
                    "sl_order_id": None,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "entry_epoch": time.time(),
                    "regime": "unknown",
                    "signal_strength": 0.0,
                    "tp_pct_used": 0.0,
                    "sl_pct_used": 0.0,
                    "atr_pct_at_entry": 0.0,
                    "primary_prob_at_entry": 0.5,
                    "meta_prob_at_entry": 0.5,
                }

        # Ghost positions: in local dict but exchange shows zero — already closed externally
        ghosts = local_syms - api_syms
        if ghosts:
            logger.warning(f"RECONCILE: {len(ghosts)} ghost position(s) in local state but closed on exchange: {ghosts}")
            for sym in ghosts:
                logger.warning(f"  {sym}: removing from local positions dict")
                order_manager.positions.pop(sym, None)
    except Exception as exc:
        logger.debug(f"Position reconciliation failed: {exc}")


def _rotate_logs(logs_dir: Path, keep_days: int = 2) -> None:
    # Delete log files older than keep_days. Keeps today and yesterday.
    cutoff = datetime.now(timezone.utc).date()
    for log_file in logs_dir.glob("*.log"):
        try:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc).date()
            age = (cutoff - mtime).days
            if age >= keep_days:
                log_file.unlink()
                logger.debug(f"Log rotated: {log_file.name} (age={age}d)")
        except Exception:
            pass


def run(cfg, **kwargs) -> None:
    logger.info("=== Stage 8: Live Execution ===")

    # Auto-rotate logs — keep today + yesterday only
    _rotate_logs(Path("logs"), keep_days=2)

    trading_mode = str(cfg.trading.mode).upper()
    if trading_mode == "MAINNET":
        _confirm = os.environ.get("CONFIRM_MAINNET_TRADING", "").lower()
        if _confirm != "yes":
            raise RuntimeError(
                "MAINNET mode requires CONFIRM_MAINNET_TRADING=yes in environment. "
                "Set this env var only after completing demo_trades_required demo trades."
            )
        logger.warning("!!! MAINNET MODE — REAL MONEY ORDERS WILL BE PLACED !!!")

    # --- FIX 6: warn if mainnet API key is loaded but mode=DEMO ---
    if trading_mode == "DEMO" and os.environ.get("BINANCE_API_KEY"):
        logger.warning(
            "Mainnet BINANCE_API_KEY is present in environment but mode=DEMO — "
            "ensure you are not accidentally using mainnet keys"
        )

    primary_tf = str(cfg.data.primary_timeframe)
    bar_seconds = _parse_timeframe_seconds(primary_tf)

    all_symbols = _load_symbol_list()
    client = BinanceClient(cfg)
    # Pass trading_mode so OrderManager can track demo trades correctly
    order_manager = OrderManager(client, cfg, _TRADE_LOG_PATH, mode=trading_mode)
    dashboard = LiveDashboard(cfg)
    lookback_needed = get_lookback_bars_needed(cfg)
    signal_floor = float(cfg.portfolio.signal_floor_prob)
    daily_profit_target = float(cfg.backtest.daily_profit_target_pct)

    try:
        server_ms = client.get_server_time()
        server_dt = datetime.fromtimestamp(server_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S UTC")
        logger.info(f"Connected [{trading_mode}] — {server_dt}  tf={primary_tf}  floor={signal_floor}")
        _rich_console.print(
            f"\n[bold cyan]━━━ STAGE 8 LIVE ━━━[/bold cyan]  "
            f"mode=[yellow]{trading_mode}[/yellow]  "
            f"tf=[cyan]{primary_tf}[/cyan]  "
            f"server=[dim]{server_dt}[/dim]"
        )
    except Exception as exc:
        logger.error(f"Cannot connect to Binance: {exc}")
        raise

    state = load_state()
    forecast_symbols = _get_forecast_symbols(all_symbols, primary_tf)

    # Filter 0: user-defined permanent exclude list from config
    _manual_exclude = set(list(getattr(cfg.trading, "exclude_symbols", [])))
    if _manual_exclude:
        _was = len(forecast_symbols)
        forecast_symbols = [s for s in forecast_symbols if s not in _manual_exclude]
        logger.info(f"Filter 0: {_was - len(forecast_symbols)} symbols excluded by exclude_symbols config: {_manual_exclude}")

    # Filter 1: artifacts (imputer/scaler must exist)
    _missing = [s for s in forecast_symbols
                if not (Path(cfg.data.checkpoints_dir) / "imputers" / f"imputer_{s}_15m.pkl").exists()
                or not (Path(cfg.data.checkpoints_dir) / "imputers" / f"scaler_{s}_15m.pkl").exists()]
    forecast_symbols = [s for s in forecast_symbols if s not in _missing]
    if _missing:
        logger.info(f"Filter 1: {len(_missing)} excluded (missing imputer/scaler artifacts): {_missing}")

    # Filter 2: negative backtest Sharpe OR critically low hit_rate (inverted signal)
    # hit_rate < 0.40 on 2:1 R:R labels means model has inverted signal, not just bad luck
    _metrics_path = Path(cfg.data.results_dir) / "per_symbol_metrics.csv"
    _bad: set = set()
    if _metrics_path.exists():
        _pm = pd.read_csv(_metrics_path)
        _bad_sharpe = set(_pm.loc[_pm["sharpe"] < 0, "symbol"].tolist())
        _hit_rate_col = "hit_rate" if "hit_rate" in _pm.columns else None
        if _hit_rate_col:
            # Exclude only rows with real hit_rate data (n_trades > 0) below threshold.
            # hit_rate=0.0 with n_trades>0 means corrupt backtest sizing — skip the filter
            # rather than blacklisting every symbol. NaN (no trades) also skipped.
            _has_trades = _pm.get("n_trades", pd.Series(0, index=_pm.index)) > 0
            _valid_hr = _pm[_hit_rate_col].notna() & (_pm[_hit_rate_col] > 0) & _has_trades
            _bad_hr = set(_pm.loc[_valid_hr & (_pm[_hit_rate_col] < 0.40), "symbol"].tolist())
        else:
            _bad_hr = set()
        _bad = _bad_sharpe | _bad_hr
        if _bad_sharpe:
            logger.info(f"Filter 2: {len(_bad_sharpe)} excluded (negative backtest Sharpe): {sorted(_bad_sharpe)}")
        if _bad_hr - _bad_sharpe:
            logger.info(f"Filter 2: {len(_bad_hr - _bad_sharpe)} excluded (hit_rate < 0.40 — inverted signal): {sorted(_bad_hr - _bad_sharpe)}")
        forecast_symbols = [s for s in forecast_symbols if s not in _bad]

    # Filter 3: structurally untradeable — exclude only if max_qty × price < min_notional.
    # This means the exchange literally cannot fill even the minimum order for this symbol.
    # Symbols where max_qty × price < our target size are NOT excluded — order_manager
    # will cap qty to max_qty automatically (size gets bumped/capped, not skipped).
    _untradeable = []
    _probe_errors = []
    logger.info("Filter 3: checking structural tradeability (max_qty × price >= min_notional)")

    for _sym in forecast_symbols:
        order_manager.heartbeat()  # DMS keepalive — Filter 3 probes ~57 symbols × 1s > 60s timeout
        try:
            _price = float(client.get_klines(_sym, primary_tf, limit=1)["close"].iloc[-1])
            _step = client.get_qty_step(_sym)
            _min_notional = client.get_min_notional(_sym)
            # Minimum qty needed to satisfy min_notional, rounded up to nearest step
            _min_qty_needed = math.ceil((_min_notional / _price) / _step) * _step if _price > 0 else float("inf")
            _max_qty = client.get_max_qty(_sym, _price)
            # Untradeable only if even 1 lot × price < min_notional AND max_qty can't cover it
            if _min_qty_needed > _max_qty:
                logger.info(
                    f"{_sym}: structurally untradeable — need qty≥{_min_qty_needed:.4f} to meet "
                    f"min_notional={_min_notional:.2f} but max_qty={_max_qty:.4f}, excluded"
                )
                _untradeable.append(_sym)
            else:
                logger.debug(f"{_sym}: ok — min_qty_needed={_min_qty_needed:.4f} max_qty={_max_qty:.4f} price={_price:.6f}")
        except Exception as _e:
            _probe_errors.append(_sym)
    forecast_symbols = [s for s in forecast_symbols if s not in _untradeable]

    logger.info(
        f"━━━ Symbol filters complete ━━━\n"
        f"  Total in config:       {len(all_symbols)}\n"
        f"  Have trained model:    {len(forecast_symbols) + len(_missing) + len(_bad) + len(_untradeable) + len(list(_manual_exclude & set(all_symbols)))}\n"
        f"  Excluded manual list:  {len(_manual_exclude & set(all_symbols))}\n"
        f"  Excluded no artifacts: {len(_missing)}\n"
        f"  Excluded neg Sharpe/HR:{len(_bad)}\n"
        f"  Excluded untradeable:  {len(_untradeable)}\n"
        f"  ─────────────────────────────\n"
        f"  ACTIVE for trading:    {len(forecast_symbols)} → {sorted(forecast_symbols)}"
    )

    # Pre-load models
    logger.info(f"Pre-loading {len(forecast_symbols)} models...")
    for _sym in forecast_symbols:
        order_manager.heartbeat()  # DMS keepalive during model preload
        try:
            _model_cache[_sym] = {
                "primary": _load_primary_model(_sym, cfg),
                "meta": _load_meta(_sym, cfg),
            }
        except Exception as _e:
            logger.debug(f"{_sym}: model preload failed — {_e}")

    leverage = int(getattr(cfg.trading, "leverage", getattr(cfg.growth_gate, "fixed_leverage", 2)))
    wallet_today: float = 0.0   # locked once per UTC day, used for ALL position sizing that day
    wallet_today_date: str = ""  # UTC date string when wallet_today was locked

    # Circuit breaker: consecutive error counter per symbol
    # Symbol removed from forecast_symbols after 5 consecutive scoring failures
    _sym_error_count: dict[str, int] = {}
    _CIRCUIT_BREAKER_LIMIT = 5

    # Rolling Sharpe guardrail: if last 7-day Sharpe from trade log < 0, log warning + halve sizing
    _sharpe_size_factor: float = 1.0  # 1.0 = full size, 0.5 = halved

    _bar_panel = LiveBarPanel()

    try:
        while True:
            wait = _seconds_until_next_bar(bar_seconds) + _BAR_CLOSE_BUFFER
            next_bar_epoch = time.time() + wait

            # Start countdown display — updates every second in background thread
            dashboard.start_countdown(next_bar_epoch)

            # Heartbeat every 30s during bar-wait so DMS (60s timeout) never fires
            _slept = 0.0
            while _slept < wait:
                _step = min(30.0, wait - _slept)
                time.sleep(_step)
                _slept += _step
                order_manager.heartbeat()

            dashboard.stop_countdown()
            bar_start = datetime.now(timezone.utc)
            today_utc = bar_start.strftime("%Y-%m-%d")

            # Fetch live wallet balance from exchange
            try:
                acct = client.get_account()
                equity = float(acct.get("totalWalletBalance", 0.0))
                update_equity(equity)
            except Exception as _eq_exc:
                state = load_state()
                equity = float(state.get("account", {}).get("current_equity", 0.0))
                logger.warning(f"Equity fetch failed — using cached ${equity:.2f}: {_eq_exc}")

            # Lock wallet_today once per UTC day — all positions today use this value for sizing.
            # Compounding happens automatically: wallet_today resets to the new balance next day.
            if today_utc != wallet_today_date:
                wallet_today = equity
                wallet_today_date = today_utc
                # Persist wallet_day_start so _get_trade_limit can compute daily P&L vs day-open
                _state_snap = load_state()
                _state_snap["wallet_day_start"] = wallet_today
                save_state(_state_snap)
                _vol_mult_log = _get_vol_mult(wallet_today, cfg)
                logger.info(f"Wallet locked for {today_utc}: ${wallet_today:.2f} × {_vol_mult_log}x = notional ${wallet_today * _vol_mult_log:.2f} per position (leverage={leverage}x)")

            state = load_state()
            trade_limit = _get_trade_limit(cfg, state)
            open_count = len(order_manager.positions)

            # Position reconciliation: every bar, check API vs local dict for orphan/ghost positions
            _reconcile_positions_from_api(client, order_manager, trading_mode)

            # Graceful shutdown via flag file — set by vps_hot_swap.sh before model swap.
            # Wait for all positions to close naturally (TP/SL) before exiting.
            if _RETRAIN_FLAG.exists():
                open_count_now = len(order_manager.positions)
                if open_count_now == 0:
                    logger.info("Retrain flag detected and no open positions — shutting down for model hot-swap")
                    _tg_maintenance("stopped", "Stage 8 stopped cleanly. Model hot-swap in progress.")
                    _RETRAIN_FLAG.unlink(missing_ok=True)
                    return
                else:
                    _tg_alert(
                        "maintenance_waiting",
                        f"Retrain pending — waiting for {open_count_now} open position(s) to close before shutdown.",
                        cooldown_s=300,
                    )
                    logger.info(f"Retrain flag present — waiting for {open_count_now} open position(s) to close")

            daily_pnl_pct = _compute_daily_pnl_pct(_TRADE_LOG_PATH, equity)
            daily_target_hit = daily_pnl_pct >= daily_profit_target

            # Rolling Sharpe guardrail: halve sizing if 7-day Sharpe is negative. NaN → full size.
            _rolling_sharpe = _compute_rolling_sharpe(_TRADE_LOG_PATH, days=7)
            if math.isnan(_rolling_sharpe):
                _sharpe_size_factor = 1.0
            elif _rolling_sharpe < 0:
                _sharpe_size_factor = 0.5
                logger.warning(f"Rolling 7d Sharpe={_rolling_sharpe:.3f} < 0 — sizing halved to 50%")
            else:
                _sharpe_size_factor = 1.0

            btc_klines_bar: pd.DataFrame | None = None
            try:
                btc_klines_bar = client.get_klines("BTCUSDT", primary_tf, limit=lookback_needed)
            except Exception as _btc_exc:
                logger.debug(f"BTC klines prefetch failed: {_btc_exc}")

            # Phase 1: score ALL symbols — fetch klines, compute features, predict, sync fills.
            # Entry decisions are NOT made here — we collect all signals first so we can
            # sort by signal_strength and open the highest-conviction positions first.
            bar_signals = []
            scored: list[dict] = []  # candidates eligible for entry (above floor, slot available)
            # Circuit breaker removals are collected here and applied AFTER the loop —
            # modifying forecast_symbols inside the tqdm loop only affects the *next* bar
            # (tqdm iterates the original list snapshot), so we use an explicit skip set.
            _circuit_tripped: set[str] = set()

            from src.utils.cli_progress import make_symbol_progress
            _score_prog = make_symbol_progress()
            _score_task = _score_prog.add_task(
                f"[cyan]scoring {len(forecast_symbols)} symbols[/cyan]",
                total=len(forecast_symbols),
            )
            _score_prog.start()
            for symbol in forecast_symbols:
                if symbol in _circuit_tripped:
                    continue
                try:
                    sig_info = _score_symbol(
                        symbol=symbol,
                        client=client,
                        order_manager=order_manager,
                        cfg=cfg,
                        lookback_needed=lookback_needed,
                        signal_floor=signal_floor,
                        equity=equity,
                        wallet_today=wallet_today * _sharpe_size_factor,
                        leverage=leverage,
                        state=state,
                        trade_limit=trade_limit,
                        skip_new_entries=daily_target_hit,
                        btc_klines_15m=btc_klines_bar,
                    )
                    if sig_info is not None:
                        bar_signals.append(sig_info)
                        if sig_info.get("action") == "CANDIDATE":
                            scored.append(sig_info)
                    # Circuit breaker: reset error count on success
                    _sym_error_count.pop(symbol, None)
                except Exception as exc:
                    logger.error(f"{symbol}: {exc}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    # Circuit breaker: increment error count
                    _sym_error_count[symbol] = _sym_error_count.get(symbol, 0) + 1
                    if _sym_error_count[symbol] >= _CIRCUIT_BREAKER_LIMIT:
                        logger.warning(
                            f"Circuit breaker: {symbol} failed {_sym_error_count[symbol]} consecutive bars "
                            f"— removing from forecast_symbols"
                        )
                        _circuit_tripped.add(symbol)
                        _sym_error_count.pop(symbol, None)
                _score_prog.advance(_score_task)
                order_manager.heartbeat()

            _score_prog.stop()

            # Apply circuit breaker removals — takes effect from the next bar onwards
            if _circuit_tripped:
                forecast_symbols = [s for s in forecast_symbols if s not in _circuit_tripped]

            # Phase 2: sort by composite score = signal_strength × tp_reach_score.
            # tp_reach_score = atr_pct / tp_pct: how many ATRs the price needs to move to hit TP.
            # Higher value → price volatility is large relative to TP distance → faster TP hit.
            # This prevents picking a slow coin when a volatile one has the same signal quality.
            for _c in scored:
                _c["composite_score"] = _c["signal_strength"] * _c.get("tp_reach_score", 1.0)
            scored.sort(key=lambda x: x["composite_score"], reverse=True)
            open_count = len(order_manager.positions)
            slots_left = trade_limit - open_count

            for cand in scored:
                if slots_left <= 0:
                    cand["action"] = "SKIP_LIMIT"
                    continue
                symbol = cand["symbol"]
                # Re-check: another symbol in this loop may have just opened a position
                if symbol in order_manager.positions:
                    cand["action"] = "HOLD"
                    continue
                try:
                    order_id = _enter_position(
                        sig_info=cand,
                        client=client,
                        order_manager=order_manager,
                        cfg=cfg,
                        equity=equity,
                    )
                    if order_id:
                        cand["action"] = "ENTERED"
                        slots_left -= 1
                        actual_size = order_manager.positions.get(symbol, {}).get("size_usd", cand["volume_usdt"])
                        logger.info(
                            f"{symbol}: ENTERED {cand['direction_str'].upper()} — "
                            f"wallet={equity:.2f} USDT volume={actual_size:.2f} USDT ({cand['leverage']}×) "
                            f"tp={cand['tp_pct']:.2%} sl={cand['sl_pct']:.2%} signal={cand['signal_strength']:.3f} orderId={order_id}"
                        )
                        # Telegram: entry notification
                        if getattr(getattr(cfg, "telegram", None), "notify_entry", True):
                            _pos = order_manager.positions.get(symbol, {})
                            _tg_entry(
                                symbol=symbol,
                                direction=cand["direction_str"],
                                entry_price=cand["entry_price"],
                                size_usd=actual_size,
                                tp_pct=cand["tp_pct"],
                                sl_pct=cand["sl_pct"],
                                signal_strength=cand["signal_strength"],
                                primary_prob=cand.get("primary_prob", 0.5),
                                meta_prob=cand.get("meta_prob", 0.5),
                                leverage=cand["leverage"],
                                regime=cand.get("regime", ""),
                            )
                    else:
                        cand["action"] = "FAILED"
                except Exception as exc:
                    logger.error(f"{symbol}: entry error: {exc}")
                    cand["action"] = "FAILED"
                order_manager.heartbeat()

            # Rich bar summary panel
            _bar_panel.print_bar_result(
                bar_signals=bar_signals,
                equity=equity,
                bar_time=bar_start,
                open_positions=order_manager.positions,
            )

            # Post-bar equity refresh for dashboard
            try:
                acct = client.get_account()
                live_equity = float(acct.get("totalWalletBalance", equity))
                update_equity(live_equity)
            except Exception:
                live_equity = equity

            _state_now = load_state()
            _demo_done = int(_state_now.get("account", {}).get("demo_trades_completed", 0))
            _demo_req = int(cfg.growth_gate.demo_trades_required)
            _daily_pnl = _compute_daily_pnl_pct(_TRADE_LOG_PATH, live_equity)

            # Refresh drift rollup (live vs training) — cheap, reads trade log + training summary
            try:
                _write_drift_rollup()
            except Exception as _rollup_exc:
                logger.debug(f"Drift rollup write failed: {_rollup_exc}")

            dashboard.update({
                "mode": trading_mode,
                "equity": live_equity,
                "daily_pnl_pct": _daily_pnl,
                "open_positions": dict(order_manager.positions),
                "signals": bar_signals,
                "demo_trades_completed": _demo_done,
                "demo_trades_required": _demo_req,
                "daily_target_pct": daily_profit_target,
            })
            try:
                dashboard.render()
            except Exception as _dash_exc:
                logger.debug(f"Dashboard render error: {_dash_exc}")

    except KeyboardInterrupt:
        open_positions = dict(order_manager.positions)
        if open_positions:
            logger.warning(f"KeyboardInterrupt — {len(open_positions)} open position(s):")
            for _sym, _pos in open_positions.items():
                _has_bracket = bool(_pos.get("tp_order_id") or _pos.get("sl_order_id"))
                logger.warning(
                    f"  {_sym}: {_pos.get('direction','?').upper()} "
                    f"entry={_pos.get('entry_price',0):.4f} "
                    f"size={_pos.get('size_usd',0):.2f} USDT "
                    f"bracket={'YES' if _has_bracket else 'NO (DMS will close if bracket missing)'}"
                )
        else:
            logger.warning("KeyboardInterrupt — no open positions at shutdown")

        if trading_mode == "DEMO":
            # DEMO: positions stay open — TP/SL monitored by sync_fills next run.
            # Do NOT issue market close (causes unnecessary fee losses).
            logger.warning("DEMO mode: positions left open on exchange. Close manually in Binance app if needed.")
            order_manager.positions.clear()
        else:
            # MAINNET: only close positions that have NO bracket orders (dangerous state).
            # Positions with bracket orders are safe — TP/SL will handle them.
            _no_bracket = [s for s, p in open_positions.items()
                           if not p.get("tp_order_id") and not p.get("sl_order_id")]
            _with_bracket = [s for s in open_positions if s not in _no_bracket]
            if _with_bracket:
                logger.info(f"MAINNET: positions with brackets left open (safe): {_with_bracket}")
                for s in _with_bracket:
                    order_manager.positions.pop(s, None)
            if _no_bracket:
                logger.warning(f"MAINNET: closing positions WITHOUT brackets (unsafe to leave): {_no_bracket}")
                for s in list(_no_bracket):
                    if s in order_manager.positions:
                        order_manager.submit_exit(s)
        logger.info("Shutdown complete")


def _compute_atr(klines_df: pd.DataFrame, period: int = 14) -> float:
    atr_series = _compute_atr_series(klines_df["high"], klines_df["low"], klines_df["close"], period)
    return float(atr_series.iloc[-1]) if not atr_series.empty else float(klines_df["close"].iloc[-1] * 0.01)


def _score_symbol(
    symbol: str,
    client: BinanceClient,
    order_manager: OrderManager,
    cfg,
    lookback_needed: int,
    signal_floor: float,
    equity: float,
    wallet_today: float,
    leverage: int,
    state: dict,
    trade_limit: int = 1,
    skip_new_entries: bool = False,
    btc_klines_15m: pd.DataFrame | None = None,
) -> dict | None:
    # Phase 1: fetch data, compute features, predict, sync fills.
    # Does NOT open positions — returns a scored signal dict.
    # action="CANDIDATE" means eligible for entry in Phase 2.
    primary_tf = str(cfg.data.primary_timeframe)

    klines_df = client.get_klines(symbol, primary_tf, limit=lookback_needed)
    if len(klines_df) < lookback_needed // 2:
        logger.warning(f"{symbol}: insufficient kline data ({len(klines_df)} bars) — skipping")
        return None
    # Drop the last (currently-forming) bar — its close is mid-bar and not a completed
    # candle. Training was done on completed bars only, so including it creates a
    # live/training feature-distribution mismatch. After shift(1) in compute_live_features
    # the penultimate bar becomes the feature row, which is always complete.
    klines_df = klines_df.iloc[:-1]

    htf: dict[str, pd.DataFrame | None] = {"1h": None, "4h": None, "1d": None}
    for tf, limit in (("1h", 500), ("4h", 200), ("1d", 100)):
        try:
            _htf_raw = client.get_klines(symbol, tf, limit=limit)
            # Drop forming bar on HTF as well
            htf[tf] = _htf_raw.iloc[:-1] if len(_htf_raw) > 1 else _htf_raw
        except Exception as _htf_exc:
            logger.warning(f"{symbol}: could not fetch {tf} klines — HTF features will be missing: {_htf_exc}")
    klines_1h, klines_4h, klines_1d = htf["1h"], htf["4h"], htf["1d"]

    feature_series = compute_live_features(
        symbol, cfg, klines_df,
        klines_1h=klines_1h, klines_4h=klines_4h, klines_1d=klines_1d,
        btc_klines_15m=btc_klines_15m,
        client=client,
    )

    cached = _model_cache.get(symbol, {})
    primary_model, calibrator = cached.get("primary", (None, None))
    meta_model, meta_entry = cached.get("meta", (None, None))
    if primary_model is None:
        logger.warning(f"{symbol}: no cached model, skipping bar")
        return None

    primary_prob, signal_strength, meta_prob = _predict(primary_model, calibrator, meta_model, feature_series, meta_entry=meta_entry)

    # Sync fills — detect TP/SL hits and DEMO simulated exits
    fill = order_manager.sync_fills(symbol)
    if fill is not None:
        pnl_usd   = fill["pnl_pct"] * fill["size_usd"]
        pnl_style = "+" if fill["pnl_pct"] >= 0 else ""
        logger.info(f"{symbol}: CLOSED — P&L {pnl_style}{pnl_usd:.2f} USDT ({pnl_style}{fill['pnl_pct']:.2%})")
        # Telegram: exit notification
        if getattr(getattr(cfg, "telegram", None), "notify_exit", True):
            _tg_exit(
                symbol=symbol,
                direction=str(fill.get("direction", "long")),
                entry_price=float(fill.get("entry_price", 0)),
                exit_price=float(fill.get("exit_price", 0)),
                pnl_pct=float(fill["pnl_pct"]),
                pnl_usd=float(pnl_usd),
                exit_reason=str(fill.get("exit_reason", "UNKNOWN")),
                bars_held=int(fill.get("bars_held", 0)),
            )

    regime = state.get("model_tiers", {}).get(symbol, {}).get("last_regime", "unknown")
    direction_str = "long" if primary_prob >= 0.5 else "short"
    # dead_zone_direction: only mark a direction (for dashboard) if conviction is outside dead zone
    direction_int = 0
    if abs(primary_prob - 0.5) >= float(cfg.portfolio.dead_zone_direction):
        direction_int = 1 if primary_prob >= 0.5 else -1

    # Sizing sesuai Est Profit.xlsx:
    # - vol_mult dari tier (wallet < $150 → 2x, $150-$2500 → 3x, >= $2500 → 2x)
    # - volume per posisi = wallet × vol_mult (TIDAK dibagi jumlah slot)
    # - leverage 10x fixed → margin per posisi = volume / 10
    vol_mult    = _get_vol_mult(wallet_today, cfg)
    volume_usdt = wallet_today * vol_mult
    logger.debug(f"{symbol}: wallet={wallet_today:.2f} × {vol_mult}x = volume {volume_usdt:.2f} USDT (leverage={leverage}x → margin={volume_usdt/max(leverage,1):.2f})")

    entry_price = float(klines_df["close"].iloc[-1])

    # Always compute ATR — used for logging/drift metrics even when fixed TP/SL override applies
    atr = _compute_atr(klines_df)
    atr_pct = atr / max(entry_price, 1e-12)
    tp_fixed = float(getattr(cfg.growth_gate, "tp_fixed_pct", 0))
    sl_fixed = float(getattr(cfg.growth_gate, "sl_fixed_pct", 0))
    if tp_fixed > 0 and sl_fixed > 0:
        tp_pct, sl_pct = tp_fixed, sl_fixed
    else:
        tp_pct = min(max(float(cfg.labels.tp_min_pct), atr_pct * float(cfg.labels.tp_atr_mult)), float(cfg.labels.tp_max_pct))
        sl_pct = min(max(float(cfg.labels.sl_min_pct), atr_pct * float(cfg.labels.sl_atr_mult)), float(cfg.labels.sl_max_pct))

    # ATR reachability score: how many ATRs is TP away?
    # atr_pct / tp_pct = 1 means one ATR move hits TP (fast); < 1 means TP > 1 ATR (slower).
    # Higher score → price volatility is large relative to TP distance → higher TP probability per bar.
    tp_reach_score = atr_pct / max(tp_pct, 1e-9)

    sig_info = {
        "symbol": symbol,
        "primary_prob": round(primary_prob, 4),
        "meta_prob": round(meta_prob, 4),
        "signal_strength": round(signal_strength, 4),
        "tp_reach_score": round(tp_reach_score, 4),
        "direction": direction_int,
        "direction_str": direction_str,
        "regime": regime,
        "entry_price": entry_price,
        "volume_usdt": volume_usdt,
        "leverage": leverage,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "atr_pct": atr_pct,
        "action": "NO_SIGNAL",
    }

    if symbol in order_manager.positions:
        sig_info["action"] = "HOLD"
        return sig_info

    if skip_new_entries:
        sig_info["action"] = "SKIP_DAILY"
        return sig_info

    # Dead zone: if model conviction is below threshold, direction is unreliable — skip.
    # Must check before signal_floor: meta_prob can push signal_strength above floor
    # even when the primary signal is near-random.
    if direction_int == 0:
        sig_info["action"] = "SKIP_DEAD_ZONE"
        return sig_info

    if signal_strength < signal_floor:
        sig_info["action"] = "SKIP_FLOOR"
        return sig_info

    # Eligible for entry — Phase 2 will decide based on rank
    sig_info["action"] = "CANDIDATE"
    return sig_info


def _enter_position(
    sig_info: dict,
    client: BinanceClient,
    order_manager: OrderManager,
    cfg,
    equity: float,
) -> str | None:
    # Phase 2: submit entry for a pre-scored candidate. Returns order_id or None.
    symbol = sig_info["symbol"]
    return order_manager.submit_entry(
        symbol=symbol,
        direction=sig_info["direction_str"],
        size_usd=sig_info["volume_usdt"],
        entry_price=sig_info["entry_price"],
        tp_pct=sig_info["tp_pct"],
        sl_pct=sig_info["sl_pct"],
        regime=sig_info["regime"],
        signal_strength=sig_info["signal_strength"],
        atr_pct=sig_info.get("atr_pct", 0.0),
        primary_prob=sig_info.get("primary_prob", 0.5),
        meta_prob=sig_info.get("meta_prob", 0.5),
    )
