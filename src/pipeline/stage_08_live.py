import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

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
from src.models.model_versioning import get_latest_model
from src.models.primary_model import load_model
from src.models.meta_labeler import load_meta_model, build_meta_features
from src.portfolio.position_sizer import (
    compute_position_size,
    get_growth_gate_limits,
    check_portfolio_capacity,
)
from src.utils.logger import get_logger
from src.utils.state_manager import load_state, update_equity

logger = get_logger("stage_08_live")

_model_cache: dict = {}

_SYMBOLS_PATH = Path("config/symbols.yaml")
_TRADE_LOG_PATH = Path("results/live_trade_log.csv")
_BAR_CLOSE_BUFFER = 5    # seconds after bar close before sampling


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
    from src.models.model_versioning import get_latest_model
    forecast = [s for s in all_symbols if get_latest_model(s, primary_timeframe, model_type="primary") is not None]
    logger.info(f"Forecast symbols ({len(forecast)} with trained models): {forecast}")
    return forecast


def _get_trade_limit(cfg, state: dict) -> int:
    # Growth gate: max simultaneous open positions allowed based on equity
    equity = float(state.get("account", {}).get("current_equity", 0.0))
    max_symbols, _ = get_growth_gate_limits(equity, cfg)
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
        return None
    try:
        return load_meta_model(symbol, primary_tf, entry["version"], cfg.data.models_dir)
    except Exception as exc:
        logger.debug(f"{symbol}: no meta model loaded: {exc}")
        return None


def _predict(model, calibrator, meta_model, feature_series: pd.Series) -> tuple[float, float]:
    X_arr = feature_series.values.reshape(1, -1)

    raw_prob = float(model.predict_proba(X_arr)[0][1])
    if calibrator is not None:
        primary_prob = float(calibrator.predict(np.array([raw_prob]))[0])
    else:
        primary_prob = raw_prob

    if meta_model is not None:
        # Meta model was trained on build_meta_features() output — NOT primary features.
        # Reconstruct the same feature space: prob_long/short, confidence, vol, zscore, ofi.
        # Regime probs are not available live per-bar so we pass None (meta_df will have NaN cols).
        oof_proba_1bar = np.array([[1.0 - primary_prob, primary_prob]])
        # Column names may have suffixes (e.g. realized_vol_20, ofi_20) — find first match
        _idx = feature_series.index
        _vol_col = next((c for c in _idx if c.startswith("realized_vol")), None)
        _zscore_col = next((c for c in _idx if c.startswith("volume_zscore")), None)
        _ofi_col = next((c for c in _idx if c.startswith("ofi")), None)
        realized_vol = pd.Series([feature_series[_vol_col] if _vol_col else np.nan])
        volume_zscore = pd.Series([feature_series[_zscore_col] if _zscore_col else np.nan])
        ofi = pd.Series([feature_series[_ofi_col] if _ofi_col else np.nan])
        meta_X = build_meta_features(oof_proba_1bar, None, realized_vol, volume_zscore, ofi)
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

    return primary_prob, signal_strength


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
                    "regime": "unknown",
                    "signal_strength": 0.0,
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
        logger.info(f"Connected [{trading_mode}] — {datetime.fromtimestamp(server_ms/1000, tz=timezone.utc).strftime('%H:%M:%S UTC')}  tf={primary_tf}  floor={signal_floor}")
    except Exception as exc:
        logger.error(f"Cannot connect to Binance: {exc}")
        raise

    state = load_state()
    forecast_symbols = _get_forecast_symbols(all_symbols, primary_tf)

    # Filter 1: artifacts (imputer/scaler must exist)
    _missing = [s for s in forecast_symbols
                if not (Path(cfg.data.checkpoints_dir) / "imputers" / f"imputer_{s}_15m.pkl").exists()
                or not (Path(cfg.data.checkpoints_dir) / "imputers" / f"scaler_{s}_15m.pkl").exists()]
    forecast_symbols = [s for s in forecast_symbols if s not in _missing]
    if _missing:
        logger.debug(f"Excluded (missing artifacts): {_missing}")

    # Filter 2: negative backtest Sharpe
    _metrics_path = Path(cfg.data.results_dir) / "per_symbol_metrics.csv"
    _bad: set = set()
    if _metrics_path.exists():
        import pandas as _pd
        _pm = _pd.read_csv(_metrics_path)
        _bad = set(_pm.loc[_pm["sharpe"] < 0, "symbol"].tolist())
        forecast_symbols = [s for s in forecast_symbols if s not in _bad]

    # Filter 3: structurally untradeable — exclude only if max_qty × price < min_notional.
    # This means the exchange literally cannot fill even the minimum order for this symbol.
    # Symbols where max_qty × price < our target size are NOT excluded — order_manager
    # will cap qty to max_qty automatically (size gets bumped/capped, not skipped).
    _untradeable = []
    _probe_errors = []
    logger.info("Filter 3: checking structural tradeability (max_qty × price >= min_notional)")

    for _sym in forecast_symbols:
        try:
            _price = float(client.get_klines(_sym, primary_tf, limit=1)["close"].iloc[-1])
            _max_qty = client.get_max_qty(_sym, _price)
            _min_notional = client.get_min_notional(_sym)
            _max_notional = _max_qty * _price
            if _max_notional < _min_notional:
                logger.info(f"{_sym}: max notional {_max_notional:.2f} < min_notional {_min_notional:.2f} — structurally untradeable, excluded")
                _untradeable.append(_sym)
            else:
                logger.debug(f"{_sym}: ok — max_notional={_max_notional:.2f} min_notional={_min_notional:.2f}")
        except Exception as _e:
            _probe_errors.append(_sym)
    forecast_symbols = [s for s in forecast_symbols if s not in _untradeable]

    logger.info(
        f"Active symbols: {len(forecast_symbols)} "
        f"(excluded: {len(_missing)} no artifacts, {len(_bad)} neg sharpe, "
        f"{len(_untradeable)} untradeable, {len(_probe_errors)} probe errors kept)"
    )

    # Pre-load models
    for _sym in forecast_symbols:
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
                logger.info(f"Wallet locked for {today_utc}: ${wallet_today:.2f} → size_usd per posisi = ${wallet_today * leverage:.2f} ({leverage}×)")

            state = load_state()
            trade_limit = _get_trade_limit(cfg, state)
            open_count = len(order_manager.positions)

            # Position reconciliation: every bar, check API vs local dict for orphan/ghost positions
            _reconcile_positions_from_api(client, order_manager, trading_mode)

            daily_pnl_pct = _compute_daily_pnl_pct(_TRADE_LOG_PATH, equity)
            daily_target_hit = daily_pnl_pct >= daily_profit_target

            # Rolling Sharpe guardrail: halve sizing if 7-day Sharpe is negative
            _rolling_sharpe = _compute_rolling_sharpe(_TRADE_LOG_PATH, days=7)
            if not (isinstance(_rolling_sharpe, float) and _rolling_sharpe != _rolling_sharpe):  # not nan
                if _rolling_sharpe < 0:
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

            for symbol in tqdm(forecast_symbols, desc="scoring", unit="sym", position=0, leave=False):
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
                        forecast_symbols = [s for s in forecast_symbols if s != symbol]
                        _sym_error_count.pop(symbol, None)
                order_manager.heartbeat()

            # Phase 2: sort candidates by signal_strength desc, open top-N positions
            scored.sort(key=lambda x: x["signal_strength"], reverse=True)
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
                        # Use actual filled size from position dict (may be capped by maxQty)
                        actual_size = order_manager.positions.get(symbol, {}).get("size_usd", cand["volume_usdt"])
                        logger.info(
                            f"{symbol}: ENTERED {cand['direction_str'].upper()} — "
                            f"wallet={equity:.2f} USDT volume={actual_size:.2f} USDT ({cand['leverage']}×) "
                            f"tp={cand['tp_pct']:.2%} sl={cand['sl_pct']:.2%} signal={cand['signal_strength']:.3f} orderId={order_id}"
                        )
                    else:
                        cand["action"] = "FAILED"
                except Exception as exc:
                    logger.error(f"{symbol}: entry error: {exc}")
                    cand["action"] = "FAILED"
                order_manager.heartbeat()

            # Bar summary — show entered/failed/skipped at a glance
            _entered = [s["symbol"] for s in bar_signals if s.get("action") == "ENTERED"]
            _failed  = [s["symbol"] for s in bar_signals if s.get("action") == "FAILED"]
            _skipped = [s["symbol"] for s in bar_signals if s.get("action", "").startswith("SKIP")]
            logger.info(
                f"Bar summary — ENTERED: {_entered} | FAILED: {_failed} | SKIPPED: {len(_skipped)}"
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
    # Wilder ATR on the fetched klines — matches triple-barrier logic in labels
    high = klines_df["high"]
    low = klines_df["low"]
    close = klines_df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    return float(atr_series.iloc[-1]) if not atr_series.empty else float(close.iloc[-1] * 0.01)


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

    klines_1h, klines_4h, klines_1d = None, None, None
    for tf, limit in [("1h", 500), ("4h", 200), ("1d", 100)]:
        try:
            df_htf = client.get_klines(symbol, tf, limit=limit)
            if tf == "1h":
                klines_1h = df_htf
            elif tf == "4h":
                klines_4h = df_htf
            elif tf == "1d":
                klines_1d = df_htf
        except Exception as _htf_exc:
            logger.warning(f"{symbol}: could not fetch {tf} klines — HTF features will be missing: {_htf_exc}")

    feature_series = compute_live_features(
        symbol, cfg, klines_df,
        klines_1h=klines_1h, klines_4h=klines_4h, klines_1d=klines_1d,
        btc_klines_15m=btc_klines_15m,
        client=client,
    )

    cached = _model_cache.get(symbol, {})
    primary_model, calibrator = cached.get("primary", (None, None))
    meta_model = cached.get("meta", None)
    if primary_model is None:
        logger.warning(f"{symbol}: no cached model, skipping bar")
        return None

    primary_prob, signal_strength = _predict(primary_model, calibrator, meta_model, feature_series)

    # Sync fills — detect TP/SL hits and DEMO simulated exits
    fill = order_manager.sync_fills(symbol)
    if fill is not None:
        pnl_usd = fill['pnl_pct'] * fill['size_usd']
        pnl_style = "+" if fill['pnl_pct'] >= 0 else ""
        logger.info(f"{symbol}: CLOSED — P&L {pnl_style}{pnl_usd:.2f} USDT ({pnl_style}{fill['pnl_pct']:.2%})")

    regime = state.get("model_tiers", {}).get(symbol, {}).get("last_regime", "unknown")

    direction_int = 0
    if abs(primary_prob - 0.5) >= float(cfg.portfolio.dead_zone_direction):
        direction_int = 1 if primary_prob >= 0.5 else -1

    direction_str = "long" if primary_prob >= 0.5 else "short"

    # Sizing: wallet_today × leverage — locked once per UTC day, same for every position.
    # wallet_today does NOT decrease as positions open (full wallet per posisi, sesuai aturan sizing).
    volume_usdt = wallet_today * float(leverage)
    max_volume_usdt = float(getattr(cfg.portfolio, "max_volume_usdt", 0))
    if max_volume_usdt > 0 and volume_usdt > max_volume_usdt:
        volume_usdt = max_volume_usdt
    logger.debug(f"{symbol}: wallet_today={wallet_today:.2f} × {leverage}× = volume {volume_usdt:.2f} USDT")

    entry_price = float(klines_df["close"].iloc[-1])

    tp_fixed = float(getattr(cfg.growth_gate, "tp_fixed_pct", 0))
    sl_fixed = float(getattr(cfg.growth_gate, "sl_fixed_pct", 0))
    if tp_fixed > 0 and sl_fixed > 0:
        tp_pct, sl_pct = tp_fixed, sl_fixed
    else:
        atr = _compute_atr(klines_df)
        atr_pct = atr / max(entry_price, 1e-12)
        tp_pct = min(max(float(cfg.labels.tp_min_pct), atr_pct * float(cfg.labels.tp_atr_mult)), float(cfg.labels.tp_max_pct))
        sl_pct = min(max(float(cfg.labels.sl_min_pct), atr_pct * float(cfg.labels.sl_atr_mult)), float(cfg.labels.sl_max_pct))

    sig_info = {
        "symbol": symbol,
        "primary_prob": round(primary_prob, 4),
        "signal_strength": round(signal_strength, 4),
        "direction": direction_int,
        "direction_str": direction_str,
        "regime": regime,
        "entry_price": entry_price,
        "volume_usdt": volume_usdt,
        "leverage": leverage,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "action": "NO_SIGNAL",
    }

    if symbol in order_manager.positions:
        sig_info["action"] = "HOLD"
        return sig_info

    if skip_new_entries:
        sig_info["action"] = "SKIP_DAILY"
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
    )
