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
from src.models.meta_labeler import load_meta_model
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
    entry = get_latest_model(symbol, primary_tf, model_type="primary")
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
    entry = get_latest_model(symbol, primary_tf, model_type="meta")
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
        meta_prob = float(meta_model.predict_proba(X_arr)[0][1])
        signal_strength = primary_prob * meta_prob
    else:
        signal_strength = primary_prob

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


def run(cfg, **kwargs) -> None:
    logger.info("=== Stage 8: Live Execution Loop ===")

    # --- FIX 2: MAINNET safety interlock ---
    trading_mode = str(cfg.trading.mode).upper()
    logger.info(f"=== Trading mode: {trading_mode} ===")
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
    max_margin_pct = float(cfg.portfolio.max_total_margin_pct)
    daily_profit_target = float(cfg.backtest.daily_profit_target_pct)

    # Verify connectivity — will raise if credentials or endpoint are wrong
    try:
        server_ms = client.get_server_time()
        logger.info(f"Connected to Binance — server time: {datetime.fromtimestamp(server_ms/1000, tz=timezone.utc)}")
    except Exception as exc:
        logger.error(f"Cannot connect to Binance: {exc}")
        raise

    logger.info(f"Entering bar-wait loop — timeframe={primary_tf} signal_floor={signal_floor} lookback={lookback_needed} bars")

    state = load_state()
    forecast_symbols = _get_forecast_symbols(all_symbols, primary_tf)

    # --- FIX 7: remove symbols missing imputer/scaler artifacts before first bar ---
    _valid_symbols = []
    for _sym in forecast_symbols:
        _imp = Path(cfg.data.checkpoints_dir) / "imputers" / f"imputer_{_sym}_15m.pkl"
        _scl = Path(cfg.data.checkpoints_dir) / "imputers" / f"scaler_{_sym}_15m.pkl"
        if _imp.exists() and _scl.exists():
            _valid_symbols.append(_sym)
        else:
            logger.warning(f"{_sym}: missing imputer or scaler — excluded from forecast")
    forecast_symbols = _valid_symbols
    logger.info(f"Forecast symbols after artifact check: {len(forecast_symbols)}")

    # Pre-load models for all forecast symbols at startup
    for _sym in forecast_symbols:
        try:
            _model_cache[_sym] = {
                "primary": _load_primary_model(_sym, cfg),
                "meta": _load_meta(_sym, cfg),
            }
        except Exception as _e:
            logger.warning(f"Could not pre-load model for {_sym}: {_e}")

    try:
        while True:
            wait = _seconds_until_next_bar(bar_seconds) + _BAR_CLOSE_BUFFER
            logger.info(f"Sleeping {wait:.1f}s until next bar + buffer...")
            # Heartbeat every 30s throughout the bar-wait so DMS (60s timeout) never fires
            _slept = 0.0
            _hb_interval = 30.0
            while _slept < wait:
                _step = min(_hb_interval, wait - _slept)
                time.sleep(_step)
                _slept += _step
                order_manager.heartbeat()

            bar_start = datetime.now(timezone.utc)

            state = load_state()
            equity = float(state.get("account", {}).get("current_equity", 0.0))
            trade_limit = _get_trade_limit(cfg, state)
            open_count = len(order_manager.positions)

            # FIX 3: check daily profit target before opening any new positions this bar
            daily_pnl_pct = _compute_daily_pnl_pct(_TRADE_LOG_PATH, equity)
            daily_target_hit = daily_pnl_pct >= daily_profit_target
            if daily_target_hit:
                logger.info(
                    f"Daily profit target reached ({daily_pnl_pct:.2%} >= {daily_profit_target:.2%}) "
                    "— no new position opens this bar (sync_fills and predictions still run)"
                )

            bar_signals = []  # collect per-symbol signal data for dashboard
            for symbol in tqdm(forecast_symbols, desc="stage_08", unit="sym", position=0, leave=False):
                open_count = len(order_manager.positions)  # refresh after each symbol
                try:
                    sig_info = _process_symbol(
                        symbol=symbol,
                        client=client,
                        order_manager=order_manager,
                        cfg=cfg,
                        lookback_needed=lookback_needed,
                        signal_floor=signal_floor,
                        max_margin_pct=max_margin_pct,
                        equity=equity,
                        state=state,
                        trade_limit=trade_limit,
                        open_count=open_count,
                        skip_new_entries=daily_target_hit,
                    )
                    if sig_info is not None:
                        bar_signals.append(sig_info)
                except Exception as exc:
                    logger.error(f"{symbol}: bar processing error: {exc}")
                    import traceback
                    logger.debug(traceback.format_exc())

            # Update equity from exchange after processing all symbols
            try:
                acct = client.get_account()
                live_equity = float(acct.get("totalWalletBalance", equity))
                update_equity(live_equity)
                logger.info(f"Equity updated: {live_equity:.2f} USDT")
            except Exception as exc:
                live_equity = equity
                logger.warning(f"Could not refresh equity from exchange: {exc}")

            # Render dashboard after each bar
            try:
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
                dashboard.render()
            except Exception as _dash_exc:
                logger.debug(f"Dashboard render error (non-fatal): {_dash_exc}")

            bar_elapsed = (datetime.now(timezone.utc) - bar_start).total_seconds()
            # FIX 1: was referencing undefined `active_symbols` — use forecast_symbols
            logger.info(f"Bar loop complete in {bar_elapsed:.1f}s — {len(forecast_symbols)} symbols processed")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received — cancelling all open positions before exit")
        order_manager.cancel_all_open()
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


def _process_symbol(
    symbol: str,
    client: BinanceClient,
    order_manager: OrderManager,
    cfg,
    lookback_needed: int,
    signal_floor: float,
    max_margin_pct: float,
    equity: float,
    state: dict,
    trade_limit: int = 1,
    open_count: int = 0,
    skip_new_entries: bool = False,
) -> dict | None:
    # Returns a signal-info dict for the dashboard, or None on hard failure
    primary_tf = str(cfg.data.primary_timeframe)

    # Fetch live OHLCV bars
    klines_df = client.get_klines(symbol, primary_tf, limit=lookback_needed)
    if len(klines_df) < lookback_needed // 2:
        logger.warning(f"{symbol}: insufficient kline data ({len(klines_df)} bars) — skipping")
        return None

    # Compute live features
    feature_series = compute_live_features(symbol, cfg, klines_df)

    cached = _model_cache.get(symbol, {})
    primary_model, calibrator = cached.get("primary", (None, None))
    meta_model = cached.get("meta", None)
    if primary_model is None:
        logger.warning(f"{symbol}: no cached model, skipping bar")
        return None

    primary_prob, signal_strength = _predict(primary_model, calibrator, meta_model, feature_series)

    # Sync fills — detect if a prior position was closed by TP/SL
    fill = order_manager.sync_fills(symbol)
    if fill is not None:
        logger.info(f"{symbol}: fill detected — pnl_pct={fill['pnl_pct']:.4%}")

    # Determine current HMM regime from state (stored by stage 2/4); default to "unknown"
    regime = state.get("model_tiers", {}).get(symbol, {}).get("last_regime", "unknown")

    logger.info(
        f"{symbol}: primary_prob={primary_prob:.3f} "
        f"signal={signal_strength:.3f} regime={regime} "
        f"open_position={symbol in order_manager.positions}"
    )

    # Encode direction for dashboard: 1=long, -1=short, 0=flat (near 0.5)
    if abs(primary_prob - 0.5) < float(cfg.portfolio.dead_zone_direction):
        direction_int = 0
    else:
        direction_int = 1 if primary_prob >= 0.5 else -1

    # Build base signal info dict — enriched with action below
    sig_info = {
        "symbol": symbol,
        "primary_prob": round(primary_prob, 4),
        "signal_strength": round(signal_strength, 4),
        "direction": direction_int,
        "action": "NO_SIGNAL",
    }

    # Skip if already in a position for this symbol
    if symbol in order_manager.positions:
        sig_info["action"] = "HOLD"
        return sig_info

    # FIX 3: daily profit target gate — predictions still run, but no new entries
    if skip_new_entries:
        sig_info["action"] = "SKIP_DAILY"
        return sig_info

    # Skip if signal is below floor
    if signal_strength < signal_floor:
        sig_info["action"] = "SKIP_FLOOR"
        return sig_info

    # Growth gate: skip opening new position if at max open positions
    if open_count >= trade_limit:
        logger.debug(f"{symbol}: signal={signal_strength:.3f} but trade_limit={trade_limit} reached ({open_count} open) — no new entry")
        sig_info["action"] = "SKIP_LIMIT"
        return sig_info

    # Size the position
    half_kelly = float(cfg.portfolio.kelly_fraction) * 0.5  # cfg already stores the full fraction; halve it

    # Determine leverage from growth gate tier
    _, leverage = get_growth_gate_limits(equity, cfg)

    pos_info = compute_position_size(
        meta_prob=signal_strength,
        half_kelly=half_kelly,
        equity=equity,
        leverage=float(leverage),
        cfg=cfg,
    )

    # FIX 4: get scale_factor from portfolio capacity check and apply it
    current_positions_margin = {
        sym: {"margin": data.get("size_usd", 0) / max(float(leverage), 1)}
        for sym, data in order_manager.positions.items()
    }
    scale_factor, should_skip = check_portfolio_capacity(
        current_positions=current_positions_margin,
        new_position={"margin": pos_info["margin"]},
        total_equity=equity,
        cfg=cfg,
    )
    if should_skip:
        logger.info(f"{symbol}: portfolio margin limit reached — skipping entry")
        sig_info["action"] = "SKIP_LIMIT"
        return sig_info
    if scale_factor < 1.0:
        # Scale down position proportionally when approaching soft limit
        pos_info["margin"] = pos_info["margin"] * scale_factor
        pos_info["size_usd"] = pos_info.get("size_usd", pos_info["margin"]) * scale_factor
        pos_info["notional"] = pos_info["notional"] * scale_factor
        logger.debug(f"{symbol}: portfolio soft-limit — position scaled to {scale_factor:.2f}×")

    # Use last close as entry price proxy (market order will fill near this)
    entry_price = float(klines_df["close"].iloc[-1])

    # FIX 2: compute ATR-based TP/SL that matches engine.py logic
    atr = _compute_atr(klines_df)
    tp_atr_mult = float(cfg.labels.tp_atr_mult)
    sl_atr_mult = float(cfg.labels.sl_atr_mult)
    tp_min_pct = float(cfg.labels.tp_min_pct)
    sl_min_pct = float(cfg.labels.sl_min_pct)
    atr_pct = atr / max(entry_price, 1e-12)
    # ATR-based percentage, floored by the configured minimum
    tp_pct = max(tp_min_pct, atr_pct * tp_atr_mult)
    sl_pct = max(sl_min_pct, atr_pct * sl_atr_mult)

    # Direction: model predicts probability of upward move (label=1 → long)
    direction = "long" if primary_prob >= 0.5 else "short"
    # Direction-aware TP/SL is handled inside order_manager.submit_entry already
    # (long: tp = entry*(1+tp_pct), sl = entry*(1-sl_pct); short: inverted)

    order_id = order_manager.submit_entry(
        symbol=symbol,
        direction=direction,
        size_usd=pos_info["margin"],
        entry_price=entry_price,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        regime=regime,
        signal_strength=signal_strength,
    )

    if order_id:
        sig_info["action"] = "ENTERED"
        logger.info(
            f"{symbol}: entry submitted — dir={direction} size_usd={pos_info['margin']:.2f} "
            f"atr={atr:.6f} tp_pct={tp_pct:.4%} sl_pct={sl_pct:.4%} "
            f"signal={signal_strength:.3f} orderId={order_id}"
        )

    return sig_info
