import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Load .env before any Binance client is instantiated so API keys are in os.environ
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed — keys must be set in environment manually

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

_SYMBOLS_PATH = Path("config/symbols.yaml")
_TRADE_LOG_PATH = Path("results/live_trade_log.csv")
_BAR_SECONDS = 15 * 60   # 15-minute bar in seconds
_BAR_CLOSE_BUFFER = 5    # seconds after bar close before sampling


def _load_symbol_list() -> list[str]:
    with open(_SYMBOLS_PATH) as f:
        raw = yaml.safe_load(f)
    # Return all symbols; tier-based filtering happens at runtime using model registry
    return list(raw.get("symbols", {}).keys())


def _seconds_until_next_bar() -> float:
    now_ts = time.time()
    # Next 15m boundary is the next multiple of 900s from epoch
    next_boundary = (int(now_ts / _BAR_SECONDS) + 1) * _BAR_SECONDS
    return next_boundary - now_ts


def _get_active_symbols(cfg, state: dict, all_symbols: list[str]) -> list[str]:
    equity = float(state.get("account", {}).get("current_equity", 0.0))
    max_symbols, _ = get_growth_gate_limits(equity, cfg)

    # Prefer symbols with a trained model; order: tier-A first, then tier-B
    model_tiers = state.get("model_tiers", {})

    tier_a = [s for s in all_symbols if model_tiers.get(s, {}).get("tier") == "A"]
    tier_b = [s for s in all_symbols if model_tiers.get(s, {}).get("tier") == "B"]
    ordered = tier_a + [s for s in tier_b if s not in tier_a]

    # Fallback: if no tier info, use symbols in yaml order
    if not ordered:
        ordered = all_symbols

    active = ordered[:max_symbols]
    logger.info(f"Active symbols ({len(active)}/{max_symbols} allowed): {active}")
    return active


def _load_primary_model(symbol: str, cfg):
    entry = get_latest_model(symbol, "15m", model_type="primary")
    if entry is None:
        return None, None
    try:
        model, calibrator = load_model(symbol, "15m", entry["version"], cfg.data.models_dir)
        return model, calibrator
    except Exception as exc:
        logger.warning(f"{symbol}: could not load primary model: {exc}")
        return None, None


def _load_meta(symbol: str, cfg):
    entry = get_latest_model(symbol, "15m", model_type="meta")
    if entry is None:
        return None
    try:
        return load_meta_model(symbol, "15m", entry["version"], cfg.data.models_dir)
    except Exception as exc:
        logger.debug(f"{symbol}: no meta model loaded: {exc}")
        return None


def _predict(model, calibrator, meta_model, feature_series: pd.Series) -> tuple[float, float]:
    X = feature_series.values.reshape(1, -1)

    # Calibrated probability from isotonic calibrator
    raw_prob = float(model.predict_proba(X)[0][1])
    if calibrator is not None:
        primary_prob = float(calibrator.predict_proba(X)[0][1])
    else:
        primary_prob = raw_prob

    if meta_model is not None:
        meta_prob = float(meta_model.predict_proba(X)[0][1])
        signal_strength = primary_prob * meta_prob
    else:
        signal_strength = primary_prob

    return primary_prob, signal_strength


def run(cfg, **kwargs) -> None:
    logger.info("=== Stage 8: Live Execution Loop ===")

    all_symbols = _load_symbol_list()
    client = BinanceClient(cfg)
    order_manager = OrderManager(client, cfg, _TRADE_LOG_PATH)
    lookback_needed = get_lookback_bars_needed(cfg)
    signal_floor = float(cfg.portfolio.signal_floor_prob)
    max_margin_pct = float(cfg.portfolio.max_total_margin_pct)

    # Verify connectivity — will raise if credentials or endpoint are wrong
    try:
        server_ms = client.get_server_time()
        logger.info(f"Connected to Binance — server time: {datetime.fromtimestamp(server_ms/1000, tz=timezone.utc)}")
    except Exception as exc:
        logger.error(f"Cannot connect to Binance: {exc}")
        raise

    logger.info(f"Entering bar-wait loop — signal_floor={signal_floor} lookback={lookback_needed} bars")

    try:
        while True:
            wait = _seconds_until_next_bar() + _BAR_CLOSE_BUFFER
            logger.info(f"Sleeping {wait:.1f}s until next bar + buffer...")
            # Heartbeat before sleep so DMS doesn't fire during the bar-wait period
            order_manager.heartbeat()
            time.sleep(wait)
            order_manager.heartbeat()  # reset again right after waking

            bar_start = datetime.now(timezone.utc)

            state = load_state()
            active_symbols = _get_active_symbols(cfg, state, all_symbols)
            equity = float(state.get("account", {}).get("current_equity", 0.0))

            for symbol in active_symbols:
                try:
                    _process_symbol(
                        symbol=symbol,
                        client=client,
                        order_manager=order_manager,
                        cfg=cfg,
                        lookback_needed=lookback_needed,
                        signal_floor=signal_floor,
                        max_margin_pct=max_margin_pct,
                        equity=equity,
                        state=state,
                    )
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
                logger.warning(f"Could not refresh equity from exchange: {exc}")

            bar_elapsed = (datetime.now(timezone.utc) - bar_start).total_seconds()
            logger.info(f"Bar loop complete in {bar_elapsed:.1f}s — {len(active_symbols)} symbols processed")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received — cancelling all open positions before exit")
        order_manager.cancel_all_open()
        logger.info("Shutdown complete")


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
) -> None:
    # Fetch live OHLCV bars
    klines_df = client.get_klines(symbol, "15m", limit=lookback_needed)
    if len(klines_df) < lookback_needed // 2:
        logger.warning(f"{symbol}: insufficient kline data ({len(klines_df)} bars) — skipping")
        return

    # Compute live features
    feature_series = compute_live_features(symbol, cfg, klines_df)

    # Load models
    primary_model, calibrator = _load_primary_model(symbol, cfg)
    if primary_model is None:
        logger.debug(f"{symbol}: no primary model — skipping")
        return

    meta_model = _load_meta(symbol, cfg)

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

    # Skip if already in a position for this symbol
    if symbol in order_manager.positions:
        return

    # Skip if signal is below floor
    if signal_strength < signal_floor:
        return

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

    # Check portfolio capacity — skip if hard limit would be breached
    current_positions_margin = {
        sym: {"margin": data.get("size_usd", 0) / max(float(leverage), 1)}
        for sym, data in order_manager.positions.items()
    }
    _, should_skip = check_portfolio_capacity(
        current_positions=current_positions_margin,
        new_position={"margin": pos_info["margin"]},
        total_equity=equity,
        cfg=cfg,
    )
    if should_skip:
        logger.info(f"{symbol}: portfolio margin limit reached — skipping entry")
        return

    # Use last close as entry price proxy (market order will fill near this)
    entry_price = float(klines_df["close"].iloc[-1])

    tp_pct = float(cfg.labels.tp_atr_mult) * float(cfg.labels.tp_min_pct)
    sl_pct = float(cfg.labels.sl_atr_mult) * float(cfg.labels.sl_min_pct)

    # Direction: model predicts probability of upward move (label=1 → long)
    direction = "long" if primary_prob >= 0.5 else "short"

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
        logger.info(
            f"{symbol}: entry submitted — dir={direction} size_usd={pos_info['margin']:.2f} "
            f"signal={signal_strength:.3f} orderId={order_id}"
        )
