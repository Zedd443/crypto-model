import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("survivorship")


def load_delisted_coins(path) -> dict:
    path = Path(path)
    if not path.exists():
        logger.warning(f"Delisted coins file not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    # Handle the case where the file contains an empty list
    if isinstance(data, list):
        if len(data) == 0:
            return {}
        # List of dicts with symbol key
        result = {}
        for item in data:
            if isinstance(item, dict) and "symbol" in item:
                result[item["symbol"]] = {
                    "delist_date": item.get("delist_date", ""),
                    "last_price": float(item.get("last_price", 0.0)),
                }
        return result
    elif isinstance(data, dict):
        return data
    return {}


def check_delistings(
    open_positions: dict,
    current_date,
    delisted_dict: dict,
    cfg,
) -> list:
    # Returns list of forced_close events for delisted positions
    forced_closes = []
    if not delisted_dict:
        return forced_closes

    if isinstance(current_date, pd.Timestamp):
        current_dt = current_date
    else:
        current_dt = pd.Timestamp(current_date, tz="UTC")

    slippage_mult = float(cfg.backtest.delisted_slippage_mult)
    normal_slippage = float(cfg.backtest.slippage_pct)

    for symbol in list(open_positions.keys()):
        if symbol not in delisted_dict:
            continue
        delist_info = delisted_dict[symbol]
        delist_date_str = delist_info.get("delist_date", "")
        if not delist_date_str:
            continue
        try:
            delist_ts = pd.Timestamp(delist_date_str, tz="UTC")
        except Exception:
            continue

        if current_dt >= delist_ts:
            last_price = float(delist_info.get("last_price", 0.0))
            pos = open_positions[symbol]
            # Apply doubled slippage for forced exit
            direction = pos.get("direction", 1)
            if direction == 1:
                # Long — forced sell at discount
                exit_price = last_price * (1.0 - normal_slippage * slippage_mult)
            else:
                # Short — forced buy at premium
                exit_price = last_price * (1.0 + normal_slippage * slippage_mult)

            forced_closes.append({
                "symbol": symbol,
                "price": exit_price,
                "reason": "delisted",
                "delist_date": delist_date_str,
            })
            logger.warning(f"Forced close: {symbol} delisted on {delist_date_str}, exit at {exit_price:.4f}")

    return forced_closes


def compute_survivorship_note(delisted_dict: dict, universe_symbols: list) -> str:
    if not delisted_dict:
        return (
            "Survivorship bias present: only surviving coins included. "
            "Estimated Sharpe overstatement: 0.1-0.3 points."
        )
    n_delisted = len(delisted_dict)
    return (
        f"{n_delisted} delisted coins included in backtest with forced-close treatment. "
        "Survivorship bias partially mitigated."
    )
