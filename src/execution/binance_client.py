import hashlib
import hmac
import os
import time
from urllib.parse import urlencode

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger("binance_client")


class BinanceClient:
    def __init__(self, cfg):
        mode = cfg.trading.mode  # "DEMO" or "MAINNET"
        self._base_url = cfg.trading.endpoints[mode]
        self._backoff_base = float(cfg.trading.rate_limit_backoff_base)
        self._backoff_max = float(cfg.trading.rate_limit_backoff_max)

        # Select keys based on mode
        # DEMO    → BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET (testnet.binancefuture.com — paper trading)
        # MAINNET → BINANCE_API_KEY / BINANCE_API_SECRET (fapi.binance.com — real money)
        if mode == "DEMO":
            self._api_key = os.environ.get("BINANCE_DEMO_API_KEY", "")
            self._api_secret = os.environ.get("BINANCE_DEMO_API_SECRET", "")
        else:  # MAINNET
            self._api_key = os.environ.get("BINANCE_API_KEY", "")
            self._api_secret = os.environ.get("BINANCE_API_SECRET", "")

        if not self._api_key:
            logger.warning(f"No API key found for mode={mode} — check .env (BINANCE_{mode}_API_KEY)")

        self._session = requests.Session()
        # Only set the API key header — do NOT set Content-Type here.
        # POST endpoints require application/x-www-form-urlencoded; requests sets
        # the correct Content-Type automatically based on whether data= or params= is used.
        self._session.headers.update({
            "X-MBX-APIKEY": self._api_key,
        })
        self._mode = mode.upper()
        # Cache for symbol exchange info (step size, max qty, min notional, tick size)
        self._qty_step_cache: dict[str, float] = {}
        self._qty_max_cache: dict[str, float] = {}      # max order qty from MARKET_LOT_SIZE / LOT_SIZE
        self._min_notional_cache: dict[str, float] = {}  # min notional from MIN_NOTIONAL filter
        self._tick_size_cache: dict[str, float] = {}     # price tick size from PRICE_FILTER
        self._price_mult_up_cache: dict[str, float] = {}    # PERCENT_PRICE multiplierUp (e.g. 1.05)
        self._price_mult_down_cache: dict[str, float] = {}  # PERCENT_PRICE multiplierDown (e.g. 0.95)
        logger.info(f"BinanceClient initialised — mode={mode} endpoint={self._base_url}")

    def _sign(self, params: dict) -> dict:
        # Append timestamp + recvWindow, then HMAC-SHA256 the query string
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        sig = hmac.new(
            self._api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = sig
        return params

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False) -> dict | list:
        params = params or {}
        if signed:
            params = self._sign(params)

        url = self._base_url + path
        delay = self._backoff_base
        attempt = 0

        while True:
            try:
                resp = self._session.request(method, url, params=params if method == "GET" else None,
                                             data=params if method != "GET" else None)
            except requests.RequestException as exc:
                logger.error(f"Request error {path}: {exc}")
                raise

            if resp.status_code in (429, 418):
                # Rate-limited — back off exponentially
                retry_after = float(resp.headers.get("Retry-After", delay))
                wait = min(max(delay, retry_after), self._backoff_max)
                logger.warning(f"Rate limited ({resp.status_code}) on {path} — sleeping {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                delay = min(delay * 2, self._backoff_max)
                attempt += 1
                continue

            if resp.status_code >= 400:
                logger.error(f"HTTP {resp.status_code} on {path}: {resp.text}")
                resp.raise_for_status()

            return resp.json()

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        raw = self._request("GET", "/fapi/v1/klines", params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        })
        # Each element: [open_time, open, high, low, close, volume, close_time, ...]
        rows = []
        for k in raw:
            rows.append({
                "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(rows).set_index("open_time")
        df.index.name = "timestamp"
        return df

    def get_account(self) -> dict:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_position(self, symbol: str) -> dict:
        data = self._request("GET", "/fapi/v2/positionRisk", params={"symbol": symbol}, signed=True)
        # Returns list; grab first match
        if isinstance(data, list):
            for p in data:
                if p.get("symbol") == symbol:
                    return {
                        "positionAmt": float(p.get("positionAmt", 0)),
                        "entryPrice": float(p.get("entryPrice", 0)),
                        "unrealizedProfit": float(p.get("unRealizedProfit", 0)),
                        "markPrice": float(p.get("markPrice", 0)),
                    }
        return {"positionAmt": 0.0, "entryPrice": 0.0, "unrealizedProfit": 0.0, "markPrice": 0.0}

    def get_funding_rate(self, symbol: str) -> float:
        # Returns latest real funding rate from Binance — used in live features to replace proxy
        try:
            data = self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})
            if isinstance(data, dict):
                return float(data.get("lastFundingRate", 0.0))
            if isinstance(data, list) and len(data) > 0:
                return float(data[0].get("lastFundingRate", 0.0))
        except Exception as exc:
            logger.debug(f"{symbol}: get_funding_rate failed: {exc}")
        return 0.0

    def get_funding_rate_history(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        # Returns historical funding rates as DataFrame with columns [timestamp, fundingRate]
        try:
            data = self._request("GET", "/fapi/v1/fundingRate", params={"symbol": symbol, "limit": limit})
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
                df["fundingRate"] = df["fundingRate"].astype(float)
                return df[["timestamp", "fundingRate"]].set_index("timestamp").sort_index()
        except Exception as exc:
            logger.debug(f"{symbol}: get_funding_rate_history failed: {exc}")
        return pd.DataFrame(columns=["fundingRate"])

    def get_commission_rate(self, symbol: str) -> dict:
        # Returns taker/maker commission rates for symbol — used to deduct actual fee from pnl
        try:
            data = self._request("GET", "/fapi/v1/commissionRate", params={"symbol": symbol}, signed=True)
            if isinstance(data, dict):
                return {
                    "taker": float(data.get("takerCommissionRate", 0.0004)),
                    "maker": float(data.get("makerCommissionRate", 0.0002)),
                }
        except Exception as exc:
            logger.debug(f"{symbol}: get_commission_rate failed: {exc}")
        return {"taker": 0.0004, "maker": 0.0002}  # Binance FAPI default fallback

    def get_all_open_positions(self) -> list[dict]:
        # Returns all positions with non-zero positionAmt from exchange — ground truth for reconciliation
        try:
            data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
            if isinstance(data, list):
                return [
                    {
                        "symbol": p["symbol"],
                        "positionAmt": float(p.get("positionAmt", 0)),
                        "entryPrice": float(p.get("entryPrice", 0)),
                        "unrealizedProfit": float(p.get("unRealizedProfit", 0)),
                        "markPrice": float(p.get("markPrice", 0)),
                    }
                    for p in data
                    if abs(float(p.get("positionAmt", 0))) > 0
                ]
        except Exception as exc:
            logger.debug(f"get_all_open_positions failed: {exc}")
        return []

    def get_open_orders(self, symbol: str) -> list:
        return self._request("GET", "/fapi/v1/openOrders", params={"symbol": symbol}, signed=True)

    # Conditional trigger order types
    _CONDITIONAL_ORDER_TYPES = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"}

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MARKET",
        price: float | None = None,
        stop_price: float | None = None,
        reduce_only: bool = False,
        close_position: bool = False,
        working_type: str | None = None,
    ) -> dict:
        # DEMO mode: demo-fapi.binance.com does NOT support STOP_MARKET / TAKE_PROFIT_MARKET
        # on either /fapi/v1/order (-4120) or /fapi/v1/algoOrder (only TWAP/VP algoTypes valid).
        # Fall back to LIMIT orders at the stop_price level (fills at that price or better).
        # Do NOT use reduceOnly — two simultaneous bracket LIMIT orders would both be reduceOnly
        # on the same qty, which triggers -2022 ReduceOnly rejected on the second order.
        # Without reduceOnly they coexist; whichever fills first closes the position, the
        # other becomes a small directional order that sync_fills + cancel_all_orders cleans up.
        if self._mode == "DEMO" and order_type in self._CONDITIONAL_ORDER_TYPES:
            limit_price = price if price is not None else stop_price
            if limit_price is None:
                raise ValueError(f"place_order DEMO fallback: stop_price required for {order_type}")
            params: dict = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "price": limit_price,
                "quantity": qty,
                "timeInForce": "GTC",
            }
            logger.debug(f"place_order DEMO {symbol} {side} {order_type}→LIMIT price={limit_price} qty={qty}")
            return self._request("POST", "/fapi/v1/order", params=params, signed=True)

        # MAINNET: conditional orders use /fapi/v1/order with stopPrice + closePosition=true.
        if order_type in self._CONDITIONAL_ORDER_TYPES:
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
            }
            if stop_price is not None:
                params["stopPrice"] = stop_price
            if price is not None:
                params["price"] = price
            if working_type is not None:
                params["workingType"] = working_type
            if close_position:
                params["closePosition"] = "true"
            else:
                params["quantity"] = qty
                if reduce_only:
                    params["reduceOnly"] = "true"
            logger.debug(f"place_order {symbol} {side} {order_type} stopPrice={stop_price}")
            return self._request("POST", "/fapi/v1/order", params=params, signed=True)

        # Regular order (LIMIT / MARKET)
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
        }
        if close_position:
            params["closePosition"] = "true"
        else:
            params["quantity"] = qty
            if reduce_only:
                params["reduceOnly"] = "true"
        if price is not None:
            params["price"] = price
            params["timeInForce"] = "GTC"
        logger.debug(f"place_order {symbol} {side} {order_type} qty={qty} price={price}")
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int, is_algo: bool = False) -> dict:
        if is_algo:
            return self._request("DELETE", "/fapi/v1/algoOrder", params={
                "symbol": symbol,
                "algoId": order_id,
            }, signed=True)
        return self._request("DELETE", "/fapi/v1/order", params={
            "symbol": symbol,
            "orderId": order_id,
        }, signed=True)

    def cancel_all_orders(self, symbol: str) -> None:
        # Cancel regular open orders
        try:
            self._request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol}, signed=True)
        except Exception as exc:
            logger.debug(f"{symbol}: cancel regular orders — {exc}")
        # Cancel open algo orders
        try:
            algo_orders = self._request("GET", "/fapi/v1/openAlgoOrders", params={"symbol": symbol}, signed=True)
            if isinstance(algo_orders, dict):
                algo_orders = algo_orders.get("orders", [])
            for o in algo_orders or []:
                algo_id = o.get("algoId")
                if algo_id:
                    self._request("DELETE", "/fapi/v1/algoOrder", params={
                        "symbol": symbol, "algoId": algo_id,
                    }, signed=True)
        except Exception as exc:
            logger.debug(f"{symbol}: cancel algo orders — {exc}")

    def get_recent_trades(self, symbol: str, limit: int = 10) -> list:
        # GET /fapi/v1/userTrades — signed, returns recent fills for this account/symbol
        return self._request("GET", "/fapi/v1/userTrades", params={"symbol": symbol, "limit": limit}, signed=True)

    def get_server_time(self) -> int:
        data = self._request("GET", "/fapi/v1/time")
        return int(data["serverTime"])

    def _fetch_symbol_info(self, symbol: str) -> None:
        """Fetch and cache all relevant exchange filters for a symbol (one API call)."""
        try:
            data = self._request("GET", "/fapi/v1/exchangeInfo", {"symbol": symbol})
            if "symbols" not in data or len(data["symbols"]) == 0:
                raise ValueError("empty symbols list")
            sym_info = data["symbols"][0]
            for filt in sym_info.get("filters", []):
                ft = filt.get("filterType")
                if ft == "LOT_SIZE":
                    self._qty_step_cache[symbol] = float(filt.get("stepSize", 1.0))
                    max_qty = float(filt.get("maxQty", 0))
                    if max_qty > 0:
                        self._qty_max_cache[symbol] = max_qty
                elif ft == "MARKET_LOT_SIZE":
                    # DEMO (testnet): MARKET_LOT_SIZE.maxQty=120 is a real limit for market orders.
                    # MAINNET: this filter does not reflect actual position capacity (leverage bracket
                    # notional limits apply instead). Only use for DEMO.
                    max_qty = float(filt.get("maxQty", 0))
                    if max_qty > 0 and self._mode == "DEMO":
                        self._qty_max_cache[symbol] = max_qty
                elif ft == "MIN_NOTIONAL":
                    min_notional = float(filt.get("notional", filt.get("minNotional", 5.0)))
                    self._min_notional_cache[symbol] = min_notional
                elif ft == "PRICE_FILTER":
                    tick = float(filt.get("tickSize", 0.01))
                    if tick > 0:
                        self._tick_size_cache[symbol] = tick
                elif ft == "PERCENT_PRICE":
                    mult_up = float(filt.get("multiplierUp", 1.05))
                    mult_down = float(filt.get("multiplierDown", 0.95))
                    self._price_mult_up_cache[symbol] = mult_up
                    self._price_mult_down_cache[symbol] = mult_down
            logger.debug(
                f"{symbol}: step={self._qty_step_cache.get(symbol)} "
                f"max_qty={self._qty_max_cache.get(symbol)} "
                f"min_notional={self._min_notional_cache.get(symbol)} "
                f"tick={self._tick_size_cache.get(symbol)} "
                f"pct_price=[{self._price_mult_down_cache.get(symbol)},{self._price_mult_up_cache.get(symbol)}]"
            )
        except Exception as e:
            logger.warning(f"{symbol}: could not fetch exchange info — {e}")
            # Fallbacks
            self._qty_step_cache.setdefault(symbol, 1.0)

    def _ensure_symbol_info(self, symbol: str) -> None:
        """Fetch and cache exchange info for symbol if not already cached."""
        if symbol not in self._qty_step_cache:
            self._fetch_symbol_info(symbol)

    def get_qty_step(self, symbol: str) -> float:
        """Minimum quantity step size for a symbol (lot size precision)."""
        self._ensure_symbol_info(symbol)
        return self._qty_step_cache.get(symbol, 1.0)

    def get_max_qty(self, symbol: str, price: float) -> float:
        """Maximum order quantity for a symbol. Returns inf if no limit found."""
        self._ensure_symbol_info(symbol)
        return self._qty_max_cache.get(symbol, float("inf"))

    def get_notional_limit(self, symbol: str, leverage: int) -> float:
        """Max position notional (USD) for a symbol at the given leverage, from leverageBracket.
        Returns a large value (1e9) if the endpoint fails — caller should not block on this."""
        try:
            data = self._request("GET", "/fapi/v1/leverageBracket", params={"symbol": symbol}, signed=True)
            # Response: [{"symbol": ..., "brackets": [{"bracket":1, "initialLeverage":125, "notionalCap":10000, ...}]}]
            brackets = []
            if isinstance(data, list) and len(data) > 0:
                brackets = data[0].get("brackets", [])
            elif isinstance(data, dict):
                brackets = data.get("brackets", [])
            # Find the bracket where initialLeverage >= requested leverage — smallest notionalCap that applies
            applicable = [b for b in brackets if int(b.get("initialLeverage", 0)) >= leverage]
            if applicable:
                # Sort by initialLeverage ascending — pick the tightest bracket that allows our leverage
                applicable.sort(key=lambda b: int(b.get("initialLeverage", 0)))
                cap = float(applicable[0].get("notionalCap", 1e9))
                logger.debug(f"{symbol}: notional cap at leverage={leverage}x → {cap:,.0f} USDT")
                return cap
        except Exception as exc:
            logger.debug(f"{symbol}: could not fetch leverageBracket — {exc}")
        return 1e9  # no cap found — don't restrict

    def get_min_notional(self, symbol: str) -> float:
        """Minimum order notional (USD value) for a symbol. Default 5.0."""
        self._ensure_symbol_info(symbol)
        return self._min_notional_cache.get(symbol, 5.0)

    def get_tick_size(self, symbol: str) -> float:
        """Price tick size for a symbol. Default 0.01."""
        self._ensure_symbol_info(symbol)
        return self._tick_size_cache.get(symbol, 0.01)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to nearest tick size for the symbol."""
        tick = self.get_tick_size(symbol)
        return round(round(price / tick) * tick, 10)

    def clamp_bracket_price(self, symbol: str, price: float, ref_price: float) -> float:
        """Clamp a bracket order price to within the PERCENT_PRICE band around ref_price.
        demo-fapi enforces this strictly; clamped prices keep orders from being rejected.
        """
        self._ensure_symbol_info(symbol)
        mult_up = self._price_mult_up_cache.get(symbol, 1.05)
        mult_down = self._price_mult_down_cache.get(symbol, 0.95)
        max_price = ref_price * mult_up
        min_price = ref_price * mult_down
        tick = self.get_tick_size(symbol)
        clamped = max(min_price, min(max_price, price))
        return round(round(clamped / tick) * tick, 10)
