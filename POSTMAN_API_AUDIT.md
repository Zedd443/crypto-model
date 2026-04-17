# Postman API Audit — Binance FAPI Endpoints

**Tujuan**: Verify semua Binance FAPI endpoints match code expectations. Test pada **DEMO (testnet)** dulu, catat response schema.

**Environment Setup di Postman**:
- Base URL: `https://demo-fapi.binance.com` (DEMO) atau `https://fapi.binance.com` (MAINNET — jangan test di mainnet!)
- API Key: Ambil dari `.env` file (`BINANCE_DEMO_API_KEY`)
- API Secret: Ambil dari `.env` file (`BINANCE_DEMO_API_SECRET`)
- Add to collection pre-request script:
  ```javascript
  // Generate timestamp + signature (HMAC-SHA256)
  const timestamp = Date.now();
  const recvWindow = 5000;
  
  // For signed requests: append to query string sebelum hash
  // Formula: HMAC-SHA256(queryString, API_SECRET).hex()
  ```

---

## 1. Market Data (Unsigned)

### 1.1 GET `/fapi/v1/exchangeInfo` — Symbol filters + limits
**Purpose**: Verify LOT_SIZE (maxQty), MIN_NOTIONAL, PRICE_FILTER (tickSize), PERCENT_PRICE (multiplierUp/Down)

**Code depends on** (from `binance_client.py:160-220`):
- `LOT_SIZE.maxQty` → `get_max_qty()`
- `MIN_NOTIONAL.minNotional` → `get_min_notional()`
- `PRICE_FILTER.tickSize` → `get_tick_size()`
- `PERCENT_PRICE.multiplierUp`, `multiplierDown` → `_price_mult_up_cache`, `_price_mult_down_cache`

**Test with these symbols** (known problematic on testnet):
- `WIFUSDT` (memecoin, low volume)
- `SOLUSDT` (liquid, mid-cap)
- `BTCUSDT` (most liquid)
- `KMNOUSDT`, `XPLUSDT`, `FORMUSDT` (untradeable — maxQty × price < minNotional)

**Expected response structure**:
```json
{
  "symbols": [
    {
      "symbol": "WIFUSDT",
      "filters": [
        {"filterType": "PRICE_FILTER", "minPrice": "...", "maxPrice": "...", "tickSize": "..."},
        {"filterType": "LOT_SIZE", "minQty": "...", "maxQty": "...", "stepSize": "..."},
        {"filterType": "MIN_NOTIONAL", "minNotional": "..."},
        {"filterType": "PERCENT_PRICE", "multiplierUp": "...", "multiplierDown": "..."},
        {"filterType": "MARKET_LOT_SIZE", "maxQty": "..."}
      ]
    }
  ]
}
```

**Verify**:
- [ ] Each symbol has `PRICE_FILTER` with `tickSize` (not 0.0001 if integer-only)
- [ ] `LOT_SIZE.stepSize` matches expected precision (1 for integer, 0.01 for 2 decimals, etc.)
- [ ] `MARKET_LOT_SIZE.maxQty` present (used in `submit_entry()` sizing)
- [ ] `PERCENT_PRICE.multiplierUp/Down` ≈ 1.05 / 0.95 (5% band on DEMO)
- [ ] `MIN_NOTIONAL.minNotional` ≥ 5 (testnet minimum is 5 USDT)

**Catat response** untuk comparison later.

---

### 1.2 GET `/fapi/v1/klines` — Historical OHLCV
**Purpose**: Verify 15m bars fetch correctly for feature engineering

**Code depends on** (from `binance_client.py:108-127`):
- Returns array of bars: `[open_time_ms, open, high, low, close, volume, ...]`
- Used in `get_klines()` → fetches up to 1500 bars (Binance hard limit)

**Test request**:
- Symbol: `BTCUSDT`
- Interval: `15m`
- Limit: `100` (small batch first)

**Expected response**: Array of 100 bars, each with 12 fields minimum.

**Verify**:
- [ ] open_time is milliseconds (e.g., 1702500900000)
- [ ] close price is realistic for BTCUSDT (e.g., 42000-45000 range in 2026-04)
- [ ] volume > 0
- [ ] Bars are in ascending order (oldest to newest)

---

### 1.3 GET `/fapi/v1/premiumIndex` — Funding rates
**Purpose**: Get current funding rate + mark price (used in live features)

**Code depends on** (from `binance_client.py:146-155`):
- `data["lastFundingRate"]` → funding rate (e.g., 0.0001 = 0.01%)
- `data["markPrice"]` → mark price for portfolio calculations

**Test request**:
- Symbol: `SOLUSDT`

**Expected response**:
```json
{
  "symbol": "SOLUSDT",
  "markPrice": "135.45",
  "indexPrice": "135.40",
  "estimatedSettlePrice": "135.50",
  "lastFundingRate": "0.0001",
  "interestRate": "0.0001",
  "nextFundingTime": 1714684800000,
  "time": 1714680378952
}
```

**Verify**:
- [ ] `lastFundingRate` is numeric (not string)
- [ ] `markPrice` is realistic (matches order book mid)

---

## 2. Account & Position Data (Signed requests)

### 2.1 GET `/fapi/v2/account` — Wallet balance + settings
**Purpose**: Get total wallet balance, margin ratios, positions list

**Code depends on** (from `stage_08_live.py:380-385`):
- `account["totalWalletBalance"]` → current equity for position sizing
- `account["positions"]` → list of open positions (fallback if API sync fails)

**Test request**:
- No parameters (signed)

**Expected response**:
```json
{
  "feeTier": 2,
  "canTrade": true,
  "canDeposit": true,
  "canWithdraw": true,
  "updateTime": 1714680378952,
  "totalInitialMargin": "100.50",
  "totalMaintMargin": "50.25",
  "totalWalletBalance": "5000.00",
  "totalUnrealizedProfit": "100.00",
  "totalMarginBalance": "5100.00",
  "totalPositionInitialMargin": "100.50",
  "totalOpenOrderInitialMargin": "0.00",
  "totalCrossWalletBalance": "5000.00",
  "totalCrossUnrealizedProfit": "100.00",
  "availableBalance": "4899.50",
  "maxWithdrawAmount": "4899.50",
  "positions": [
    {
      "symbol": "SOLUSDT",
      "initialMargin": "50.25",
      "maintMargin": "25.12",
      "unrealizedProfit": "10.00",
      "positionInitialMargin": "50.25",
      "positionMaintMargin": "25.12",
      "openOrderInitialMargin": "0.00",
      "maxNotional": 500000,
      "leverage": 2,
      "marginType": "crossed",
      "isAutoAddMargin": false,
      "positionSide": "BOTH",
      "notional": 2500,
      "isolatedCreatedAt": 0,
      "isolatedLiquidatePrice": "0.00000000",
      "isolatedLiquidateQty": "0.00000000",
      "adlQuantile": null
    }
  ]
}
```

**Verify**:
- [ ] `totalWalletBalance` > 0 (actual account balance, not paper)
- [ ] `availableBalance` = `totalWalletBalance` - `totalInitialMargin` (free margin)
- [ ] Field names match code expectations (camelCase, not snake_case)
- [ ] If open positions: check `positionAmt`, `entryPrice`, `unrealizedProfit` are present

---

### 2.2 GET `/fapi/v2/positionRisk` — Per-symbol position detail
**Purpose**: Get current position size, entry price, liquidation price for a specific symbol

**Code depends on** (from `binance_client.py:132-144`):
- Returns **list** (not single object), one per symbol
- Extract fields: `positionAmt`, `entryPrice`, `unRealizedProfit`, `markPrice`

**Test request**:
- Symbol: `SOLUSDT` (or any open position)
- Signed

**Expected response** (array of position objects):
```json
[
  {
    "symbol": "SOLUSDT",
    "positionAmt": "10.0",
    "entryPrice": "130.00",
    "breakEvenPrice": "130.10",
    "markPrice": "135.00",
    "unRealizedProfit": "50.00",
    "liquidationPrice": "100.00",
    "isolatedCreatedAt": 0,
    "isolatedLiquidatePrice": "0.00000000",
    "isolatedLiquidateQty": "0.00000000",
    "adlQuantile": null,
    "marginType": "crossed",
    "isolatedMargin": "0.00000000",
    "isAutoAddMargin": false,
    "positionSide": "BOTH",
    "notional": 1350.00,
    "isolatedWallet": "0",
    "leverage": 2,
    "maxNotionalValue": 500000,
    "maxQty": 250,
    "percentage": 100
  }
]
```

**Verify**:
- [ ] Response is **array** (code does `isinstance(data, list)`)
- [ ] `positionAmt` field present (code uses this, not `amount`)
- [ ] `markPrice` present (used in PnL calc)
- [ ] If position exists: `positionAmt` != 0

---

### 2.3 GET `/fapi/v2/openOrders` — Active orders
**Purpose**: Check if any pending limit/stop orders exist

**Code depends on** (from `order_manager.py:400+`):
- Array of order objects with: `orderId`, `symbol`, `status`, `type`, `side`, etc.

**Test request**:
- No symbol (gets ALL open orders) or Symbol: `SOLUSDT`
- Signed

**Expected response**:
```json
[
  {
    "symbol": "SOLUSDT",
    "orderId": 12345678,
    "orderListId": -1,
    "clientOrderId": "web_1234567890",
    "price": "140.00",
    "origQty": "10.0",
    "executedQty": "0.0",
    "cummulativeQuoteAssetTransactionQty": "0.00",
    "status": "NEW",
    "timeInForce": "GTC",
    "type": "TAKE_PROFIT",
    "side": "SELL",
    "stopPrice": "140.00",
    "time": 1714680378952,
    "updateTime": 1714680378952,
    "isWorking": true,
    "origQuoteOrderQty": "1400.00"
  }
]
```

**Verify**:
- [ ] Response is **array** (empty if no orders)
- [ ] `status` field present (NEW, FILLED, CANCELED, etc.)
- [ ] `orderId` is unique identifier for cancel/update

---

## 3. Order Management (Signed, POST/DELETE)

### 3.1 POST `/fapi/v1/order` — Place market entry order
**Purpose**: Test market entry + TP/SL bracket placement (or fallback to LIMIT on DEMO)

**Code depends on** (from `binance_client.py:250+`, `order_manager.py:172`):
- Request: `symbol`, `side` (BUY/SELL), `quantity`, `type` (MARKET), etc.
- Response: `orderId`, `avgPrice`, `executedQty`, `status`

**Test request (MARKET entry)**:
```
POST /fapi/v1/order
side: BUY
symbol: SOLUSDT
type: MARKET
quantity: 1.0
timeInForce: GTC
recvWindow: 5000
timestamp: <signed>
signature: <HMAC-SHA256>
```

**Expected response**:
```json
{
  "orderId": 12345678,
  "symbol": "SOLUSDT",
  "status": "FILLED",
  "clientOrderId": "my_order_1",
  "price": "0.00000000",
  "avgPrice": "135.50",
  "origQty": "1.0",
  "executedQty": "1.0",
  "cummulativeQuoteAssetTransactionQty": "135.50",
  "timeInForce": "GTC",
  "type": "MARKET",
  "side": "BUY",
  "time": 1714680378952,
  "updateTime": 1714680378952,
  "fills": [
    {"price": "135.50", "qty": "1.0", "commission": "0.01355", "commissionAsset": "USDT", "tradeId": 12345}
  ]
}
```

**Verify**:
- [ ] `status: FILLED` (market orders execute immediately on testnet)
- [ ] `avgPrice` filled (price paid, used in code)
- [ ] `executedQty` matches requested quantity
- [ ] No error (code checks `resp.status_code >= 400`)

---

### 3.2 POST `/fapi/v1/order` — TP/SL bracket orders (DEMO LIMITATION)
**Purpose**: Test take-profit + stop-loss bracket placement

**⚠️ CRITICAL**: Binance testnet **does NOT support** `STOP_MARKET` / `TAKE_PROFIT_MARKET`. 
- Code uses **LIMIT fallback on DEMO** (see `binance_client.py:place_order()`, lines 50-70)
- MAINNET uses real `STOP_MARKET + closePosition=true`

**Test request (TP LIMIT fallback on DEMO)**:
```
POST /fapi/v1/order
side: SELL
symbol: SOLUSDT
type: LIMIT
price: 140.00
stopPrice: 140.00
quantity: 1.0
timeInForce: GTC
reduceOnly: true
recvWindow: 5000
timestamp: <signed>
signature: <HMAC-SHA256>
```

**Expected behavior on DEMO**:
- ✅ Order placed as LIMIT at `price=140.00`
- ❌ `STOP_MARKET` would fail with error `-4120: "use Algo Order API"` (this is why fallback exists)

**Verify**:
- [ ] LIMIT TP order succeeds
- [ ] Check `status: NEW` (limit waiting to fill)
- [ ] Price within PERCENT_PRICE ±5% band (demo enforces this)

---

### 3.3 DELETE `/fapi/v1/order` — Cancel order
**Purpose**: Test cancellation for stale limit orders

**Code depends on** (from `order_manager.py:470+`):
- Request: `symbol`, `orderId`
- Response: cancelled order details

**Test request**:
```
DELETE /fapi/v1/order
symbol: SOLUSDT
orderId: <from previous test>
timestamp: <signed>
signature: <HMAC-SHA256>
```

**Expected response**:
```json
{
  "symbol": "SOLUSDT",
  "orderId": 12345678,
  "status": "CANCELED",
  "origQty": "1.0",
  "executedQty": "0.0",
  "price": "140.00"
}
```

**Verify**:
- [ ] `status: CANCELED` (order removed)
- [ ] Order no longer appears in `/fapi/v2/openOrders`

---

### 3.4 POST `/fapi/v1/allOpenOrders` — Cancel all orders
**Purpose**: Emergency shutdown (e.g., dead-man-switch)

**Test request**:
```
DELETE /fapi/v1/allOpenOrders
symbol: SOLUSDT
timestamp: <signed>
signature: <HMAC-SHA256>
```

**Expected response**: Array of cancelled orders.

**Verify**:
- [ ] All orders for symbol cancelled
- [ ] Subsequent `/fapi/v2/openOrders` returns empty

---

## 4. Leverage & Margin Settings (Signed, POST)

### 4.1 POST `/fapi/v1/leverage` — Set symbol leverage
**Purpose**: Verify leverage bracket system (used in `get_notional_limit()`)

**Code depends on** (from `binance_client.py:235+`):
- Request: `symbol`, `leverage` (1-125, depends on bracket)
- Response: `leverage`, `maxNotionalValue` (notional cap for this bracket)

**Test request**:
```
POST /fapi/v1/leverage
symbol: SOLUSDT
leverage: 2
timestamp: <signed>
signature: <HMAC-SHA256>
```

**Expected response**:
```json
{
  "symbol": "SOLUSDT",
  "leverage": 2,
  "maxNotionalValue": "500000.00"
}
```

**Verify**:
- [ ] Leverage accepted (2× is normal)
- [ ] `maxNotionalValue` returned (e.g., 500k USDT = max position notional at 2× leverage)
- [ ] Attempting leverage > bracket max (e.g., 50×) returns error `-4046: "Leverage above the maximum allowed"`

---

### 4.2 GET `/fapi/v1/leverageBracket` — Query leverage brackets
**Purpose**: Verify tier structure (for future tier-based position sizing)

**Code depends on**: Currently not read, but useful for understanding leverage tiers.

**Test request**:
```
GET /fapi/v1/leverageBracket
symbol: BTCUSDT
timestamp: <signed>
signature: <HMAC-SHA256>
```

**Expected response**:
```json
[
  {
    "symbol": "BTCUSDT",
    "brackets": [
      {"bracket": 1, "initialLeverage": 125, "notionalCap": 10000, "notionalFloor": 0, "maintMarginRatio": 0.004, "cum": 0},
      {"bracket": 2, "initialLeverage": 100, "notionalCap": 50000, "notionalFloor": 10000, "maintMarginRatio": 0.005, "cum": 50},
      ...
    ]
  }
]
```

**Verify**:
- [ ] Brackets exist (at least 3-4 tiers)
- [ ] `notionalCap` decreases as leverage increases (inverse relationship)
- [ ] `maintMarginRatio` provides liquidation distance

---

## 5. Market Data Streams (Future: WebSocket)
**Status**: Currently not tested (code uses REST polling in `stage_08_live.py`).

These are for future optimization if live latency becomes issue:
- `/ws/...@kline_15m` — stream 15m bars in real-time
- `/ws/...@markPrice@1s` — stream mark price every 1s
- Fallback: current REST polling is robust

---

## Test Checklist — Copy to Postman

Run in this order:

### Phase 1: Market Data (No Auth)
- [ ] `GET /fapi/v1/exchangeInfo` → Catat `maxQty`, `tickSize`, `minNotional`, `PERCENT_PRICE` bounds
- [ ] `GET /fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=100` → Verify OHLCV structure
- [ ] `GET /fapi/v1/premiumIndex?symbol=SOLUSDT` → Check funding rate format

### Phase 2: Account (Signed)
- [ ] `GET /fapi/v2/account` → Record `totalWalletBalance`, available margin
- [ ] `GET /fapi/v2/positionRisk?symbol=SOLUSDT` → Verify position (if any)
- [ ] `GET /fapi/v2/openOrders` → Check if any stale orders exist

### Phase 3: Order Management (Signed)
- [ ] `POST /fapi/v1/order` (MARKET BUY SOLUSDT qty=1) → Should FILL immediately
- [ ] Check `/fapi/v2/positionRisk` again → Should show new position
- [ ] `POST /fapi/v1/order` (LIMIT SELL as TP, price=150, stopPrice=150, reduceOnly=true) → Should place
- [ ] `GET /fapi/v2/openOrders` → Should show TP order NEW
- [ ] `DELETE /fapi/v1/order?orderId=<from previous>` → Cancel TP order
- [ ] `DELETE /fapi/v1/allOpenOrders?symbol=SOLUSDT` → Cancel all remaining

### Phase 4: Leverage (Signed)
- [ ] `POST /fapi/v1/leverage?symbol=SOLUSDT&leverage=2` → Set 2× leverage
- [ ] `GET /fapi/v1/leverageBracket?symbol=BTCUSDT` → View bracket tiers

---

## Catat Findings

**After running tests, catat**:
1. **Actual response schemas** — compare with code expectations
2. **Error codes** encountered (if any) + solutions
3. **Demo vs Mainnet differences** — what works/fails on testnet?
4. **API limits hit** — rate limiting behavior, retry-after headers
5. **Latency** — response times for each endpoint (important for live loop timing)

**Report format**:
```
## DEMO TESTNET RESULTS (2026-04-17)

✅ Passing:
- GET /fapi/v1/exchangeInfo: 8 symbols audited, all filters present
- GET /fapi/v1/klines: 1500-bar limit confirmed, 15m interval works
- POST /fapi/v1/order (MARKET): Fills immediately, avgPrice present
- GET /fapi/v2/account: Returns totalWalletBalance correctly

⚠️ Warnings:
- GET /fapi/v1/leverageBracket: Response 200ms (slow, cache locally)
- STOP_MARKET not supported on DEMO (expected, using LIMIT fallback)

❌ Failures:
- (none so far if using fallback)

🔧 Adjustments needed:
- (if any discrepancies vs code)
```

---

**Ready to test?** Copy the test checklist into Postman collection, run through Phase 1-4, and report findings.
