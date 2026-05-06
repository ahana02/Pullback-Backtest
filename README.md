# Pullback-Backtest

A rule-based structural bullish pullback trading strategy, implemented and backtested on 15-minute OHLC market data using Python.

The project focuses on **clear thinking and transparent implementation** — no indicator stacking, no curve fitting. One regime filter, one entry signal, two exit rules.

---

## Results

| Metric | Value |
|---|---|
| Total Return | +33.39% |
| Annualised Return | +10.21% |
| Sharpe Ratio | 1.992 |
| Max Drawdown | −4.22% |
| Total Trades | 671 |
| Win Rate | 82.0% |
| Profit Factor | 1.535 |
| Avg Trade Duration | 18 bars (~4.5 hrs) |

> Backtested on `ohlc_data.csv` — 28,418 bars of 15-min OHLC data spanning **Jan 2021 → Dec 2023 (~3 years)**.

---

## Repository Structure

```
structural-pullback-backtest/
│
├── data_set/
│   └── ohlc_data.csv              ← 15-min OHLC dataset (28,418 bars)
│
├── backtester/
│   └── backtester.ipynb           ← Jupyter notebook (EDA + backtest + charts)
│
├── trade_log/
│   └── trade_log.csv              ← Auto-generated log of all 671 trades
│
├── backtester.py                  ← Standalone Python script
└── README.md
```

---

## Dataset

| Field | Detail |
|---|---|
| Format | CSV — `timestamp, open, high, low, close` |
| Frequency | 15-minute bars |
| Period | 2021-01-03 → 2023-12-31 |
| Rows | 28,418 |
| Price range | 94.36 – 108.46 (anonymised) |
| Ann. volatility | ~5.2% |

The dataset is anonymised. It spans multiple market regimes — bullish trending periods interspersed with sideways and corrective phases — which allows the strategy to be stress-tested across varied conditions.

---

## Strategy

### Core Hypothesis

> During a bullish structural regime, short-term pullbacks to local support often create favourable entry opportunities before the broader upward structure resumes.

The strategy does **not** trade all market conditions. It activates only when the market is in a defined bullish regime, and enters only when a confirmed pullback signal appears within that regime.

---

### 1. Regime Filter

<!-- ```python
df["regime_ma"]         = df["close"].rolling(200).mean()
df["is_bullish_regime"] = df["close"] > df["regime_ma"]
``` -->

- Price above the **200-bar moving average** → bullish regime active
- No trades are taken when price is below the MA
- The MA acts as a slow-moving structural anchor, not a signal line

---

### 2. Entry — Confirmed Local Low

<!-- ```python
def is_local_low(df, i):
    at_low  = df["low"].iloc[i] <= df["rolling_low"].iloc[i] * 1.002
    upward  = df["close"].iloc[i + CONFIRM_BARS] > df["close"].iloc[i]
    return at_low and upward
``` -->

A trade is entered when **both** conditions are true simultaneously:

1. The current bar's low touches the **10-bar rolling minimum** (within 0.2% tolerance) — confirming a short-term pullback has occurred
2. Price **closes higher** within the next 3 bars — confirming the pullback is reversing upward

Entry is taken at the **close of the 3rd confirmation bar**, not at the signal bar itself. This avoids entries into bars that are still falling.

---

### 3. Exit 1 — Support-Break Stop-Loss

<!-- ```python
stop_price = recent_support * (1 - 0.001)   # 0.1% below 20-bar support
stop_hit   = price < active_trade.stop_price
``` -->

- At entry, the **20-bar rolling minimum of lows** is recorded as the support level
- Stop is placed **0.1% below** that support (a small buffer to avoid noise-triggered exits)
- If price closes below the stop, **the pullback thesis has failed** → exit immediately to cut the loss
- Logic: a pullback strategy assumes support holds. If it structurally breaks, the trade idea is invalidated.

---

### 4. Exit 2 — Momentum-Fade Take-Profit

<!-- ```python
momentum      = df["close"].diff(5)          # 5-bar price change
momentum_fade = (price > entry_price) and (momentum < 0)
``` -->

- `momentum` is the simple **5-bar price change** (close[i] − close[i−5])
- Exit triggers when the trade is **profitable** (price above entry) **but** the last 5 bars show a negative price change
- Logic: don't wait for a full reversal. When a winning trade starts losing steam, lock in the gain before it erodes back to breakeven.

This exit covers the opposite failure mode from the stop-loss — protecting against giving back a gain already earned, rather than cutting a loss.

---

### Execution Assumptions

| Assumption | Value |
|---|---|
| Active trades at once | 1 (no pyramiding) |
| Execution price | Close-to-close |
| Leverage | None (1×) |
| Slippage | Not modelled |
| Transaction costs | Not modelled |

---

## Implementation

### Setup

```bash
# Clone the repo
git clone https://github.com/ahana02/pullback-backtest.git
cd pullback-backtest

# Install dependencies
pip install pandas numpy matplotlib
```

### Run

```bash
python backtester.py
```

The script will:
1. Load `data_set/ohlc_data.csv`
2. Compute indicators and run the backtest bar-by-bar
3. Print the full metrics table to console
4. Save the trade log to `trade_log/trade_log.csv`
5. Save a chart to `backtest_results.png` and display it

<!-- > If `ohlc_data.csv` is not found, the script automatically generates synthetic data as a fallback. -->

### Notebook

Open `backtester/backtester.ipynb` in Jupyter for the full walkthrough including EDA, indicator visualisation, and annotated backtest logic.

```bash
jupyter notebook backtester/backtester.ipynb
```

---

## Configuration

All parameters are in the `Config` class at the top of `backtester.py`:

```python
class Config:
    REGIME_MA_PERIOD       = 200    # Long-term MA for regime filter
    LOCAL_LOW_LOOKBACK     = 10     # Bars to look back for local low
    LOCAL_LOW_CONFIRM_BARS = 3      # Confirmation bars after local low
    SUPPORT_LOOKBACK       = 20     # Bars for recent support (stop-loss base)
    STOP_BUFFER            = 0.001  # 0.1% buffer below support
    MOMENTUM_LOOKBACK      = 5      # Bars for momentum fade detection
    MOMENTUM_THRESHOLD     = 0.0    # Threshold: negative momentum triggers exit
    BARS_PER_YEAR          = 252 * 26  # For Sharpe annualisation (15-min bars)
```

---

## Output Files

### `trade_log/trade_log.csv`

Auto-generated after every run. Contains one row per trade:

| Column | Description |
|---|---|
| `entry_time` | Timestamp of entry bar |
| `exit_time` | Timestamp of exit bar |
| `entry_price` | Close price at entry |
| `exit_price` | Close price at exit |
| `stop_price` | Stop-loss level set at entry |
| `pnl_pct` | Trade return as a decimal (e.g. `0.0015` = +0.15%) |
| `exit_reason` | `stop_loss` / `momentum_fade` / `end_of_data` |
| `duration_bars` | Number of bars the trade was held |

### `backtest_results.png`

Four-panel chart saved after every run:
- **Top**: Price with regime MA overlay and trade entry markers (green = win, red = loss)
- **Middle**: Equity curve with total and annualised return annotated
- **Bottom left**: PnL distribution histogram split by wins and losses
- **Bottom right**: Exit reason breakdown bar chart



---

## Limitations

- **No transaction costs or slippage** — real-world returns will be lower, particularly given the high trade frequency
- **Single instrument** — strategy behaviour on other assets is untested
- **No walk-forward validation** — parameters were not optimised out-of-sample
- **High Sharpe caveat** — the Sharpe of 1.992 reflects low per-trade variance on a mean-reverting pullback strategy; it would compress significantly once costs are factored in

