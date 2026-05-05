"""
Structural Pullback Backtest
============================
A rule-based structural bullish pullback strategy backtested on 15-min OHLC data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    # Regime filter
    REGIME_MA_PERIOD: int = 200          # Long-term MA for bullish regime detection

    # Local-low detection
    LOCAL_LOW_LOOKBACK: int = 10         # Bars to look back for local low
    LOCAL_LOW_CONFIRM_BARS: int = 3      # Bars after low to confirm upward response

    # Exit: stop-loss
    SUPPORT_LOOKBACK: int = 20           # Bars to define recent support
    STOP_BUFFER: float = 0.001           # 0.1% below support as stop

    # Exit: take-profit / momentum fade
    MOMENTUM_LOOKBACK: int = 5           # Bars to check momentum fade
    MOMENTUM_THRESHOLD: float = 0.0      # Price change threshold for fade detection

    # Data
    DATA_PATH: str = "data_set/train_data1.csv"
    TRADE_LOG_PATH: str = "trade_log/trade_log.csv"


cfg = Config()


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load and validate OHLC data."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"[Data] Loaded {len(df):,} rows | "
          f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df


def generate_synthetic_data(n: int = 28000) -> pd.DataFrame:
    """
    Generate synthetic 15-min OHLC data that mimics the described dataset:
    mixed trending / sideways regimes, multiple years.
    Used when train_data1.csv is not available.
    """
    np.random.seed(42)
    timestamps = pd.date_range(start="2020-01-02 09:30", periods=n, freq="15min")

    # Simulate regime-switching price process
    price = 100.0
    prices = [price]
    regime = 1          # 1 = bull, 0 = sideways, -1 = bear
    regime_counter = 0

    for _ in range(n - 1):
        regime_counter += 1
        if regime_counter > np.random.randint(200, 600):
            regime = np.random.choice([1, 1, 0, -1], p=[0.5, 0.2, 0.2, 0.1])
            regime_counter = 0

        drift = {"1": 0.0003, "0": 0.0, "-1": -0.0002}[str(regime)]
        vol = 0.003
        ret = drift + vol * np.random.randn()
        price = max(price * (1 + ret), 1.0)
        prices.append(price)

    closes = np.array(prices)
    noise = np.random.uniform(0.001, 0.005, n)

    opens  = closes * (1 + np.random.uniform(-0.002, 0.002, n))
    highs  = np.maximum(opens, closes) * (1 + noise)
    lows   = np.minimum(opens, closes) * (1 - noise)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open":  np.round(opens,  4),
        "high":  np.round(highs,  4),
        "low":   np.round(lows,   4),
        "close": np.round(closes, 4),
    })


# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Regime filter
    df["regime_ma"] = df["close"].rolling(cfg.REGIME_MA_PERIOD).mean()
    df["is_bullish_regime"] = df["close"] > df["regime_ma"]

    # Rolling local low (minimum close over lookback)
    df["rolling_low"] = df["low"].rolling(cfg.LOCAL_LOW_LOOKBACK).min()

    # Recent support (for stop-loss)
    df["recent_support"] = df["low"].rolling(cfg.SUPPORT_LOOKBACK).min()

    # Momentum: n-bar price change
    df["momentum"] = df["close"].diff(cfg.MOMENTUM_LOOKBACK)

    return df


# ─────────────────────────────────────────────
#  SIGNAL DETECTION
# ─────────────────────────────────────────────
def is_local_low(df: pd.DataFrame, i: int) -> bool:
    """
    True if bar i is a confirmed local low:
    - current low touches rolling minimum
    - subsequent bars show upward response
    """
    if i < cfg.LOCAL_LOW_LOOKBACK:
        return False
    if i + cfg.LOCAL_LOW_CONFIRM_BARS >= len(df):
        return False

    current_low = df["low"].iloc[i]
    rolling_min = df["rolling_low"].iloc[i]

    # Price touches the local low
    at_low = current_low <= rolling_min * 1.002   # 0.2% tolerance

    # Confirm upward response in subsequent bars
    future_closes = df["close"].iloc[i+1 : i+1+cfg.LOCAL_LOW_CONFIRM_BARS]
    upward_response = future_closes.iloc[-1] > df["close"].iloc[i]

    return bool(at_low and upward_response)


# ─────────────────────────────────────────────
#  BACKTESTER
# ─────────────────────────────────────────────
class Trade:
    def __init__(self, entry_bar: int, entry_price: float,
                 stop_price: float, entry_time):
        self.entry_bar    = entry_bar
        self.entry_price  = entry_price
        self.stop_price   = stop_price
        self.entry_time   = entry_time
        self.exit_bar     = None
        self.exit_price   = None
        self.exit_time    = None
        self.exit_reason  = None
        self.pnl_pct      = None

    def close(self, bar: int, price: float, time, reason: str):
        self.exit_bar    = bar
        self.exit_price  = price
        self.exit_time   = time
        self.exit_reason = reason
        self.pnl_pct     = (price - self.entry_price) / self.entry_price

    def to_dict(self) -> dict:
        return {
            "entry_time":   self.entry_time,
            "exit_time":    self.exit_time,
            "entry_price":  self.entry_price,
            "exit_price":   self.exit_price,
            "stop_price":   self.stop_price,
            "pnl_pct":      self.pnl_pct,
            "exit_reason":  self.exit_reason,
            "duration_bars": self.exit_bar - self.entry_bar,
        }


def run_backtest(df: pd.DataFrame) -> tuple[list[Trade], pd.Series]:
    """
    Execute backtest. Returns (trades list, equity curve).
    """
    df = compute_indicators(df)

    trades: list[Trade] = []
    active_trade: Trade | None = None
    equity = 1.0
    equity_curve = [equity]

    # We need confirm bars before we can enter, so start lookback-safe
    start = cfg.REGIME_MA_PERIOD + cfg.LOCAL_LOW_LOOKBACK + cfg.LOCAL_LOW_CONFIRM_BARS

    for i in range(start, len(df)):
        row = df.iloc[i]

        # ── Manage open trade ──────────────────
        if active_trade is not None:
            price = row["close"]

            # Stop-loss: price breaks below recent support
            stop_hit = price < active_trade.stop_price

            # Momentum fade: price rose but momentum turned negative
            momentum_fade = (
                price > active_trade.entry_price and
                row["momentum"] < cfg.MOMENTUM_THRESHOLD
            )

            if stop_hit:
                active_trade.close(i, price, row["timestamp"], "stop_loss")
                equity *= (1 + active_trade.pnl_pct)
                trades.append(active_trade)
                active_trade = None

            elif momentum_fade:
                active_trade.close(i, price, row["timestamp"], "momentum_fade")
                equity *= (1 + active_trade.pnl_pct)
                trades.append(active_trade)
                active_trade = None

        # ── Look for new entry ─────────────────
        if active_trade is None:
            in_regime = row["is_bullish_regime"]

            if in_regime and is_local_low(df, i):
                entry_price = df["close"].iloc[i + cfg.LOCAL_LOW_CONFIRM_BARS]
                support     = row["recent_support"]
                stop_price  = support * (1 - cfg.STOP_BUFFER)
                entry_time  = df["timestamp"].iloc[i + cfg.LOCAL_LOW_CONFIRM_BARS]
                entry_bar   = i + cfg.LOCAL_LOW_CONFIRM_BARS

                active_trade = Trade(entry_bar, entry_price, stop_price, entry_time)
                i = entry_bar   # skip ahead (loop handles this naturally)

        equity_curve.append(equity)

    # Close any lingering trade at end
    if active_trade is not None:
        last = df.iloc[-1]
        active_trade.close(len(df)-1, last["close"], last["timestamp"], "end_of_data")
        equity *= (1 + active_trade.pnl_pct)
        trades.append(active_trade)

    equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    return trades, equity_series


# ─────────────────────────────────────────────
#  PERFORMANCE METRICS
# ─────────────────────────────────────────────
def compute_metrics(trades: list[Trade], equity: pd.Series) -> dict:
    if not trades:
        return {}

    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total_return = equity.iloc[-1] - 1.0

    # Sharpe (annualised on per-trade returns, ~1944 trades across ~8 years)
    if pnls.std() > 0:
        # Annualise: assume ~252 trading days × 26 bars/day for 15-min data
        bars_per_year = 252 * 26
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Max drawdown on equity curve
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()

    return {
        "total_trades":     len(trades),
        "winning_trades":   len(wins),
        "losing_trades":    len(losses),
        "win_rate":         len(wins) / len(trades) if trades else 0,
        "avg_win":          wins.mean()   if len(wins)   else 0,
        "avg_loss":         losses.mean() if len(losses) else 0,
        "profit_factor":    (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf,
        "total_return":     total_return,
        "sharpe_ratio":     sharpe,
        "max_drawdown":     max_dd,
        "avg_trade_pnl":    pnls.mean(),
        "std_trade_pnl":    pnls.std(),
        "avg_duration_bars": np.mean([t.exit_bar - t.entry_bar for t in trades]),
    }


def print_metrics(m: dict):
    sep = "─" * 42
    print(f"\n{'═'*42}")
    print("  STRUCTURAL PULLBACK BACKTEST RESULTS")
    print(f"{'═'*42}")
    print(f"  Total Trades       : {m['total_trades']:>10,}")
    print(f"  Win Rate           : {m['win_rate']:>10.1%}")
    print(f"  Avg Win            : {m['avg_win']:>10.4%}")
    print(f"  Avg Loss           : {m['avg_loss']:>10.4%}")
    print(f"  Profit Factor      : {m['profit_factor']:>10.3f}")
    print(sep)
    print(f"  Total Return       : {m['total_return']:>10.2%}")
    print(f"  Sharpe Ratio       : {m['sharpe_ratio']:>10.3f}")
    print(f"  Max Drawdown       : {m['max_drawdown']:>10.2%}")
    print(sep)
    print(f"  Avg Trade PnL      : {m['avg_trade_pnl']:>10.4%}")
    print(f"  Avg Duration (bars): {m['avg_duration_bars']:>10.1f}")
    print(f"{'═'*42}\n")


# ─────────────────────────────────────────────
#  TRADE LOG
# ─────────────────────────────────────────────
def save_trade_log(trades: list[Trade], path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([t.to_dict() for t in trades]).to_csv(path, index=False)
    print(f"[Log] Trade log saved → {path}")


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GREEN     = "#3fb950"
RED       = "#f85149"
BLUE      = "#58a6ff"
AMBER     = "#e3b341"
MUTED     = "#8b949e"
WHITE     = "#e6edf3"


def plot_results(df: pd.DataFrame, trades: list[Trade],
                 equity: pd.Series, metrics: dict):
    df = compute_indicators(df)

    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[2, 1, 1],
                            hspace=0.45, wspace=0.3)

    ax_price  = fig.add_subplot(gs[0, :])
    ax_equity = fig.add_subplot(gs[1, :])
    ax_dist   = fig.add_subplot(gs[2, 0])
    ax_reason = fig.add_subplot(gs[2, 1])

    for ax in [ax_price, ax_equity, ax_dist, ax_reason]:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── Price chart (last 2000 bars for clarity) ───────────
    n_show = 2000
    df_show = df.iloc[-n_show:].copy()
    idx = df_show.index

    ax_price.plot(idx, df_show["close"], color=BLUE, lw=0.6, alpha=0.9, label="Close")
    ax_price.plot(idx, df_show["regime_ma"], color=AMBER, lw=1.1,
                  linestyle="--", alpha=0.7, label=f"Regime MA ({cfg.REGIME_MA_PERIOD})")

    # Shade bullish regime
    bull = df_show["is_bullish_regime"]
    ax_price.fill_between(idx, df_show["close"].min(), df_show["close"].max(),
                          where=bull, alpha=0.06, color=GREEN, label="Bullish Regime")

    # Plot trades in visible window
    min_idx = df_show.index[0]
    for t in trades:
        if t.entry_bar >= min_idx and t.entry_bar < min_idx + n_show:
            color = GREEN if t.pnl_pct > 0 else RED
            ax_price.axvline(t.entry_bar, color=color, lw=0.4, alpha=0.5)

    ax_price.set_title("Price  ·  Regime MA  ·  Trade Entries (last 2,000 bars)",
                        color=WHITE, fontsize=11, fontweight="bold", pad=8)
    ax_price.set_ylabel("Price", color=MUTED, fontsize=9)
    ax_price.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#30363d",
                    labelcolor=WHITE, loc="upper left")
    ax_price.grid(alpha=0.12, color=MUTED)

    # ── Equity curve ──────────────────────────────────────
    eq_idx = equity.index
    ax_equity.plot(eq_idx, equity.values, color=GREEN, lw=1.2)
    ax_equity.fill_between(eq_idx, 1, equity.values,
                           where=equity.values >= 1, alpha=0.15, color=GREEN)
    ax_equity.fill_between(eq_idx, 1, equity.values,
                           where=equity.values < 1,  alpha=0.15, color=RED)
    ax_equity.axhline(1.0, color=MUTED, lw=0.7, linestyle="--")

    # Annotate final return
    final = equity.iloc[-1]
    color = GREEN if final >= 1 else RED
    ax_equity.annotate(f"+{(final-1)*100:.2f}%" if final >= 1 else f"{(final-1)*100:.2f}%",
                       xy=(eq_idx[-1], final),
                       xytext=(-60, 10), textcoords="offset points",
                       color=color, fontsize=9, fontweight="bold")

    ax_equity.set_title("Equity Curve", color=WHITE, fontsize=11,
                        fontweight="bold", pad=8)
    ax_equity.set_ylabel("Equity (×)", color=MUTED, fontsize=9)
    ax_equity.grid(alpha=0.12, color=MUTED)

    # ── PnL distribution ──────────────────────────────────
    pnls = np.array([t.pnl_pct * 100 for t in trades])
    bins = np.linspace(pnls.min(), pnls.max(), 40)
    ax_dist.hist(pnls[pnls > 0], bins=bins, color=GREEN, alpha=0.7, label="Win")
    ax_dist.hist(pnls[pnls <= 0], bins=bins, color=RED,   alpha=0.7, label="Loss")
    ax_dist.axvline(0, color=WHITE, lw=0.8, linestyle="--")
    ax_dist.axvline(pnls.mean(), color=AMBER, lw=1.2, linestyle="-",
                    label=f"Mean: {pnls.mean():.3f}%")
    ax_dist.set_title("Trade PnL Distribution (%)", color=WHITE,
                      fontsize=10, fontweight="bold", pad=6)
    ax_dist.set_xlabel("PnL (%)", color=MUTED, fontsize=8)
    ax_dist.set_ylabel("Frequency",  color=MUTED, fontsize=8)
    ax_dist.legend(fontsize=7, facecolor=PANEL_BG, edgecolor="#30363d",
                   labelcolor=WHITE)
    ax_dist.grid(alpha=0.12, color=MUTED)

    # ── Exit reason breakdown ────────────────────────────
    reasons = pd.Series([t.exit_reason for t in trades]).value_counts()
    bar_colors = {
        "stop_loss":       RED,
        "momentum_fade":   AMBER,
        "end_of_data":     MUTED,
    }
    colors_list = [bar_colors.get(r, BLUE) for r in reasons.index]
    bars = ax_reason.barh(reasons.index, reasons.values, color=colors_list, alpha=0.85)
    for bar, val in zip(bars, reasons.values):
        ax_reason.text(val + 2, bar.get_y() + bar.get_height()/2,
                       str(val), va="center", color=WHITE, fontsize=8)
    ax_reason.set_title("Exit Reason Breakdown", color=WHITE,
                        fontsize=10, fontweight="bold", pad=6)
    ax_reason.set_xlabel("Count", color=MUTED, fontsize=8)
    ax_reason.grid(alpha=0.12, color=MUTED, axis="x")

    # ── Summary stats text box ──────────────────────────
    stats = (
        f"Total Return : {metrics['total_return']:+.2%}\n"
        f"Sharpe Ratio : {metrics['sharpe_ratio']:.3f}\n"
        f"Max Drawdown : {metrics['max_drawdown']:.2%}\n"
        f"Total Trades : {metrics['total_trades']:,}\n"
        f"Win Rate     : {metrics['win_rate']:.1%}\n"
        f"Profit Factor: {metrics['profit_factor']:.3f}"
    )
    fig.text(0.76, 0.96, stats,
             transform=fig.transFigure,
             fontsize=9, color=WHITE,
             fontfamily="monospace",
             verticalalignment="top",
             bbox=dict(facecolor=PANEL_BG, edgecolor="#30363d",
                       boxstyle="round,pad=0.5", alpha=0.9))

    fig.suptitle("Structural Pullback Backtest  ·  15-Min OHLC",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.99)

    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    print("[Plot] Saved → backtest_results.png")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    import os

    # Load or generate data
    if os.path.exists(cfg.DATA_PATH):
        df = load_data(cfg.DATA_PATH)
    else:
        print(f"[Data] '{cfg.DATA_PATH}' not found → generating synthetic data")
        df = generate_synthetic_data(28_000)

    print("[Backtest] Running …")
    trades, equity = run_backtest(df)

    metrics = compute_metrics(trades, equity)
    print_metrics(metrics)

    os.makedirs(os.path.dirname(cfg.TRADE_LOG_PATH), exist_ok=True)
    save_trade_log(trades, cfg.TRADE_LOG_PATH)

    plot_results(df, trades, equity, metrics)


if __name__ == "__main__":
    main()
