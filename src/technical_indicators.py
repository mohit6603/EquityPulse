"""
Technical Indicators Module for EquityPulse.

This module computes popular technical analysis indicators on OHLCV DataFrames
using *only* Pandas and NumPy — no TA-Lib or other specialised libraries.

Technical indicators are mathematical calculations based on a stock's price
and/or volume history.  Traders use them to forecast future price direction.

Supported indicators
--------------------
- SMA  (Simple Moving Average)
- EMA  (Exponential Moving Average)
- RSI  (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR  (Average True Range)

Usage
-----
>>> from src.technical_indicators import add_all_indicators, get_signal_summary
>>> df = add_all_indicators(df)          # adds every indicator column
>>> signals = get_signal_summary(df)     # readable buy/sell summary
"""

import numpy as np
import pandas as pd

from src.utils import TRADING_DAYS

# ──────────────────────────────────────────────────────────────────────
#  1. Simple Moving Average (SMA)
# ──────────────────────────────────────────────────────────────────────


def add_sma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add Simple Moving Average columns to the DataFrame.

    What is SMA?
    -------------
    The Simple Moving Average is the unweighted mean of the previous *N*
    closing prices.  It smooths out short-term fluctuations and shows the
    longer-term trend.

    Formula
    -------
    SMA(N) = (C₁ + C₂ + … + Cₙ) / N

    where C is the closing price and N is the window size.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``Close`` column and DatetimeIndex.
    windows : list[int], optional
        Window sizes to compute.  Defaults to ``[20, 50]``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new columns ``SMA_<window>`` for each window.
    """
    if windows is None:
        windows = [20, 50]

    # Guard: nothing to compute on an empty frame
    if df.empty or "Close" not in df.columns:
        return df

    for window in windows:
        col_name = f"SMA_{window}"
        # rolling(window) creates a sliding window; .mean() averages it.
        # min_periods=1 ensures we get values even before we have 'window'
        # data-points (they'll just be averages of fewer points).
        df[col_name] = df["Close"].rolling(window=window, min_periods=1).mean()

    return df


# ──────────────────────────────────────────────────────────────────────
#  2. Exponential Moving Average (EMA)
# ──────────────────────────────────────────────────────────────────────


def add_ema(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add Exponential Moving Average columns to the DataFrame.

    What is EMA?
    ------------
    Unlike SMA, the EMA gives **more weight to recent prices**, making it
    react faster to new information.  The weighting decreases exponentially.

    Formula
    -------
    EMA_today = Close_today × k  +  EMA_yesterday × (1 − k)
    where k = 2 / (N + 1)   (the "smoothing factor")

    Pandas ``ewm(span=N)`` handles this calculation for us.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``Close`` column and DatetimeIndex.
    windows : list[int], optional
        Span values.  Defaults to ``[12, 26]`` (used by MACD).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new columns ``EMA_<window>``.
    """
    if windows is None:
        windows = [12, 26]

    if df.empty or "Close" not in df.columns:
        return df

    for window in windows:
        col_name = f"EMA_{window}"
        # span=window tells pandas the "N" in the smoothing factor formula.
        # adjust=False uses the recursive formula (faster and standard).
        df[col_name] = (
            df["Close"].ewm(span=window, adjust=False, min_periods=1).mean()
        )

    return df


# ──────────────────────────────────────────────────────────────────────
#  3. Relative Strength Index (RSI)
# ──────────────────────────────────────────────────────────────────────


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add the Relative Strength Index (RSI) column.

    What is RSI?
    ------------
    RSI measures the *speed and magnitude* of recent price changes to
    evaluate whether a stock is **overbought** or **oversold**.

    * RSI > 70  →  Overbought  (price may be due for a pullback)
    * RSI < 30  →  Oversold    (price may bounce back up)
    * 30–70     →  Neutral zone

    Formula
    -------
    1. delta   = Close − Close_previous
    2. gain    = max(delta, 0)
    3. loss    = abs(min(delta, 0))
    4. avg_gain = EWM(gain, span=period)
    5. avg_loss = EWM(loss, span=period)
    6. RS       = avg_gain / avg_loss
    7. RSI      = 100 − (100 / (1 + RS))

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with ``Close`` column.
    period : int
        Look-back period (default 14 — the most common setting).

    Returns
    -------
    pd.DataFrame
        DataFrame with a new ``RSI_<period>`` column.
    """
    if df.empty or "Close" not in df.columns or len(df) < 2:
        if not df.empty and "Close" in df.columns:
            df[f"RSI_{period}"] = np.nan
        return df

    # Step 1: daily price change
    delta = df["Close"].diff()

    # Step 2 & 3: separate gains from losses
    gain = delta.clip(lower=0)           # keep only positive changes
    loss = (-delta).clip(lower=0)        # absolute value of negative changes

    # Step 4 & 5: smoothed averages using exponential weighted mean
    # Using com (center-of-mass) = period − 1 is equivalent to span = period,
    # which matches the Wilder smoothing method used in the original RSI.
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Step 6: Relative Strength
    rs = avg_gain / avg_loss

    # Step 7: RSI formula — note: when avg_loss is 0, RS → ∞, RSI → 100
    rsi = 100.0 - (100.0 / (1.0 + rs))

    df[f"RSI_{period}"] = rsi

    return df


# ──────────────────────────────────────────────────────────────────────
#  4. Moving Average Convergence Divergence (MACD)
# ──────────────────────────────────────────────────────────────────────


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD, Signal Line, and Histogram columns.

    What is MACD?
    -------------
    MACD is a **trend-following momentum** indicator.  It shows the
    relationship between two EMAs of a stock's price.

    Components
    ----------
    * MACD Line      = EMA(12) − EMA(26)
    * Signal Line    = EMA(9) of the MACD Line
    * Histogram      = MACD Line − Signal Line

    Trading signals
    ~~~~~~~~~~~~~~~
    * MACD crosses **above** Signal → Bullish (buy signal)
    * MACD crosses **below** Signal → Bearish (sell signal)
    * Histogram growing → strengthening trend

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with ``Close`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``MACD``, ``MACD_Signal``, and ``MACD_Histogram``.
    """
    if df.empty or "Close" not in df.columns:
        return df

    # Compute the 12-day and 26-day EMAs (fast and slow)
    ema_fast = df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()

    # MACD Line: difference between fast and slow EMA
    df["MACD"] = ema_fast - ema_slow

    # Signal Line: 9-day EMA of the MACD Line itself
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()

    # Histogram: visual representation of MACD − Signal divergence
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    return df


# ──────────────────────────────────────────────────────────────────────
#  5. Bollinger Bands
# ──────────────────────────────────────────────────────────────────────


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: int = 2
) -> pd.DataFrame:
    """Add Bollinger Bands and %B columns.

    What are Bollinger Bands?
    -------------------------
    Created by John Bollinger, these bands expand and contract based on
    **volatility**.  When the market is volatile the bands widen; when the
    market is calm they narrow.

    Components
    ----------
    * Middle Band = SMA(window)
    * Upper Band  = Middle + num_std × σ(window)
    * Lower Band  = Middle − num_std × σ(window)
    * %B          = (Close − Lower) / (Upper − Lower)

    %B interpretation
    ~~~~~~~~~~~~~~~~~
    * %B > 1    → price is above the upper band (potentially overbought)
    * %B < 0    → price is below the lower band (potentially oversold)
    * %B ≈ 0.5  → price is at the middle band

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with ``Close`` column.
    window : int
        Rolling window for the moving average and std (default 20).
    num_std : int
        Number of standard deviations for band width (default 2).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``BB_Upper``, ``BB_Lower``, ``BB_Middle``, ``BB_Pct_B``.
    """
    if df.empty or "Close" not in df.columns:
        return df

    # Middle band is just the SMA
    rolling = df["Close"].rolling(window=window, min_periods=1)
    df["BB_Middle"] = rolling.mean()

    # Standard deviation of closing prices over the window
    rolling_std = rolling.std()

    # Upper and lower bands are num_std standard deviations away from the mean
    df["BB_Upper"] = df["BB_Middle"] + (num_std * rolling_std)
    df["BB_Lower"] = df["BB_Middle"] - (num_std * rolling_std)

    # %B tells us where the current price sits relative to the bands
    band_width = df["BB_Upper"] - df["BB_Lower"]
    # Avoid division by zero when bands are flat (e.g., only 1 data point)
    df["BB_Pct_B"] = np.where(
        band_width != 0,
        (df["Close"] - df["BB_Lower"]) / band_width,
        0.5,  # default to middle when band width is zero
    )

    return df


# ──────────────────────────────────────────────────────────────────────
#  6. Average True Range (ATR)
# ──────────────────────────────────────────────────────────────────────


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range (ATR) column.

    What is ATR?
    ------------
    ATR measures **market volatility** by decomposing the entire range of a
    stock's price for the period.  Higher ATR = more volatile market.
    Unlike indicators that track direction, ATR only tracks *magnitude*.

    Formula
    -------
    True Range (TR) for each day is the **greatest** of:
        1. High − Low                   (today's range)
        2. |High − Previous Close|      (gap up detection)
        3. |Low  − Previous Close|      (gap down detection)

    ATR = Simple moving average of TR over ``period`` days.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with ``High``, ``Low``, ``Close`` columns.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    pd.DataFrame
        DataFrame with a new ``ATR_<period>`` column.
    """
    required = {"High", "Low", "Close"}
    if df.empty or not required.issubset(df.columns) or len(df) < 2:
        if not df.empty:
            df[f"ATR_{period}"] = np.nan
        return df

    # Previous closing price (shifted by 1 day)
    prev_close = df["Close"].shift(1)

    # Three components of True Range
    tr1 = df["High"] - df["Low"]               # intra-day range
    tr2 = (df["High"] - prev_close).abs()       # gap-up range
    tr3 = (df["Low"] - prev_close).abs()        # gap-down range

    # True Range is the max of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is the rolling mean of the True Range series
    df[f"ATR_{period}"] = true_range.rolling(
        window=period, min_periods=1
    ).mean()

    return df


# ──────────────────────────────────────────────────────────────────────
#  7. Convenience: add ALL indicators at once
# ──────────────────────────────────────────────────────────────────────


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add every supported technical indicator to the DataFrame.

    This is a convenience wrapper that calls each ``add_*`` function in
    sequence.  Use this when you want a fully-enriched DataFrame for
    analysis or charting.

    Indicators added
    ----------------
    * SMA (20, 50)
    * EMA (12, 26)
    * RSI (14)
    * MACD, Signal, Histogram
    * Bollinger Bands (20-day, 2σ)
    * ATR (14)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with all indicator columns appended.
    """
    if df.empty:
        return df

    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)

    return df


# ──────────────────────────────────────────────────────────────────────
#  8. Signal Summary
# ──────────────────────────────────────────────────────────────────────


def get_signal_summary(df: pd.DataFrame) -> dict:
    """Generate a human-readable summary of the latest technical signals.

    This function looks at the **most recent row** of the DataFrame and
    interprets each indicator into a simple status string that even a
    beginner can understand.

    Signal Logic
    -------------
    * **RSI**: > 70 = Overbought, < 30 = Oversold, else Neutral
    * **MACD**: MACD > Signal = Bullish Crossover, else Bearish
    * **SMA Trend**: SMA_20 > SMA_50 = Golden Cross (bullish),
      SMA_20 < SMA_50 = Death Cross (bearish), else Neutral
    * **Bollinger %B**: > 0.8 = Near Upper Band, < 0.2 = Near Lower Band,
      else Middle Band
    * **Overall**: majority vote across the four signals above

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame that already has indicators added
        (call ``add_all_indicators`` first).

    Returns
    -------
    dict
        Dictionary with keys ``rsi_status``, ``macd_status``,
        ``sma_trend``, ``bollinger_position``, and ``overall_signal``.
    """
    # Default values when data is insufficient
    summary: dict = {
        "rsi_status": "Insufficient Data",
        "macd_status": "Insufficient Data",
        "sma_trend": "Insufficient Data",
        "bollinger_position": "Insufficient Data",
        "overall_signal": "Insufficient Data",
    }

    if df.empty or len(df) < 2:
        return summary

    # Grab the most recent row
    latest = df.iloc[-1]

    # ── RSI ──────────────────────────────────────────────
    bullish_count = 0
    bearish_count = 0

    rsi_col = "RSI_14"
    if rsi_col in df.columns and pd.notna(latest.get(rsi_col)):
        rsi_value = latest[rsi_col]
        if rsi_value > 70:
            summary["rsi_status"] = "Overbought"
            bearish_count += 1          # overbought → expect price drop
        elif rsi_value < 30:
            summary["rsi_status"] = "Oversold"
            bullish_count += 1          # oversold → expect price rise
        else:
            summary["rsi_status"] = "Neutral"

    # ── MACD ─────────────────────────────────────────────
    if (
        "MACD" in df.columns
        and "MACD_Signal" in df.columns
        and pd.notna(latest.get("MACD"))
        and pd.notna(latest.get("MACD_Signal"))
    ):
        if latest["MACD"] > latest["MACD_Signal"]:
            summary["macd_status"] = "Bullish Crossover"
            bullish_count += 1
        else:
            summary["macd_status"] = "Bearish Crossover"
            bearish_count += 1

    # ── SMA Trend (Golden Cross / Death Cross) ───────────
    # Golden Cross: short-term SMA crosses above long-term SMA → bullish
    # Death Cross:  short-term SMA crosses below long-term SMA → bearish
    if (
        "SMA_20" in df.columns
        and "SMA_50" in df.columns
        and pd.notna(latest.get("SMA_20"))
        and pd.notna(latest.get("SMA_50"))
    ):
        if latest["SMA_20"] > latest["SMA_50"]:
            summary["sma_trend"] = "Golden Cross"
            bullish_count += 1
        elif latest["SMA_20"] < latest["SMA_50"]:
            summary["sma_trend"] = "Death Cross"
            bearish_count += 1
        else:
            summary["sma_trend"] = "Neutral"

    # ── Bollinger Bands position ─────────────────────────
    if "BB_Pct_B" in df.columns and pd.notna(latest.get("BB_Pct_B")):
        pct_b = latest["BB_Pct_B"]
        if pct_b > 0.8:
            summary["bollinger_position"] = "Near Upper Band"
            bearish_count += 1          # near upper band → may be overbought
        elif pct_b < 0.2:
            summary["bollinger_position"] = "Near Lower Band"
            bullish_count += 1          # near lower band → may be oversold
        else:
            summary["bollinger_position"] = "Middle Band"

    # ── Overall Signal (majority vote) ───────────────────
    if bullish_count > bearish_count:
        summary["overall_signal"] = "Bullish"
    elif bearish_count > bullish_count:
        summary["overall_signal"] = "Bearish"
    else:
        summary["overall_signal"] = "Neutral"

    return summary
