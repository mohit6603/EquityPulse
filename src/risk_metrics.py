"""
Risk Metrics Module for EquityPulse.

This module computes essential financial risk metrics used by portfolio managers
and individual investors to evaluate stock performance **on a risk-adjusted
basis**.

Why risk metrics matter
-----------------------
Raw returns don't tell the whole story.  Two stocks can both return 15 % per
year, but if one swings wildly every day while the other climbs steadily,
they carry very different risks.  The functions here quantify that risk.

Supported metrics
-----------------
- Daily Returns
- Sharpe Ratio       (risk-adjusted return)
- Maximum Drawdown   (worst peak-to-trough loss)
- Value at Risk      (potential loss at a confidence level)
- Beta               (sensitivity to market movements)
- Jensen's Alpha     (excess return vs. the market)
- Annualized Volatility

Usage
-----
>>> from src.risk_metrics import compute_all_metrics
>>> metrics = compute_all_metrics(stock_df, market_df=nifty_df)
"""

import numpy as np
import pandas as pd

from src.utils import RISK_FREE_RATE, TRADING_DAYS


# ──────────────────────────────────────────────────────────────────────
#  1. Daily Returns
# ──────────────────────────────────────────────────────────────────────


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``Daily_Returns`` column to the DataFrame.

    What are daily returns?
    -----------------------
    The *percentage change* of the closing price from one trading day to
    the next.  This is the fundamental building block for almost every
    risk metric.

    Formula
    -------
    Daily Return = (Close_today − Close_yesterday) / Close_yesterday

    Pandas ``pct_change()`` computes exactly this.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a new ``Daily_Returns`` column.  The first
        row will be NaN because there is no "yesterday" to compare to.
    """
    if df.empty or "Close" not in df.columns:
        return df

    # pct_change() = (x_t - x_{t-1}) / x_{t-1}
    df["Daily_Returns"] = df["Close"].pct_change()

    return df


# ──────────────────────────────────────────────────────────────────────
#  2. Sharpe Ratio
# ──────────────────────────────────────────────────────────────────────


def compute_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float | None = None
) -> float:
    """Compute the annualized Sharpe Ratio.

    What is the Sharpe Ratio?
    -------------------------
    Invented by Nobel laureate William Sharpe, it measures the **return
    per unit of risk**.  A higher Sharpe means better risk-adjusted
    performance.

    Interpretation
    ~~~~~~~~~~~~~~
    * < 1    →  Sub-optimal risk-reward
    * 1 – 2  →  Good
    * 2 – 3  →  Very good
    * > 3    →  Excellent

    Formula
    -------
    Sharpe = (mean_daily_return − daily_risk_free_rate) / std_daily × √(TRADING_DAYS)

    We annualise by multiplying by √252 because variance scales linearly
    with time, so standard deviation scales with the square root.

    Parameters
    ----------
    returns : pd.Series
        Series of daily returns (output of ``compute_daily_returns``).
    risk_free_rate : float, optional
        Annual risk-free rate.  Uses ``RISK_FREE_RATE`` from utils if
        not provided (default ≈ 7 % for India).

    Returns
    -------
    float
        Annualized Sharpe Ratio.  Returns ``np.nan`` if computation is
        not possible.
    """
    if risk_free_rate is None:
        risk_free_rate = RISK_FREE_RATE

    # Drop NaN values that pct_change() creates in the first row
    clean = returns.dropna()

    if len(clean) < 2:
        return np.nan

    # Convert annual risk-free rate to a daily rate
    daily_rf = risk_free_rate / TRADING_DAYS

    # Excess return per day
    excess_return = clean.mean() - daily_rf

    # Standard deviation of daily returns
    std_daily = clean.std()

    if std_daily == 0:
        # No variation → ratio is undefined (infinite or meaningless)
        return np.nan

    # Annualise: multiply by √252
    sharpe = (excess_return / std_daily) * np.sqrt(TRADING_DAYS)

    return float(sharpe)


# ──────────────────────────────────────────────────────────────────────
#  3. Maximum Drawdown
# ──────────────────────────────────────────────────────────────────────


def compute_max_drawdown(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """Compute the Maximum Drawdown (MDD) of a stock.

    What is Maximum Drawdown?
    -------------------------
    MDD measures the **largest peak-to-trough decline** before a new
    peak is reached.  It answers the question: "What is the worst loss an
    investor could have suffered?"

    Formula
    -------
    1. cumulative_max = running maximum of Close prices
    2. drawdown       = (Close − cumulative_max) / cumulative_max
    3. max_drawdown   = min(drawdown)    ← most negative value

    A drawdown of −0.25 means the stock fell 25 % from its peak.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``Close`` column.

    Returns
    -------
    tuple[float, pd.Series]
        * max_drawdown_pct : The worst drawdown as a decimal (negative).
        * drawdown_series  : The full drawdown series for charting.
        Returns ``(np.nan, pd.Series(dtype=float))`` when data is
        insufficient.
    """
    if df.empty or "Close" not in df.columns or len(df) < 2:
        return np.nan, pd.Series(dtype=float)

    close = df["Close"]

    # Running peak: the highest price seen so far at each point in time
    cumulative_max = close.cummax()

    # Drawdown at each point: how far below the peak we are (as a %)
    drawdown_series = (close - cumulative_max) / cumulative_max

    # The worst (most negative) drawdown in the entire period
    max_drawdown = float(drawdown_series.min())

    return max_drawdown, drawdown_series


# ──────────────────────────────────────────────────────────────────────
#  4. Value at Risk (VaR) — Historical Method
# ──────────────────────────────────────────────────────────────────────


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute Historical Value at Risk (VaR).

    What is VaR?
    ------------
    VaR answers: "With X % confidence, what is the **most** I could lose
    in a single day?"

    For example, a 95 % VaR of −2 % means that on 95 % of trading days,
    the daily loss will not exceed 2 %.

    Method
    ------
    **Historical simulation**: we simply look at the actual distribution
    of past returns and find the appropriate percentile.

    Formula
    -------
    VaR(95 %) = 5th percentile of daily returns

    Why percentile?  Because we want the return value below which only
    5 % of historical returns fall — i.e. the worst 5 % of days.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
        Confidence level (default 0.95 = 95 %).

    Returns
    -------
    float
        The VaR value (a negative number representing potential loss).
        Returns ``np.nan`` if not enough data.
    """
    clean = returns.dropna()

    if len(clean) < 2:
        return np.nan

    # For 95 % confidence, we want the 5th percentile
    # (100 − 95 = 5 → the left tail of the distribution)
    percentile = (1 - confidence) * 100
    var_value = float(np.percentile(clean, percentile))

    return var_value


# ──────────────────────────────────────────────────────────────────────
#  5. Beta — Sensitivity to Market
# ──────────────────────────────────────────────────────────────────────


def compute_beta(
    stock_returns: pd.Series, market_returns: pd.Series
) -> float:
    """Compute Beta — a stock's sensitivity to overall market movements.

    What is Beta?
    -------------
    * Beta = 1  →  stock moves exactly with the market
    * Beta > 1  →  stock is **more** volatile than the market
    * Beta < 1  →  stock is **less** volatile (defensive)
    * Beta < 0  →  stock moves **opposite** to the market (rare)

    Formula (Linear Algebra!)
    -------------------------
    Beta = Cov(stock, market) / Var(market)

    This is the slope of the **ordinary least squares** regression line
    when you plot stock returns (Y-axis) against market returns (X-axis).

    ``np.cov`` returns the **covariance matrix** — a 2×2 matrix:

        [[Var(stock),       Cov(stock, market)],
         [Cov(stock, market), Var(market)      ]]

    So ``cov_matrix[0, 1]`` is the covariance and ``cov_matrix[1, 1]``
    is the variance of the market.

    Parameters
    ----------
    stock_returns : pd.Series
        Daily returns of the stock.
    market_returns : pd.Series
        Daily returns of the market benchmark (e.g., Nifty 50).

    Returns
    -------
    float
        Beta coefficient.  Returns ``np.nan`` if computation fails.
    """
    # Align both series on their shared dates and drop NaN rows
    combined = pd.concat(
        [stock_returns.rename("stock"), market_returns.rename("market")],
        axis=1,
    ).dropna()

    if len(combined) < 2:
        return np.nan

    # np.cov returns the covariance matrix (2×2)
    cov_matrix = np.cov(combined["stock"], combined["market"])

    market_variance = cov_matrix[1, 1]

    if market_variance == 0:
        # Market has zero variance → beta is undefined
        return np.nan

    # Beta = Cov(stock, market) / Var(market)
    beta = cov_matrix[0, 1] / market_variance

    return float(beta)


# ──────────────────────────────────────────────────────────────────────
#  6. Jensen's Alpha
# ──────────────────────────────────────────────────────────────────────


def compute_alpha(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    beta: float,
    risk_free_rate: float | None = None,
) -> float:
    """Compute Jensen's Alpha — excess return beyond what CAPM predicts.

    What is Alpha?
    --------------
    Alpha measures a stock's **outperformance** (or underperformance)
    relative to its expected return given its risk level (beta).

    * Alpha > 0 →  the stock beat the market on a risk-adjusted basis
    * Alpha < 0 →  the stock lagged behind

    Formula (CAPM-based)
    --------------------
    Expected Return = Rf + β × (Rm − Rf)
    Alpha            = Actual Return − Expected Return
                     = R_stock − [Rf + β × (R_market − Rf)]

    where:
        Rf       = risk-free rate (annualised)
        Rm       = market return (annualised)
        β        = beta of the stock
        R_stock  = stock's actual annualised return

    Parameters
    ----------
    stock_returns : pd.Series
        Daily returns of the stock.
    market_returns : pd.Series
        Daily returns of the market benchmark.
    beta : float
        Pre-computed beta (from ``compute_beta``).
    risk_free_rate : float, optional
        Annual risk-free rate.  Defaults to ``RISK_FREE_RATE``.

    Returns
    -------
    float
        Annualized Jensen's Alpha.  Returns ``np.nan`` when inputs are
        insufficient.
    """
    if risk_free_rate is None:
        risk_free_rate = RISK_FREE_RATE

    stock_clean = stock_returns.dropna()
    market_clean = market_returns.dropna()

    if len(stock_clean) < 2 or len(market_clean) < 2 or np.isnan(beta):
        return np.nan

    # Annualise mean daily returns  (mean_daily × TRADING_DAYS)
    stock_annual = stock_clean.mean() * TRADING_DAYS
    market_annual = market_clean.mean() * TRADING_DAYS

    # CAPM expected return
    expected_return = risk_free_rate + beta * (market_annual - risk_free_rate)

    # Alpha is the difference between actual and expected
    alpha = stock_annual - expected_return

    return float(alpha)


# ──────────────────────────────────────────────────────────────────────
#  7. Annualized Volatility
# ──────────────────────────────────────────────────────────────────────


def compute_volatility(returns: pd.Series) -> float:
    """Compute annualized volatility (standard deviation of returns).

    What is Volatility?
    -------------------
    Volatility quantifies **how much** a stock's price fluctuates.  High
    volatility ≠ bad — it simply means bigger swings in both directions.

    Formula
    -------
    Annualized Volatility = σ_daily × √(TRADING_DAYS)

    We multiply by √252 because:
    - Variance scales linearly with time  (σ² × T)
    - Therefore standard deviation scales with √T

    Parameters
    ----------
    returns : pd.Series
        Daily return series.

    Returns
    -------
    float
        Annualized volatility as a decimal (e.g. 0.30 = 30 %).
        Returns ``np.nan`` for insufficient data.
    """
    clean = returns.dropna()

    if len(clean) < 2:
        return np.nan

    volatility = clean.std() * np.sqrt(TRADING_DAYS)

    return float(volatility)


# ──────────────────────────────────────────────────────────────────────
#  8. Compute ALL metrics at once
# ──────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    df: pd.DataFrame, market_df: pd.DataFrame | None = None
) -> dict:
    """Compute every risk metric and return them as a dictionary.

    This is a convenience function that runs all individual metric
    functions and packages the results.  If a market benchmark DataFrame
    is supplied, Beta and Alpha are also computed.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame for the stock.
    market_df : pd.DataFrame, optional
        OHLCV DataFrame for the market benchmark (e.g. Nifty 50).
        Must also have a ``Close`` column with a DatetimeIndex.

    Returns
    -------
    dict
        Keys: ``sharpe_ratio``, ``max_drawdown``, ``var_95``,
        ``volatility``, ``beta``, ``alpha``.  Beta and Alpha are
        ``np.nan`` when no market data is provided.
    """
    # Start with default values
    metrics: dict = {
        "sharpe_ratio": np.nan,
        "max_drawdown": np.nan,
        "var_95": np.nan,
        "volatility": np.nan,
        "beta": np.nan,
        "alpha": np.nan,
    }

    if df.empty or "Close" not in df.columns:
        return metrics

    # Step 1: Ensure daily returns are present
    if "Daily_Returns" not in df.columns:
        df = compute_daily_returns(df)

    returns = df["Daily_Returns"]

    # Step 2: Individual risk metrics
    metrics["sharpe_ratio"] = compute_sharpe_ratio(returns)
    metrics["max_drawdown"], _ = compute_max_drawdown(df)
    metrics["var_95"] = compute_var(returns, confidence=0.95)
    metrics["volatility"] = compute_volatility(returns)

    # Step 3: Beta & Alpha require a market benchmark
    if market_df is not None and not market_df.empty and "Close" in market_df.columns:
        # Compute market daily returns if not already present
        if "Daily_Returns" not in market_df.columns:
            market_df = compute_daily_returns(market_df)

        market_returns = market_df["Daily_Returns"]

        beta = compute_beta(returns, market_returns)
        metrics["beta"] = beta

        metrics["alpha"] = compute_alpha(
            returns, market_returns, beta
        )

    return metrics
