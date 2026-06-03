"""
Portfolio Optimizer Module — Modern Portfolio Theory (MPT) Implementation.

This module helps you build the BEST possible portfolio of stocks by
balancing risk and return using Harry Markowitz's Nobel-Prize-winning
framework from 1952.

Core Idea of MPT:
    "Don't put all your eggs in one basket."
    By combining stocks that don't move perfectly together (low correlation),
    you can reduce total portfolio risk WITHOUT sacrificing return.

Key Linear Algebra Concepts (explained inline):
    - Returns Matrix:    Each column is a stock's daily returns
    - Covariance Matrix: Captures how stocks move TOGETHER
    - Portfolio Variance: w^T · Σ · w  (quadratic form)
    - Efficient Frontier: Set of portfolios with the best return-for-risk

Pipeline:
    1. compute_returns_matrix()      → Daily returns for each stock
    2. compute_correlation_matrix()  → How stocks co-move (-1 to +1)
    3. compute_covariance_matrix()   → Annualised risk relationships
    4. generate_efficient_frontier() → Monte Carlo random portfolios
    5. optimize_min_variance()       → Least risky portfolio (scipy)
    6. optimize_max_sharpe()         → Best risk-adjusted portfolio
    7. compare_portfolios()          → Your portfolio vs equal-weight
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils import TRADING_DAYS, RISK_FREE_RATE


# ──────────────────────────────────────────────────────────────
#  1. Returns Matrix
# ──────────────────────────────────────────────────────────────

def compute_returns_matrix(stock_data_dict: dict) -> pd.DataFrame:
    """
    Build a matrix of daily percentage returns for multiple stocks.

    WHY daily returns (not raw prices)?
        Raw prices are on different scales (₹100 vs ₹3000).
        Returns (% change) normalise everything so we can compare
        apples to apples.

    Math:
        return_t = (price_t - price_{t-1}) / price_{t-1}
        (This is just pct_change in pandas.)

    Date Alignment:
        Different stocks may have different trading days (e.g., one
        stock halted on a day).  We use an INNER JOIN on dates so
        we only keep days where ALL stocks traded.

    Args:
        stock_data_dict: Dictionary mapping stock symbols to DataFrames.
            Example: {"RELIANCE.NS": df1, "TCS.NS": df2}
            Each DataFrame must have a 'Close' column and DatetimeIndex.

    Returns:
        DataFrame where:
            - Each COLUMN is a stock's daily returns
            - Each ROW is a trading date
            - Index is the DatetimeIndex (inner join of all stocks)

    Raises:
        ValueError: If fewer than 2 stocks are provided (can't diversify
                    with just one stock!) or if DataFrames are empty.

    Example:
        >>> returns = compute_returns_matrix({"RELIANCE.NS": df1, "TCS.NS": df2})
        >>> returns.head()
                     RELIANCE.NS   TCS.NS
        2024-01-02      0.0123    -0.0045
        2024-01-03     -0.0087     0.0156
    """
    if not stock_data_dict:
        raise ValueError("stock_data_dict is empty. Provide at least 2 stocks.")

    if len(stock_data_dict) < 2:
        raise ValueError(
            "Need at least 2 stocks to build a portfolio. "
            f"Got {len(stock_data_dict)}."
        )

    # Extract Close prices into a single DataFrame (one column per stock).
    close_prices = pd.DataFrame()
    for symbol, df in stock_data_dict.items():
        if df is None or df.empty:
            continue
        if "Close" not in df.columns:
            continue
        close_prices[symbol] = df["Close"]

    if close_prices.shape[1] < 2:
        raise ValueError(
            "After filtering, fewer than 2 stocks have valid Close data."
        )

    # Inner join: keep only dates where ALL stocks have data.
    # This prevents NaN issues in covariance/correlation calculations.
    close_prices.dropna(how="any", inplace=True)

    if close_prices.empty:
        raise ValueError(
            "No overlapping trading dates found across the provided stocks."
        )

    # Compute daily returns using pct_change.
    # The first row becomes NaN (no previous day), so we drop it.
    returns_df = close_prices.pct_change().dropna()

    return returns_df


# ──────────────────────────────────────────────────────────────
#  2. Correlation Matrix
# ──────────────────────────────────────────────────────────────

def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the pairwise correlation matrix of stock returns.

    WHAT is correlation?
        Correlation measures how two stocks move TOGETHER:
            +1.0 = perfectly in sync (both go up/down together)
             0.0 = no relationship
            -1.0 = perfectly opposite (one up, other down)

    WHY does this matter for portfolios?
        If two stocks have LOW correlation, holding both REDUCES your
        overall risk.  This is the magic of diversification!

        Example:
            - Banking stocks are highly correlated (~0.8) with each other
            - IT stocks + FMCG stocks might have low correlation (~0.3)
            → A portfolio mixing IT + FMCG is safer than all-banking.

    Math:
        corr(A, B) = cov(A, B) / (std(A) * std(B))
        Pandas' .corr() does this for every pair automatically.

    Args:
        returns_df: DataFrame of daily returns (from compute_returns_matrix).
                    Each column is a stock, each row is a date.

    Returns:
        DataFrame (N×N) where element (i,j) is the correlation between
        stock i and stock j.  Diagonal is always 1.0 (stock vs itself).

    Raises:
        ValueError: If returns_df is empty or has fewer than 2 columns.

    Example:
        >>> corr = compute_correlation_matrix(returns)
        >>> corr.loc["RELIANCE.NS", "TCS.NS"]
        0.42  # Moderate positive correlation
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("Cannot compute correlation on empty returns data.")

    if returns_df.shape[1] < 2:
        raise ValueError("Need at least 2 stocks to compute correlations.")

    return returns_df.corr()


# ──────────────────────────────────────────────────────────────
#  3. Covariance Matrix
# ──────────────────────────────────────────────────────────────

def compute_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the ANNUALISED covariance matrix of stock returns.

    WHAT is a covariance matrix?
        ┌──────────────────────────────────────────────────────────┐
        │  This is LINEAR ALGEBRA — the covariance matrix captures │
        │  how stocks move together (direction AND magnitude).     │
        └──────────────────────────────────────────────────────────┘

        Unlike correlation (which is normalised to [-1, +1]),
        covariance preserves the SCALE of movements:
            cov(A, B) = E[(A - μ_A)(B - μ_B)]

        The covariance matrix Σ is an N×N symmetric matrix where:
            - Diagonal elements = variance of each stock (risk²)
            - Off-diagonal elements = covariance between stock pairs

    WHY annualise?
        Daily covariance is tiny (0.0001 scale).  Multiplying by
        TRADING_DAYS (252) converts to annual scale, matching how
        we think about returns (e.g., "12% annual return").

        Formula:  Σ_annual = Σ_daily × 252

    KEY PROPERTY:
        Σ is always symmetric and positive semi-definite. This is
        important because portfolio variance = w^T · Σ · w must
        always be ≥ 0 (you can't have negative risk).

    Args:
        returns_df: DataFrame of daily returns (from compute_returns_matrix).

    Returns:
        DataFrame (N×N) — annualised covariance matrix.

    Raises:
        ValueError: If returns_df is empty or has fewer than 2 columns.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("Cannot compute covariance on empty returns data.")

    if returns_df.shape[1] < 2:
        raise ValueError("Need at least 2 stocks to compute covariance.")

    # .cov() gives the DAILY sample covariance matrix.
    # Multiply by TRADING_DAYS to annualise.
    # This scaling works because returns are roughly independently
    # distributed across days (variance scales linearly with time).
    cov_matrix = returns_df.cov() * TRADING_DAYS

    return cov_matrix


# ──────────────────────────────────────────────────────────────
#  4. Efficient Frontier (Monte Carlo Simulation)
# ──────────────────────────────────────────────────────────────

def generate_efficient_frontier(returns_df: pd.DataFrame,
                                num_portfolios: int = 5000) -> pd.DataFrame:
    """
    Generate random portfolios to approximate the Efficient Frontier.

    WHAT is the Efficient Frontier?
        Imagine plotting thousands of random portfolios on a chart:
            X-axis = Risk (volatility / standard deviation)
            Y-axis = Expected Return

        The UPPER-LEFT boundary of this cloud is the "Efficient Frontier"
        — portfolios where you get the MAXIMUM return for each level
        of risk.  Any portfolio NOT on this line is suboptimal.

    HOW (Monte Carlo method):
        For each of `num_portfolios` random weight combinations:
        1. Generate random weights that sum to 1 (long-only, no shorting)
        2. Compute expected annual return:
               E[R] = w · μ  (dot product of weights and mean returns)
        3. Compute annual volatility (risk):
               σ = √(w^T · Σ · w)  ← this is the KEY linear algebra step!
           Where:
               w = column vector of weights  (N × 1)
               Σ = covariance matrix          (N × N)
               w^T · Σ · w = scalar (the portfolio variance)
        4. Compute Sharpe Ratio:
               S = (E[R] - R_f) / σ
           Where R_f = risk-free rate (govt bond yield)

    Args:
        returns_df:     DataFrame of daily returns (from compute_returns_matrix).
        num_portfolios: Number of random portfolios to generate (default 5000).
                        More = smoother frontier but slower.

    Returns:
        DataFrame with columns:
            - 'Return':     Expected annual return
            - 'Volatility': Annual standard deviation (risk)
            - 'Sharpe':     Sharpe ratio (risk-adjusted return)
            - One column per stock symbol (the weight of that stock)

    Raises:
        ValueError: If returns_df is empty or has fewer than 2 columns.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("Cannot generate frontier from empty returns data.")

    if returns_df.shape[1] < 2:
        raise ValueError("Need at least 2 stocks for portfolio optimisation.")

    num_stocks = returns_df.shape[1]
    stock_symbols = list(returns_df.columns)

    # Annualised mean returns per stock.
    # E.g., if daily mean return is 0.05%, annual ≈ 0.05% × 252 ≈ 12.6%.
    mean_returns = returns_df.mean() * TRADING_DAYS

    # Annualised covariance matrix (N × N).
    cov_matrix = returns_df.cov() * TRADING_DAYS

    # Pre-allocate arrays for speed (avoid appending to lists).
    all_returns = np.zeros(num_portfolios)
    all_volatilities = np.zeros(num_portfolios)
    all_sharpes = np.zeros(num_portfolios)
    all_weights = np.zeros((num_portfolios, num_stocks))

    # Set random seed for reproducibility.
    rng = np.random.default_rng(seed=42)

    for i in range(num_portfolios):
        # ── Step 1: Generate random weights ──
        # Draw from uniform distribution, then normalise so they sum to 1.
        # All weights ≥ 0 → "long only" (no short selling).
        raw_weights = rng.random(num_stocks)
        weights = raw_weights / raw_weights.sum()

        # ── Step 2: Expected portfolio return ──
        # This is a simple dot product: w · μ
        # If weight = [0.5, 0.5] and returns = [10%, 20%],
        # portfolio return = 0.5*10% + 0.5*20% = 15%.
        port_return = np.dot(weights, mean_returns)

        # ── Step 3: Portfolio volatility (risk) ──
        # This is the KEY formula from Modern Portfolio Theory:
        #     σ_p = √(w^T · Σ · w)
        #
        # Breaking it down:
        #     Σ · w   → N×1 vector (covariance-weighted exposure)
        #     w^T · (Σ · w) → scalar (portfolio variance)
        #     √(...)  → standard deviation (volatility)
        #
        # This is LESS than the weighted average of individual volatilities
        # when stocks aren't perfectly correlated — that's diversification!
        port_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        port_volatility = np.sqrt(port_variance)

        # ── Step 4: Sharpe Ratio ──
        # Sharpe = excess return per unit of risk.
        # Higher Sharpe = better risk-adjusted performance.
        # A Sharpe of 1.0+ is generally considered good.
        if port_volatility > 0:
            port_sharpe = (port_return - RISK_FREE_RATE) / port_volatility
        else:
            port_sharpe = 0.0

        # Store results.
        all_returns[i] = port_return
        all_volatilities[i] = port_volatility
        all_sharpes[i] = port_sharpe
        all_weights[i, :] = weights

    # Build results DataFrame.
    results = pd.DataFrame({
        "Return": all_returns,
        "Volatility": all_volatilities,
        "Sharpe": all_sharpes,
    })

    # Add weight columns (one per stock).
    for j, symbol in enumerate(stock_symbols):
        results[symbol] = all_weights[:, j]

    return results


# ──────────────────────────────────────────────────────────────
#  Helper: Portfolio Performance Calculator
# ──────────────────────────────────────────────────────────────

def _portfolio_performance(weights: np.ndarray,
                           mean_returns: pd.Series,
                           cov_matrix: pd.DataFrame) -> tuple:
    """
    Compute expected return, volatility, and Sharpe ratio for given weights.

    This is the core calculation shared by both optimisers.

    MATH:
        Return     = w · μ                    (dot product)
        Volatility = √(w^T · Σ · w)           (quadratic form)
        Sharpe     = (Return - R_f) / Vol      (risk-adjusted return)

    Args:
        weights:      1D array of portfolio weights (must sum to 1).
        mean_returns: Series of annualised mean returns per stock.
        cov_matrix:   Annualised covariance matrix DataFrame.

    Returns:
        Tuple of (expected_return, volatility, sharpe_ratio).
    """
    # Portfolio expected return: dot product of weights and mean returns.
    port_return = np.dot(weights, mean_returns)

    # Portfolio variance: quadratic form  w^T · Σ · w
    port_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    port_volatility = np.sqrt(port_variance)

    # Sharpe ratio: excess return per unit of risk.
    if port_volatility > 0:
        sharpe = (port_return - RISK_FREE_RATE) / port_volatility
    else:
        sharpe = 0.0

    return port_return, port_volatility, sharpe


# ──────────────────────────────────────────────────────────────
#  5. Minimum Variance Portfolio
# ──────────────────────────────────────────────────────────────

def optimize_min_variance(returns_df: pd.DataFrame) -> dict:
    """
    Find the portfolio with the LOWEST possible risk (minimum volatility).

    WHY minimum variance?
        This is the "safest" portfolio — it minimises the total
        standard deviation (risk).  It's ideal for conservative investors
        who want to reduce drawdowns.

    HOW (Mathematical Optimisation):
        We use scipy.optimize.minimize with the SLSQP algorithm to solve:

            minimise:  σ_p = √(w^T · Σ · w)
            subject to:
                - Σ w_i = 1     (weights must sum to 1 = fully invested)
                - w_i ≥ 0       (no short selling = long only)

        SLSQP = Sequential Least Squares Quadratic Programming.
        It's efficient for this type of constrained optimisation because
        the objective function is smooth and the constraints are linear.

    Args:
        returns_df: DataFrame of daily returns (from compute_returns_matrix).

    Returns:
        Dictionary with keys:
            - 'weights':  dict mapping stock symbol → optimal weight
            - 'return':   expected annual return (float)
            - 'volatility': annual volatility / risk (float)
            - 'sharpe':   Sharpe ratio (float)

    Raises:
        ValueError: If returns_df is empty or has fewer than 2 stocks.
        RuntimeError: If the optimiser fails to converge.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("Cannot optimise on empty returns data.")

    if returns_df.shape[1] < 2:
        raise ValueError("Need at least 2 stocks for optimisation.")

    num_stocks = returns_df.shape[1]
    stock_symbols = list(returns_df.columns)

    mean_returns = returns_df.mean() * TRADING_DAYS
    cov_matrix = returns_df.cov() * TRADING_DAYS

    # ── Objective: minimise portfolio volatility ──
    def objective(weights):
        """Return portfolio volatility (what we want to minimise)."""
        port_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        return np.sqrt(port_variance)

    # ── Constraints ──
    # 1. All weights must sum to exactly 1 (fully invested).
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )

    # 2. Each weight must be between 0 and 1 (no shorting, no leverage).
    bounds = tuple((0.0, 1.0) for _ in range(num_stocks))

    # ── Initial guess: equal weights ──
    initial_weights = np.array([1.0 / num_stocks] * num_stocks)

    # ── Run the optimiser ──
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(
            f"Min-variance optimisation failed: {result.message}"
        )

    optimal_weights = result.x

    # Compute final portfolio metrics with the optimal weights.
    port_return, port_vol, port_sharpe = _portfolio_performance(
        optimal_weights, mean_returns, cov_matrix
    )

    return {
        "weights": dict(zip(stock_symbols, optimal_weights)),
        "return": float(port_return),
        "volatility": float(port_vol),
        "sharpe": float(port_sharpe),
    }


# ──────────────────────────────────────────────────────────────
#  6. Maximum Sharpe Ratio Portfolio
# ──────────────────────────────────────────────────────────────

def optimize_max_sharpe(returns_df: pd.DataFrame) -> dict:
    """
    Find the portfolio with the HIGHEST risk-adjusted return (max Sharpe).

    WHY maximum Sharpe?
        The Sharpe ratio tells you how much EXTRA return you get for
        each unit of risk you take.  Maximising Sharpe finds the
        portfolio on the Efficient Frontier that a rational investor
        would prefer (assuming they can borrow/lend at the risk-free rate).

        Sharpe = (R_p - R_f) / σ_p
        Where:
            R_p = portfolio return
            R_f = risk-free rate (govt bond yield, ~7% in India)
            σ_p = portfolio volatility

    HOW:
        Since scipy.optimize.minimize MINIMISES, we minimise the
        NEGATIVE Sharpe ratio (which is equivalent to maximising Sharpe).

            minimise:  -(R_p - R_f) / σ_p
            subject to:
                - Σ w_i = 1
                - w_i ≥ 0

    Args:
        returns_df: DataFrame of daily returns (from compute_returns_matrix).

    Returns:
        Dictionary with keys:
            - 'weights':    dict mapping stock symbol → optimal weight
            - 'return':     expected annual return (float)
            - 'volatility': annual volatility / risk (float)
            - 'sharpe':     Sharpe ratio (float)

    Raises:
        ValueError: If returns_df is empty or has fewer than 2 stocks.
        RuntimeError: If the optimiser fails to converge.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("Cannot optimise on empty returns data.")

    if returns_df.shape[1] < 2:
        raise ValueError("Need at least 2 stocks for optimisation.")

    num_stocks = returns_df.shape[1]
    stock_symbols = list(returns_df.columns)

    mean_returns = returns_df.mean() * TRADING_DAYS
    cov_matrix = returns_df.cov() * TRADING_DAYS

    # ── Objective: minimise NEGATIVE Sharpe (= maximise Sharpe) ──
    def neg_sharpe(weights):
        """Return negative Sharpe ratio (minimising this = maximising Sharpe)."""
        port_return = np.dot(weights, mean_returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        port_volatility = np.sqrt(port_variance)

        if port_volatility == 0:
            return 0.0  # Avoid division by zero

        return -(port_return - RISK_FREE_RATE) / port_volatility

    # ── Constraints & bounds (same as min-variance) ──
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    bounds = tuple((0.0, 1.0) for _ in range(num_stocks))
    initial_weights = np.array([1.0 / num_stocks] * num_stocks)

    # ── Run the optimiser ──
    result = minimize(
        neg_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(
            f"Max-Sharpe optimisation failed: {result.message}"
        )

    optimal_weights = result.x

    # Compute final portfolio metrics.
    port_return, port_vol, port_sharpe = _portfolio_performance(
        optimal_weights, mean_returns, cov_matrix
    )

    return {
        "weights": dict(zip(stock_symbols, optimal_weights)),
        "return": float(port_return),
        "volatility": float(port_vol),
        "sharpe": float(port_sharpe),
    }


# ──────────────────────────────────────────────────────────────
#  7. Portfolio Comparison
# ──────────────────────────────────────────────────────────────

def compare_portfolios(returns_df: pd.DataFrame,
                       weights_dict: dict) -> pd.DataFrame:
    """
    Compare a custom-weighted portfolio against an equal-weight portfolio.

    WHY compare against equal-weight?
        Equal-weight (1/N) is the simplest possible strategy — give
        every stock the same weight.  Surprisingly, it's a tough
        benchmark to beat!  Comparing against it shows whether your
        optimisation actually adds value.

    HOW cumulative returns work:
        Day 1 return = +2%  →  cumulative = 1.02
        Day 2 return = -1%  →  cumulative = 1.02 × 0.99 = 1.0098
        Formula: cum_return = Π(1 + r_t)  (product of 1+daily returns)

        If cumulative = 1.15, you made 15% total.
        If cumulative = 0.92, you lost 8%.

    Args:
        returns_df:   DataFrame of daily returns (from compute_returns_matrix).
        weights_dict: Dictionary mapping stock symbol → weight (float).
                      Weights should sum to 1.0.
                      Example: {"RELIANCE.NS": 0.4, "TCS.NS": 0.6}

    Returns:
        DataFrame with columns:
            - 'Custom':      Cumulative return of the custom portfolio
            - 'Equal_Weight': Cumulative return of the equal-weight portfolio
            - DatetimeIndex matching the returns data

    Raises:
        ValueError: If returns_df is empty, or weights_dict doesn't match
                    the stocks in returns_df.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("Cannot compare portfolios with empty returns data.")

    if not weights_dict:
        raise ValueError("weights_dict is empty. Provide stock weights.")

    stock_symbols = list(returns_df.columns)
    num_stocks = len(stock_symbols)

    # ── Build the custom weight vector ──
    # Ensure the weights align with the columns of returns_df.
    custom_weights = np.array([
        weights_dict.get(symbol, 0.0) for symbol in stock_symbols
    ])

    # Normalise weights to sum to 1 (in case they don't already).
    weight_sum = custom_weights.sum()
    if weight_sum == 0:
        raise ValueError(
            "All custom weights are zero. At least one stock must have "
            "a positive weight."
        )
    custom_weights = custom_weights / weight_sum

    # ── Equal weight vector ──
    equal_weights = np.array([1.0 / num_stocks] * num_stocks)

    # ── Compute daily portfolio returns ──
    # Portfolio daily return = Σ (weight_i × return_i) = w · r_t
    custom_daily = returns_df.values @ custom_weights
    equal_daily = returns_df.values @ equal_weights

    # ── Compute cumulative returns ──
    # cumprod of (1 + daily_return) gives the growth of ₹1 invested.
    custom_cumulative = (1 + custom_daily).cumprod()
    equal_cumulative = (1 + equal_daily).cumprod()

    comparison = pd.DataFrame(
        {
            "Custom": custom_cumulative,
            "Equal_Weight": equal_cumulative,
        },
        index=returns_df.index,
    )

    return comparison
