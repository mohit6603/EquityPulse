"""
Data Fetcher Module — Downloads and caches stock market data from Yahoo Finance.

This module handles all data acquisition:
- Download historical OHLCV data for any NSE/BSE stock
- Cache data to avoid repeated API calls
- Validate tickers and handle errors gracefully
- Provide Nifty 50 benchmark data for Beta calculations

Key Concept (for your learning):
    OHLCV = Open, High, Low, Close, Volume
    These 5 values per day are the foundation of ALL technical analysis.
    Every indicator, every model, every chart starts here.
"""

import pandas as pd
import yfinance as yf
import streamlit as st

from src.utils import NIFTY50_STOCKS, PERIOD_OPTIONS


# ──────────────────────────────────────────────
#  Core Data Download Functions
# ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker (e.g., 'RELIANCE.NS' for NSE)
        period: Time period — '1mo', '3mo', '6mo', '1y', '2y', '5y'

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex (the trading dates)

    How it works:
        1. yfinance calls Yahoo Finance API (free, no key needed)
        2. Returns daily OHLCV data
        3. We clean it: drop NaN rows, ensure proper types
        4. @st.cache_data stores the result so clicking around
           in the dashboard doesn't re-download every time

    Example:
        >>> df = fetch_stock_data("RELIANCE.NS", "1y")
        >>> df.head()
        # Shows last 1 year of Reliance daily prices
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            return pd.DataFrame()

        # Keep only the columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Drop any rows with missing data
        df.dropna(inplace=True)

        # Ensure the index is a proper DatetimeIndex
        df.index = pd.to_datetime(df.index)
        # Remove timezone info for cleaner display
        df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multiple_stocks(symbols: list, period: str = "1y") -> dict:
    """
    Download data for multiple stocks at once.

    Used by the Portfolio Optimizer tab to get data for 
    correlation analysis and efficient frontier calculations.

    Args:
        symbols: List of ticker symbols
        period: Time period for all stocks

    Returns:
        Dictionary mapping symbol → DataFrame

    Example:
        >>> data = fetch_multiple_stocks(["RELIANCE.NS", "TCS.NS"], "1y")
        >>> data["RELIANCE.NS"].head()
    """
    result = {}
    for symbol in symbols:
        df = fetch_stock_data(symbol, period)
        if not df.empty:
            result[symbol] = df
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty50_data(period: str = "1y") -> pd.DataFrame:
    """
    Download Nifty 50 index data — used as the market benchmark.

    Why we need this:
        Beta = how much a stock moves relative to the market.
        To calculate Beta, we need the market's returns.
        Nifty 50 (^NSEI) is India's primary market benchmark.

    Returns:
        DataFrame with Nifty 50 OHLCV data
    """
    return fetch_stock_data("^NSEI", period)


# ──────────────────────────────────────────────
#  Stock Info & Validation
# ──────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def get_stock_info(symbol: str) -> dict:
    """
    Get basic information about a stock.

    Returns dict with: name, sector, industry, market cap, etc.
    Falls back gracefully if any field is missing.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "name": info.get("longName", info.get("shortName", symbol)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "INR"),
            "exchange": info.get("exchange", "NSE"),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0),
        }
    except Exception:
        return {
            "name": symbol.replace(".NS", ""),
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": 0,
            "currency": "INR",
            "exchange": "NSE",
            "52w_high": 0,
            "52w_low": 0,
            "pe_ratio": 0,
            "dividend_yield": 0,
        }


def get_ticker_list() -> dict:
    """
    Return the full dictionary of available stock tickers.
    Keys are symbols (e.g., 'RELIANCE.NS'), values are display names.
    """
    return NIFTY50_STOCKS.copy()


def validate_ticker(symbol: str) -> bool:
    """
    Check if a ticker symbol is valid by trying to fetch minimal data.

    Args:
        symbol: Stock ticker to validate

    Returns:
        True if the ticker exists and returns data, False otherwise
    """
    try:
        df = yf.Ticker(symbol).history(period="5d")
        return not df.empty
    except Exception:
        return False
