"""
Utility functions and constants for EquityPulse.
Shared helpers used across all modules.
"""

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

# Trading days in a year (Indian market)
TRADING_DAYS = 252

# Risk-free rate (approximate Indian 10-year govt bond yield)
RISK_FREE_RATE = 0.07  # 7% annual

# Nifty 50 — Top stocks with their display names
# Covers major sectors: IT, Banking, Energy, FMCG, Pharma, Auto
NIFTY50_STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Limited",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharma",
    "TATAMOTORS.NS": "Tata Motors",
    "WIPRO.NS": "Wipro",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "TITAN.NS": "Titan Company",
    "BAJFINANCE.NS": "Bajaj Finance",
    "NESTLEIND.NS": "Nestle India",
    "TATASTEEL.NS": "Tata Steel",
    "POWERGRID.NS": "Power Grid Corp",
    "NTPC.NS": "NTPC Limited",
    "HCLTECH.NS": "HCL Technologies",
    "TECHM.NS": "Tech Mahindra",
    "ONGC.NS": "ONGC",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ADANIENT.NS": "Adani Enterprises",
    "JSWSTEEL.NS": "JSW Steel",
}

# Period options for data download
PERIOD_OPTIONS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}

# Color palette for the dashboard
COLORS = {
    "green": "#00D4AA",
    "red": "#FF4B4B",
    "blue": "#4A9EFF",
    "yellow": "#FFD700",
    "purple": "#B388FF",
    "orange": "#FF8C42",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1F2E",
    "text": "#FAFAFA",
    "text_muted": "#8899A6",
}


# ──────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────

def format_currency(value: float) -> str:
    """Format a number as Indian Rupees.
    
    Examples:
        format_currency(1234567.89) → '₹12,34,567.89'
        format_currency(-500.5)    → '-₹500.50'
    """
    if value < 0:
        return f"-₹{abs(value):,.2f}"
    return f"₹{value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage with color indicator.
    
    Examples:
        format_percentage(0.0534) → '+5.34%'
        format_percentage(-0.021) → '-2.10%'
    """
    pct = value * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def format_large_number(value: float) -> str:
    """Format large numbers in Indian notation (Cr/L).
    
    Examples:
        format_large_number(15000000) → '1.50 Cr'
        format_large_number(500000)   → '5.00 L'
    """
    if abs(value) >= 1e7:
        return f"{value / 1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"{value / 1e5:.2f} L"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.2f} K"
    return f"{value:.2f}"


def get_signal_emoji(value: float, thresholds: tuple = (0, 0)) -> str:
    """Return emoji based on value and thresholds.
    
    Args:
        value: The metric value
        thresholds: (low, high) — below low = bearish, above high = bullish
    """
    low, high = thresholds
    if value > high:
        return "🟢"
    elif value < low:
        return "🔴"
    return "🟡"
