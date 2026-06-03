"""
EquityPulse — AI-Powered Stock Analytics Dashboard
===================================================
Author: Mohit Patle
Description: A comprehensive stock analysis tool featuring technical indicators,
             risk metrics, ML-based direction prediction with SHAP explainability,
             and Markowitz portfolio optimization.

Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
#  Page Configuration (MUST be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="EquityPulse | AI Stock Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  Imports from our modules
# ──────────────────────────────────────────────
from src.data_fetcher import (
    fetch_stock_data,
    fetch_nifty50_data,
    fetch_multiple_stocks,
    get_stock_info,
    get_ticker_list,
)
from src.technical_indicators import add_all_indicators, get_signal_summary
from src.risk_metrics import compute_daily_returns, compute_all_metrics
from src.ml_predictor import (
    engineer_features,
    train_model,
    predict_direction,
    explain_prediction,
    get_feature_importance,
)
from src.portfolio_optimizer import (
    compute_returns_matrix,
    compute_correlation_matrix,
    generate_efficient_frontier,
    optimize_max_sharpe,
    optimize_min_variance,
)
from src.utils import (
    NIFTY50_STOCKS,
    PERIOD_OPTIONS,
    COLORS,
    format_currency,
    format_percentage,
    format_large_number,
    get_signal_emoji,
)


# ──────────────────────────────────────────────
#  Custom CSS for Premium Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 212, 170, 0.15);
        text-align: center;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #00D4AA, #4A9EFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #8899A6;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: #1A1F2E;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card .label {
        color: #8899A6;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 600;
    }
    .metric-card .value.green { color: #00D4AA; }
    .metric-card .value.red { color: #FF4B4B; }
    .metric-card .value.blue { color: #4A9EFF; }
    .metric-card .value.yellow { color: #FFD700; }

    /* Signal badge */
    .signal-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.3px;
    }
    .signal-bullish {
        background: rgba(0, 212, 170, 0.15);
        color: #00D4AA;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    .signal-bearish {
        background: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    .signal-neutral {
        background: rgba(255, 215, 0, 0.12);
        color: #FFD700;
        border: 1px solid rgba(255, 215, 0, 0.25);
    }

    /* Section dividers */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,170,0.25), transparent);
        margin: 1.5rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 EquityPulse</h1>
    <p>AI-Powered Stock Analytics • Technical Analysis • ML Prediction • Portfolio Optimization</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Sidebar Controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # Stock selection
    tickers = get_ticker_list()
    ticker_options = list(tickers.keys())
    ticker_labels = [f"{tickers[t]}  ({t.replace('.NS', '')})" for t in ticker_options]

    selected_index = st.selectbox(
        "📈 Select Stock",
        range(len(ticker_options)),
        format_func=lambda i: ticker_labels[i],
        index=0,
    )
    selected_ticker = ticker_options[selected_index]

    # Custom ticker option
    use_custom = st.checkbox("Use custom ticker")
    if use_custom:
        custom_ticker = st.text_input(
            "Enter ticker (e.g., ZOMATO.NS)",
            placeholder="SYMBOL.NS"
        )
        if custom_ticker:
            selected_ticker = custom_ticker.upper()
            if not selected_ticker.endswith(".NS") and not selected_ticker.endswith(".BO"):
                selected_ticker += ".NS"

    # Period selection
    selected_period_label = st.selectbox(
        "📅 Time Period",
        list(PERIOD_OPTIONS.keys()),
        index=3,  # Default: 1 Year
    )
    selected_period = PERIOD_OPTIONS[selected_period_label]

    st.markdown("---")
    st.markdown(
        "<p style='color: #8899A6; font-size: 0.75rem; text-align: center;'>"
        "Built by Mohit Patle<br>Data from Yahoo Finance"
        "</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
#  Load Data
# ──────────────────────────────────────────────
with st.spinner(f"Fetching data for {selected_ticker}..."):
    stock_df = fetch_stock_data(selected_ticker, selected_period)

if stock_df.empty:
    st.error(f"❌ Could not fetch data for **{selected_ticker}**. Please check the ticker symbol.")
    st.stop()

# Add indicators and returns
stock_df = add_all_indicators(stock_df.copy())
stock_df = compute_daily_returns(stock_df)

# Get stock info
info = get_stock_info(selected_ticker)

# Fetch Nifty 50 for Beta calculation
market_df = fetch_nifty50_data(selected_period)
if not market_df.empty:
    market_df = compute_daily_returns(market_df)


# ──────────────────────────────────────────────
#  Stock Info Bar
# ──────────────────────────────────────────────
latest_close = stock_df["Close"].iloc[-1]
prev_close = stock_df["Close"].iloc[-2] if len(stock_df) > 1 else latest_close
daily_change = (latest_close - prev_close) / prev_close
change_color = "green" if daily_change >= 0 else "red"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Current Price</div>
        <div class="value {change_color}">{format_currency(latest_close)}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    sign = "+" if daily_change >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Daily Change</div>
        <div class="value {change_color}">{sign}{daily_change*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Sector</div>
        <div class="value blue">{info.get('sector', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    mcap = info.get("market_cap", 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Market Cap</div>
        <div class="value yellow">{format_large_number(mcap) if mcap else 'N/A'}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Main Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Stock Analysis",
    "🤖 ML Prediction",
    "💼 Portfolio Optimizer",
    "ℹ️ About",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1: STOCK ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab1:
    st.markdown("### 📈 Technical Analysis")

    # ── Candlestick Chart with Overlays ──
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.20, 0.25],
        vertical_spacing=0.04,
        subplot_titles=("Price & Indicators", "Volume", "RSI"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=stock_df.index,
            open=stock_df["Open"],
            high=stock_df["High"],
            low=stock_df["Low"],
            close=stock_df["Close"],
            name="Price",
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
        ),
        row=1, col=1,
    )

    # SMA lines
    if "SMA_20" in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df.index, y=stock_df["SMA_20"],
                name="SMA 20", line=dict(color=COLORS["blue"], width=1.2),
            ),
            row=1, col=1,
        )
    if "SMA_50" in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df.index, y=stock_df["SMA_50"],
                name="SMA 50", line=dict(color=COLORS["orange"], width=1.2),
            ),
            row=1, col=1,
        )

    # Bollinger Bands
    if "BB_Upper" in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df.index, y=stock_df["BB_Upper"],
                name="BB Upper", line=dict(color=COLORS["purple"], width=0.8, dash="dot"),
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=stock_df.index, y=stock_df["BB_Lower"],
                name="BB Lower", line=dict(color=COLORS["purple"], width=0.8, dash="dot"),
                fill="tonexty", fillcolor="rgba(179,136,255,0.06)",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Volume bars
    vol_colors = [
        COLORS["green"] if stock_df["Close"].iloc[i] >= stock_df["Open"].iloc[i]
        else COLORS["red"]
        for i in range(len(stock_df))
    ]
    fig.add_trace(
        go.Bar(
            x=stock_df.index, y=stock_df["Volume"],
            name="Volume", marker_color=vol_colors, opacity=0.5,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # RSI
    if "RSI_14" in stock_df.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_df.index, y=stock_df["RSI_14"],
                name="RSI(14)", line=dict(color=COLORS["blue"], width=1.5),
            ),
            row=3, col=1,
        )
        # Overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"],
                      opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"],
                      opacity=0.5, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        height=700,
        margin=dict(l=50, r=30, t=40, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        xaxis_rangeslider_visible=False,
        font=dict(family="Inter"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")

    st.plotly_chart(fig, width="stretch")

    # ── Signal Summary ──
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🚦 Signal Summary")

    signals = get_signal_summary(stock_df)

    sig_cols = st.columns(5)
    rsi_val = stock_df["RSI_14"].iloc[-1] if "RSI_14" in stock_df.columns else None
    signal_items = [
        ("RSI", signals.get("rsi_status", "N/A"), rsi_val),
        ("MACD", signals.get("macd_status", "N/A"), None),
        ("SMA Trend", signals.get("sma_trend", "N/A"), None),
        ("Bollinger", signals.get("bollinger_position", "N/A"), None),
        ("Overall", signals.get("overall_signal", "N/A"), None),
    ]

    for col, (label, status, val) in zip(sig_cols, signal_items):
        with col:
            # Classify signal for styling
            bullish_signals = ["Bullish", "Bullish Crossover", "Golden Cross", "Oversold", "Near Lower Band"]
            bearish_signals = ["Bearish", "Bearish Crossover", "Death Cross", "Overbought", "Near Upper Band"]
            if status in bullish_signals:
                css_class = "signal-bullish"
            elif status in bearish_signals:
                css_class = "signal-bearish"
            else:
                css_class = "signal-neutral"

            val_text = f" ({val:.1f})" if val is not None else ""
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="label">{label}</div>
                <span class="signal-badge {css_class}">{status}{val_text}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Risk Metrics ──
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk Metrics")

    market_data_for_metrics = market_df if not market_df.empty else None
    metrics = compute_all_metrics(stock_df, market_data_for_metrics)

    # Add extra metrics that compute_all_metrics doesn't include
    returns_series = stock_df["Daily_Returns"].dropna()
    if len(returns_series) > 0:
        metrics["avg_daily_return"] = float(returns_series.mean())
        metrics["total_return"] = float(
            (stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[0]) - 1
        )
        metrics["annualized_volatility"] = metrics.get("volatility", 0)
    else:
        metrics["avg_daily_return"] = 0
        metrics["total_return"] = 0
        metrics["annualized_volatility"] = 0

    m_cols = st.columns(4)
    metric_items = [
        ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}", "blue",
         "Risk-adjusted return. >1 = good, >2 = great"),
        ("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%", "red",
         "Worst peak-to-trough decline"),
        ("VaR (95%)", f"{metrics.get('var_95', 0)*100:.2f}%", "yellow",
         "Max expected daily loss (95% confidence)"),
        ("Volatility", f"{metrics.get('annualized_volatility', 0)*100:.1f}%", "blue",
         "Annualized price fluctuation"),
    ]

    for col, (label, value, color, tooltip) in zip(m_cols, metric_items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value {color}">{value}</div>
                <div style="color:#8899A6; font-size:0.7rem; margin-top:0.3rem;">{tooltip}</div>
            </div>
            """, unsafe_allow_html=True)

    # Second row of metrics
    m_cols2 = st.columns(4)
    beta_val = metrics.get('beta', np.nan)
    alpha_val = metrics.get('alpha', np.nan)
    beta_str = f"{beta_val:.2f}" if not np.isnan(beta_val) else "N/A"
    alpha_str = f"{alpha_val*100:.2f}%" if not np.isnan(alpha_val) else "N/A"
    metric_items2 = [
        ("Beta", beta_str, "blue",
         ">1 = more volatile than market"),
        ("Alpha (Annual)", alpha_str, "green",
         "Excess return over market"),
        ("Avg Daily Return", format_percentage(metrics.get("avg_daily_return", 0)), "green",
         "Mean daily return"),
        ("Total Return", format_percentage(metrics.get("total_return", 0)), "green",
         "Overall period return"),
    ]
    for col, (label, value, color, tooltip) in zip(m_cols2, metric_items2):
        with col:
            val_display = value if isinstance(value, str) else f"{value:.4f}"
            actual_color = color
            if isinstance(value, str) and value.startswith("-"):
                actual_color = "red"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value {actual_color}">{val_display}</div>
                <div style="color:#8899A6; font-size:0.7rem; margin-top:0.3rem;">{tooltip}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── MACD Chart ──
    if "MACD" in stock_df.columns:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 📉 MACD (Moving Average Convergence Divergence)")

        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=stock_df.index, y=stock_df["MACD"],
            name="MACD Line", line=dict(color=COLORS["blue"], width=1.5),
        ))
        macd_fig.add_trace(go.Scatter(
            x=stock_df.index, y=stock_df["MACD_Signal"],
            name="Signal Line", line=dict(color=COLORS["orange"], width=1.5),
        ))
        hist_colors = [
            COLORS["green"] if v >= 0 else COLORS["red"]
            for v in stock_df["MACD_Histogram"]
        ]
        macd_fig.add_trace(go.Bar(
            x=stock_df.index, y=stock_df["MACD_Histogram"],
            name="Histogram", marker_color=hist_colors, opacity=0.6,
        ))
        macd_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["bg_dark"],
            plot_bgcolor=COLORS["bg_dark"],
            height=300,
            margin=dict(l=50, r=30, t=20, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter"),
        )
        macd_fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
        macd_fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(macd_fig, width="stretch")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2: ML PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    st.markdown("### 🤖 ML-Based Direction Prediction")
    st.markdown(
        "<p style='color:#8899A6;'>Using XGBoost with 15 engineered features and walk-forward validation.</p>",
        unsafe_allow_html=True,
    )

    # Need enough data for ML
    if len(stock_df) < 100:
        st.warning("⚠️ Need at least 100 data points for ML prediction. Please select a longer time period (1Y+).")
    else:
        with st.spinner("🧠 Training model & generating predictions... This may take a moment."):
            try:
                # Fetch fresh data with enough history for training
                ml_df = fetch_stock_data(selected_ticker, "5y")
                if len(ml_df) < 200:
                    ml_df = fetch_stock_data(selected_ticker, "2y")

                ml_df = add_all_indicators(ml_df.copy())
                ml_df = compute_daily_returns(ml_df)

                # Engineer features
                features, target, feature_names = engineer_features(ml_df)

                if len(features) < 100:
                    st.warning("⚠️ Insufficient data after feature engineering. Try a different stock.")
                else:
                    # Train model
                    model, metrics, feat_names = train_model(features, target)

                    # Get latest prediction
                    latest_features = features.iloc[[-1]]
                    direction, confidence = predict_direction(model, latest_features)

                    # ── Prediction Display ──
                    pred_cols = st.columns([1, 1, 1])
                    with pred_cols[0]:
                        dir_color = "green" if direction == "UP" else "red"
                        dir_emoji = "📈" if direction == "UP" else "📉"
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center; padding:1.5rem;">
                            <div class="label">Tomorrow's Predicted Direction</div>
                            <div class="value {dir_color}" style="font-size:2rem;">
                                {dir_emoji} {direction}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with pred_cols[1]:
                        conf_color = "green" if confidence > 0.6 else "yellow" if confidence > 0.5 else "red"
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center; padding:1.5rem;">
                            <div class="label">Model Confidence</div>
                            <div class="value {conf_color}" style="font-size:2rem;">
                                {confidence*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with pred_cols[2]:
                        avg_acc = metrics.get("accuracy", {}).get("mean", 0)
                        acc_color = "green" if avg_acc > 0.55 else "yellow"
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center; padding:1.5rem;">
                            <div class="label">Walk-Forward Accuracy</div>
                            <div class="value {acc_color}" style="font-size:2rem;">
                                {avg_acc*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Model Performance Metrics ──
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("### 📊 Model Performance (Walk-Forward Validation)")

                    perf_cols = st.columns(4)
                    n_folds = len(metrics.get("accuracy", {}).get("folds", []))
                    perf_metrics = [
                        ("Precision", metrics.get("precision", {}).get("mean", 0)),
                        ("Recall", metrics.get("recall", {}).get("mean", 0)),
                        ("F1 Score", metrics.get("f1", {}).get("mean", 0)),
                        ("Folds Used", n_folds),
                    ]
                    for col, (label, val) in zip(perf_cols, perf_metrics):
                        with col:
                            if isinstance(val, float):
                                display_val = f"{val*100:.1f}%"
                            else:
                                display_val = str(val)
                            st.markdown(f"""
                            <div class="metric-card" style="text-align:center;">
                                <div class="label">{label}</div>
                                <div class="value blue">{display_val}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # ── Feature Importance ──
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("### 🏆 Feature Importance (What drives predictions?)")

                    importances = get_feature_importance(model, feat_names)
                    imp_df = pd.DataFrame(importances, columns=["Feature", "Importance"])
                    imp_df = imp_df.head(10)  # Top 10

                    imp_fig = go.Figure(go.Bar(
                        x=imp_df["Importance"],
                        y=imp_df["Feature"],
                        orientation="h",
                        marker_color=COLORS["green"],
                        marker=dict(
                            line=dict(color=COLORS["green"], width=0.5),
                        ),
                        opacity=0.85,
                    ))
                    imp_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor=COLORS["bg_dark"],
                        plot_bgcolor=COLORS["bg_dark"],
                        height=400,
                        margin=dict(l=120, r=30, t=20, b=30),
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="Importance Score",
                        font=dict(family="Inter"),
                    )
                    imp_fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
                    imp_fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
                    st.plotly_chart(imp_fig, width="stretch")

                    # ── SHAP Explanation ──
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("### 🔍 SHAP — Why this prediction?")
                    st.markdown(
                        "<p style='color:#8899A6;'>SHAP values show how each feature pushed the prediction "
                        "toward UP (positive) or DOWN (negative).</p>",
                        unsafe_allow_html=True,
                    )

                    try:
                        shap_values, expected_value, shap_feat_names = explain_prediction(
                            model, features, feat_names
                        )

                        if shap_values is not None:
                            # Get SHAP values for the latest prediction
                            latest_shap = shap_values[-1] if len(shap_values.shape) > 1 else shap_values
                            latest_feat_values = features.iloc[-1]

                            # Create waterfall-style bar chart with SHAP values
                            shap_df = pd.DataFrame({
                                "Feature": shap_feat_names,
                                "SHAP Value": latest_shap,
                                "Feature Value": [latest_feat_values[f] for f in shap_feat_names],
                            })
                            shap_df["Abs_SHAP"] = shap_df["SHAP Value"].abs()
                            shap_df = shap_df.nlargest(10, "Abs_SHAP")
                            shap_df = shap_df.sort_values("SHAP Value")

                            shap_colors = [
                                COLORS["green"] if v > 0 else COLORS["red"]
                                for v in shap_df["SHAP Value"]
                            ]

                            shap_fig = go.Figure(go.Bar(
                                x=shap_df["SHAP Value"],
                                y=[f"{row['Feature']} = {row['Feature Value']:.2f}"
                                   for _, row in shap_df.iterrows()],
                                orientation="h",
                                marker_color=shap_colors,
                                opacity=0.85,
                            ))
                            shap_fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor=COLORS["bg_dark"],
                                plot_bgcolor=COLORS["bg_dark"],
                                height=400,
                                margin=dict(l=180, r=30, t=20, b=30),
                                xaxis_title="SHAP Value (impact on prediction)",
                                font=dict(family="Inter"),
                            )
                            shap_fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
                            shap_fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
                            st.plotly_chart(shap_fig, width="stretch")

                            st.info(
                                "🟢 **Green bars** push the prediction toward **UP**.  \n"
                                "🔴 **Red bars** push the prediction toward **DOWN**.  \n"
                                "The longer the bar, the stronger the influence."
                            )
                    except Exception as e:
                        st.warning(f"SHAP explanation could not be generated: {str(e)}")

                    # Disclaimer
                    st.markdown(
                        "<div style='background:#1A1F2E; padding:1rem; border-radius:8px; "
                        "border-left:3px solid #FFD700; margin-top:1rem;'>"
                        "<strong style='color:#FFD700;'>⚠️ Disclaimer:</strong> "
                        "<span style='color:#8899A6;'>This is a learning project, not financial advice. "
                        "Stock markets are inherently unpredictable. A 58% accuracy model that's right "
                        "slightly more than a coin flip is actually realistic for financial prediction. "
                        "Never trade based solely on model predictions.</span></div>",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"❌ ML Pipeline Error: {str(e)}")
                st.info("This can happen with insufficient data or unusual stock behavior. Try a Nifty 50 stock with 1Y+ data.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3: PORTFOLIO OPTIMIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab3:
    st.markdown("### 💼 Portfolio Optimization (Markowitz)")
    st.markdown(
        "<p style='color:#8899A6;'>Select 3+ stocks to find the optimal portfolio allocation "
        "using Modern Portfolio Theory.</p>",
        unsafe_allow_html=True,
    )

    # Stock selection for portfolio
    portfolio_tickers = st.multiselect(
        "Select stocks for portfolio",
        options=list(NIFTY50_STOCKS.keys()),
        default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"],
        format_func=lambda x: f"{NIFTY50_STOCKS.get(x, x)} ({x.replace('.NS', '')})",
    )

    if len(portfolio_tickers) < 3:
        st.warning("⚠️ Please select at least 3 stocks for portfolio optimization.")
    else:
        with st.spinner("📊 Computing portfolio analytics..."):
            try:
                # Fetch data for all selected stocks
                stock_data = fetch_multiple_stocks(portfolio_tickers, "2y")

                if len(stock_data) < 3:
                    st.error("Could not fetch data for enough stocks. Try different selections.")
                else:
                    # Compute returns matrix
                    returns_df = compute_returns_matrix(stock_data)

                    # ── Correlation Matrix ──
                    st.markdown("#### 🔗 Correlation Matrix")
                    st.markdown(
                        "<p style='color:#8899A6; font-size:0.85rem;'>"
                        "Shows how stock returns move together. Low correlation = better diversification.</p>",
                        unsafe_allow_html=True,
                    )

                    corr_matrix = compute_correlation_matrix(returns_df)
                    # Clean column names for display
                    clean_labels = [t.replace(".NS", "") for t in corr_matrix.columns]

                    corr_fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=clean_labels,
                        y=clean_labels,
                        colorscale=[[0, COLORS["red"]], [0.5, COLORS["bg_dark"]], [1, COLORS["green"]]],
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate="%{text}",
                        textfont={"size": 11},
                    ))
                    corr_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor=COLORS["bg_dark"],
                        plot_bgcolor=COLORS["bg_dark"],
                        height=450,
                        margin=dict(l=80, r=30, t=20, b=80),
                        font=dict(family="Inter"),
                    )
                    st.plotly_chart(corr_fig, width="stretch")

                    # ── Efficient Frontier ──
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("#### 🎯 Efficient Frontier")
                    st.markdown(
                        "<p style='color:#8899A6; font-size:0.85rem;'>"
                        "Each dot is a random portfolio. The curve shows the optimal risk-return tradeoff. "
                        "This is Modern Portfolio Theory (Markowitz, Nobel Prize 1990) in action.</p>",
                        unsafe_allow_html=True,
                    )

                    # Generate random portfolios
                    frontier_df = generate_efficient_frontier(returns_df, num_portfolios=3000)

                    # Find optimal portfolios
                    max_sharpe = optimize_max_sharpe(returns_df)
                    min_var = optimize_min_variance(returns_df)

                    # Plot
                    ef_fig = go.Figure()

                    # Random portfolios (scatter)
                    ef_fig.add_trace(go.Scatter(
                        x=frontier_df["Volatility"] * 100,
                        y=frontier_df["Return"] * 100,
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=frontier_df["Sharpe"],
                            colorscale=[[0, COLORS["red"]], [0.5, COLORS["yellow"]], [1, COLORS["green"]]],
                            colorbar=dict(title="Sharpe Ratio", tickfont=dict(size=10)),
                            opacity=0.6,
                        ),
                        name="Random Portfolios",
                        hovertemplate="Return: %{y:.1f}%<br>Volatility: %{x:.1f}%<br>Sharpe: %{marker.color:.2f}",
                    ))

                    # Max Sharpe point
                    if max_sharpe:
                        ef_fig.add_trace(go.Scatter(
                            x=[max_sharpe["volatility"] * 100],
                            y=[max_sharpe["return"] * 100],
                            mode="markers+text",
                            marker=dict(size=16, color=COLORS["green"], symbol="star",
                                        line=dict(width=2, color="white")),
                            text=["Max Sharpe"],
                            textposition="top center",
                            textfont=dict(color=COLORS["green"], size=11),
                            name=f"Max Sharpe ({max_sharpe['sharpe']:.2f})",
                        ))

                    # Min Variance point
                    if min_var:
                        ef_fig.add_trace(go.Scatter(
                            x=[min_var["volatility"] * 100],
                            y=[min_var["return"] * 100],
                            mode="markers+text",
                            marker=dict(size=16, color=COLORS["blue"], symbol="diamond",
                                        line=dict(width=2, color="white")),
                            text=["Min Risk"],
                            textposition="top center",
                            textfont=dict(color=COLORS["blue"], size=11),
                            name=f"Min Variance ({min_var['sharpe']:.2f})",
                        ))

                    ef_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor=COLORS["bg_dark"],
                        plot_bgcolor=COLORS["bg_dark"],
                        height=500,
                        margin=dict(l=60, r=30, t=20, b=50),
                        xaxis_title="Annual Volatility (%)",
                        yaxis_title="Expected Annual Return (%)",
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(size=11),
                        ),
                        font=dict(family="Inter"),
                    )
                    ef_fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
                    ef_fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
                    st.plotly_chart(ef_fig, width="stretch")

                    # ── Optimal Weights ──
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("#### ⚖️ Optimal Portfolio Allocation")

                    opt_cols = st.columns(2)

                    for col, (title, result, color) in zip(opt_cols, [
                        ("🟢 Maximum Sharpe Ratio", max_sharpe, COLORS["green"]),
                        ("🔵 Minimum Variance", min_var, COLORS["blue"]),
                    ]):
                        with col:
                            if result:
                                st.markdown(f"**{title}**")
                                st.markdown(
                                    f"Return: **{result['return']*100:.1f}%** | "
                                    f"Risk: **{result['volatility']*100:.1f}%** | "
                                    f"Sharpe: **{result['sharpe']:.2f}**"
                                )

                                weights = result["weights"]
                                clean_weights = {
                                    k.replace(".NS", ""): v
                                    for k, v in weights.items() if v > 0.01
                                }

                                if clean_weights:
                                    pie_fig = go.Figure(data=[go.Pie(
                                        labels=list(clean_weights.keys()),
                                        values=list(clean_weights.values()),
                                        hole=0.45,
                                        marker=dict(
                                            colors=px.colors.qualitative.Set3[:len(clean_weights)],
                                            line=dict(color=COLORS["bg_dark"], width=2),
                                        ),
                                        textinfo="label+percent",
                                        textfont=dict(size=11),
                                    )])
                                    pie_fig.update_layout(
                                        template="plotly_dark",
                                        paper_bgcolor=COLORS["bg_dark"],
                                        plot_bgcolor=COLORS["bg_dark"],
                                        height=350,
                                        margin=dict(l=20, r=20, t=20, b=20),
                                        showlegend=False,
                                        font=dict(family="Inter"),
                                    )
                                    st.plotly_chart(pie_fig, width="stretch")
                            else:
                                st.warning("Optimization did not converge.")

            except Exception as e:
                st.error(f"❌ Portfolio optimization error: {str(e)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 4: ABOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab4:
    st.markdown("### ℹ️ About EquityPulse")

    st.markdown("""
    <div style="background:#1A1F2E; border-radius:12px; padding:1.5rem; border:1px solid rgba(0,212,170,0.1);">

    **EquityPulse** is an AI-powered stock analytics dashboard that combines technical analysis,
    quantitative risk assessment, machine learning predictions, and Modern Portfolio Theory into
    a single, interactive application.

    #### 🏗️ Architecture

    | Layer | Technology | Purpose |
    |---|---|---|
    | **Data** | yfinance, Pandas, NumPy | Real-time NSE/BSE stock data |
    | **Analytics** | Custom indicators, SciPy | Technical & risk analysis |
    | **ML** | XGBoost, SHAP | Direction prediction & explainability |
    | **Visualization** | Plotly | Interactive charts |
    | **Frontend** | Streamlit | Dashboard UI |

    #### 🧠 How the ML Model Works

    1. **Feature Engineering**: 15 features extracted from technical indicators (RSI, MACD, Bollinger Bands, volume changes, momentum)
    2. **Walk-Forward Validation**: Trained on past data, tested on future — no data leakage
    3. **XGBoost Classifier**: Predicts UP/DOWN direction for next trading day
    4. **SHAP Explainability**: Shows *why* the model made each prediction

    #### 📐 Linear Algebra in Action

    - **Cosine Similarity** in embedding spaces
    - **Covariance Matrices** for portfolio risk
    - **Matrix Multiplication** for portfolio variance: `σ² = w^T × Σ × w`
    - **Optimization** via scipy for efficient frontier

    #### ⚠️ Disclaimer

    This is a **learning project** built for educational and portfolio purposes.
    It is **not financial advice**. Stock markets involve risk, and past performance
    does not guarantee future results. Always consult a qualified financial advisor.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    about_cols = st.columns(3)
    with about_cols[0]:
        st.markdown("""
        #### 👨‍💻 Built by
        **Mohit Patle**

        [GitHub](https://github.com/mohit6603) •
        [LinkedIn](https://linkedin.com/in/mohit6603)
        """)
    with about_cols[1]:
        st.markdown("""
        #### 📊 Data Source
        Yahoo Finance (via yfinance)

        Real-time & historical data
        for NSE/BSE listed stocks
        """)
    with about_cols[2]:
        st.markdown("""
        #### 🛠️ Tech Stack
        Python • Pandas • NumPy •
        XGBoost • SHAP • Plotly •
        Streamlit • SciPy
        """)
