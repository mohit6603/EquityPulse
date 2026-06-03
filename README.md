# 📊 EquityPulse — AI-Powered Stock Analytics Dashboard

> Real-time technical analysis, ML-based direction prediction, and portfolio optimization for NSE/BSE stocks.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo

👉 **[Launch EquityPulse](https://equitypulse.streamlit.app)**  *(Update with your actual URL after deployment)*

---

## ✨ Features

### 📈 Technical Analysis
- Interactive candlestick charts with SMA, EMA, and Bollinger Band overlays
- RSI (Relative Strength Index) with overbought/oversold zones
- MACD with signal line and histogram
- Real-time signal summary (Bullish / Bearish / Neutral)

### 📊 Risk Metrics
- **Sharpe Ratio** — Risk-adjusted return measurement
- **Value at Risk (VaR)** — Maximum expected loss at 95% confidence
- **Maximum Drawdown** — Worst historical peak-to-trough decline
- **Beta & Alpha** — CAPM-based metrics vs. Nifty 50 benchmark

### 🤖 ML Prediction
- **XGBoost** classifier predicting next-day direction (UP/DOWN)
- **15 engineered features** from technical indicators
- **Walk-forward validation** — no data leakage, honest evaluation
- **SHAP explainability** — see *why* the model made each prediction

### 💼 Portfolio Optimizer
- **Correlation heatmap** — visualize stock relationships
- **Efficient Frontier** — 3,000 simulated portfolios
- **Maximum Sharpe** and **Minimum Variance** optimal portfolios
- Based on **Modern Portfolio Theory** (Markowitz)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 Streamlit UI                      │
│          (Plotly Charts + Custom CSS)             │
├──────────┬──────────┬───────────┬────────────────┤
│ Technical│  Risk    │    ML     │   Portfolio     │
│Indicators│ Metrics  │ Predictor │   Optimizer     │
├──────────┴──────────┴───────────┴────────────────┤
│              Data Fetcher (yfinance)              │
│            + Streamlit Cache Layer                │
├──────────────────────────────────────────────────┤
│           Yahoo Finance API (Free)                │
└──────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.11+ |
| **Data** | Pandas, NumPy, yfinance |
| **Machine Learning** | XGBoost, Scikit-learn, SHAP |
| **Math/Optimization** | SciPy, NumPy (Linear Algebra) |
| **Visualization** | Plotly |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Cloud |

---

## 📦 Setup & Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/mohit6603/EquityPulse.git
cd EquityPulse

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📐 Key Concepts & Math

### Technical Indicators
- **SMA(n)**: Simple Moving Average — `SMA = Σ(Close_i) / n`
- **RSI**: Relative Strength Index — `RSI = 100 - (100 / (1 + RS))` where `RS = avg_gain / avg_loss`
- **MACD**: `EMA(12) - EMA(26)` with signal line `EMA(9)`
- **Bollinger Bands**: `SMA(20) ± 2σ`

### Risk Metrics
- **Sharpe Ratio**: `(R - Rf) / σ × √252` — Risk-adjusted return
- **Beta**: `Cov(Ri, Rm) / Var(Rm)` — Market sensitivity
- **VaR(95%)**: 5th percentile of daily returns distribution

### Portfolio Optimization (Linear Algebra)
- **Portfolio Variance**: `σ²p = wᵀ × Σ × w` (matrix multiplication)
- **Efficient Frontier**: Monte Carlo simulation of weight vectors
- **Optimization**: SciPy SLSQP minimizer with constraints

---

## 📁 Project Structure

```
equity-pulse/
├── app.py                         # Main Streamlit dashboard
├── requirements.txt               # Dependencies
├── .streamlit/
│   └── config.toml                # Dark theme configuration
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Constants & formatting helpers
│   ├── data_fetcher.py            # Yahoo Finance data download
│   ├── technical_indicators.py    # SMA, EMA, RSI, MACD, Bollinger
│   ├── risk_metrics.py            # Sharpe, VaR, Drawdown, Beta
│   ├── ml_predictor.py            # XGBoost + SHAP explainability
│   └── portfolio_optimizer.py     # Efficient Frontier & Markowitz
├── README.md
├── .gitignore
└── LICENSE
```

---

## ⚠️ Disclaimer

This project is built for **educational and portfolio purposes only**. It is **not financial advice**. Stock markets involve significant risk, and past performance does not guarantee future results. The ML model achieves ~55-60% directional accuracy, which is realistic for financial prediction but should never be used as a sole trading signal.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Mohit Patle**

- 🔗 [GitHub](https://github.com/mohit6603)
- 🔗 [LinkedIn](https://linkedin.com/in/mohit6603)

---

*Built with ❤️ and Python*
