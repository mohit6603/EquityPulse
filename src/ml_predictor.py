"""
ML Predictor Module — Stock Direction Prediction with XGBoost + SHAP.

This module predicts whether a stock's price will go UP or DOWN tomorrow
using machine learning (XGBoost gradient-boosted trees) and explains
WHY the model made that prediction using SHAP values.

Pipeline Overview:
    1. engineer_features()  → Turn raw OHLCV into 15 ML-ready features
    2. walk_forward_split() → Time-aware cross-validation (no data leakage!)
    3. train_model()        → Train XGBoost with walk-forward validation
    4. predict_direction()  → Predict tomorrow's direction + confidence
    5. explain_prediction() → SHAP values showing feature contributions
    6. get_feature_importance() → Global feature ranking

Key Concept (for your learning):
    We NEVER randomly shuffle stock data for train/test splits.
    Markets are sequential — you can only learn from the PAST to
    predict the FUTURE. Random splits cause "data leakage" where
    the model accidentally sees future data during training.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import add_all_indicators from the technical indicators module.
# This function enriches a raw OHLCV DataFrame with RSI, MACD, Bollinger
# Bands, SMA, EMA, ATR, and other indicator columns we need as features.
try:
    from src.technical_indicators import add_all_indicators
except ImportError:
    # If the technical_indicators module is not yet available, define a
    # lightweight fallback that computes the exact columns we need.
    # This ensures ml_predictor.py works standalone during development.
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: compute required indicator columns inline.

        This is used ONLY when src.technical_indicators is not available.
        In production, the real add_all_indicators() is preferred because
        it computes a broader set of indicators with more robust logic.
        """
        df = df.copy()

        # --- RSI (Relative Strength Index, 14-period) ---
        # RSI measures momentum on a 0-100 scale.
        # RSI = 100 - 100/(1 + RS), where RS = avg_gain / avg_loss
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # --- MACD (Moving Average Convergence Divergence) ---
        # MACD = EMA_12 - EMA_26; Signal = EMA_9(MACD)
        # Histogram = MACD - Signal  →  positive = bullish momentum
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df["MACD_Histogram"] = macd_line - signal_line

        # --- Bollinger Bands (20-period, 2 std deviations) ---
        # Bands expand/contract with volatility. %B shows where
        # the price sits relative to the bands (0 = lower, 1 = upper).
        sma_20 = df["Close"].rolling(window=20).mean()
        std_20 = df["Close"].rolling(window=20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        band_width = upper_band - lower_band
        df["BB_Pct_B"] = (df["Close"] - lower_band) / band_width.replace(0, np.nan)

        # --- Simple Moving Averages ---
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # --- Exponential Moving Average (26-period) ---
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # --- ATR (Average True Range, 14-period) ---
        # ATR measures volatility using the greatest of:
        #   High-Low, |High-PrevClose|, |Low-PrevClose|
        high_low = df["High"] - df["Low"]
        high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
        low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        df["ATR_14"] = true_range.rolling(window=14).mean()

        return df


# ──────────────────────────────────────────────────────────────
#  1. Feature Engineering
# ──────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Transform raw OHLCV data into 15 machine-learning-ready features.

    WHY feature engineering?
        Raw prices (Open/High/Low/Close) are not great ML inputs because
        they are non-stationary (always going up over time).  Instead we
        derive *relative* features — ratios, returns, crossovers — that
        capture the PATTERNS in price movement, not the absolute levels.

    Features Created (15 total):
        Technical Indicators:
            1.  RSI_14          — Relative Strength Index (0-100, momentum)
            2.  MACD_Histogram  — MACD minus Signal line (momentum)
            3.  BB_Pct_B        — Bollinger Band %B (0=lower, 1=upper band)
            4.  ATR_14          — Average True Range (volatility measure)

        Trend Signals (binary):
            5.  SMA_20_50_cross — 1 if SMA_20 > SMA_50 (golden cross)
            6.  EMA_trend       — 1 if Close > EMA_26 (uptrend)

        Price Returns:
            7.  Returns_1d      — 1-day price return
            8.  Returns_5d      — 5-day (weekly) return
            9.  Returns_10d     — 10-day (bi-weekly) return

        Volume & Volatility:
            10. Volume_Change   — Daily volume % change
            11. Volatility_20d  — 20-day rolling std of daily returns

        Derived:
            12. Price_vs_SMA50  — How far price is from 50-day SMA (%)
            13. RSI_Momentum    — RSI change over 5 days

        Calendar (seasonality):
            14. Day_of_Week     — 0=Mon, 4=Fri
            15. Month           — 1-12

    Target Variable:
        Binary classification: 1 if tomorrow's Close > today's Close, else 0.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume]
            and a DatetimeIndex. Must have at least ~60 rows so that
            indicators like SMA_50 can be computed.

    Returns:
        tuple of (features_df, target_series, feature_names_list):
            - features_df:  DataFrame with 15 feature columns, NaN-free
            - target_series: Series of 0/1 labels aligned to features_df
            - feature_names_list: List[str] of feature column names

    Raises:
        ValueError: If the input DataFrame is empty or has insufficient data.

    Example:
        >>> features, target, names = engineer_features(stock_df)
        >>> print(names)
        ['RSI_14', 'MACD_Histogram', 'BB_Pct_B', ...]
    """
    # --- Input Validation ---
    if df is None or df.empty:
        raise ValueError(
            "Cannot engineer features from an empty DataFrame. "
            "Need at least 60 trading days of OHLCV data."
        )

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if len(df) < 60:
        raise ValueError(
            f"Need at least 60 rows for feature engineering "
            f"(SMA_50 needs 50 days), but got {len(df)} rows."
        )

    # --- Step 1: Add technical indicator columns via add_all_indicators ---
    # This enriches df with RSI_14, MACD_Histogram, BB_Pct_B, SMA_20,
    # SMA_50, EMA_26, ATR_14, etc.
    df = add_all_indicators(df)

    # --- Step 2: Build the feature DataFrame ---
    features = pd.DataFrame(index=df.index)

    # ── Technical indicator features ──
    features["RSI_14"] = df["RSI_14"]
    features["MACD_Histogram"] = df["MACD_Histogram"]
    features["BB_Pct_B"] = df["BB_Pct_B"]
    features["ATR_14"] = df["ATR_14"]

    # ── Binary trend signals ──
    # SMA_20_50_cross: "Golden Cross" — short-term SMA above long-term SMA
    # is a classic bullish signal.  1 = bullish, 0 = bearish.
    features["SMA_20_50_cross"] = (df["SMA_20"] > df["SMA_50"]).astype(int)

    # EMA_trend: Price above its 26-day EMA suggests an uptrend.
    features["EMA_trend"] = (df["Close"] > df["EMA_26"]).astype(int)

    # ── Price returns over different horizons ──
    # Returns = (price_today - price_N_days_ago) / price_N_days_ago
    # These capture short, medium, and longer-term momentum.
    features["Returns_1d"] = df["Close"].pct_change(periods=1)
    features["Returns_5d"] = df["Close"].pct_change(periods=5)
    features["Returns_10d"] = df["Close"].pct_change(periods=10)

    # ── Volume dynamics ──
    # Volume_Change: spike in volume often precedes big price moves.
    features["Volume_Change"] = df["Volume"].pct_change(periods=1)

    # ── Volatility ──
    # Volatility_20d: standard deviation of daily returns over 20 days.
    # High volatility = uncertain, large swings expected.
    daily_returns = df["Close"].pct_change()
    features["Volatility_20d"] = daily_returns.rolling(window=20).std()

    # ── Derived features ──
    # Price_vs_SMA50: how far the price is from its 50-day average (%).
    # Positive = above average (potentially overbought),
    # Negative = below average (potentially oversold).
    features["Price_vs_SMA50"] = (
        (df["Close"] - df["SMA_50"]) / df["SMA_50"].replace(0, np.nan)
    )

    # RSI_Momentum: how RSI itself is changing — is momentum accelerating?
    features["RSI_Momentum"] = df["RSI_14"].diff(periods=5)

    # ── Calendar / seasonality features ──
    # Markets often have day-of-week effects (e.g., "Monday effect")
    # and monthly seasonality (e.g., "Sell in May").
    features["Day_of_Week"] = df.index.dayofweek    # 0=Mon, 4=Fri
    features["Month"] = df.index.month              # 1-12

    # --- Step 3: Create the target variable ---
    # Target = 1 if tomorrow's Close > today's Close, else 0.
    # We use shift(-1) to look one day into the future.
    target = (df["Close"].shift(-1) > df["Close"]).astype(int)
    target.name = "Target"

    # --- Step 4: Align and clean ---
    # Drop the last row (target is NaN because there's no "tomorrow" yet)
    # and drop any rows where features have NaN (from rolling windows).
    # IMPORTANT: Replace inf/-inf with NaN first — pct_change() can produce
    # inf when volume is 0 (division by zero), which crashes XGBoost.
    combined = pd.concat([features, target], axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    feature_names = [
        "RSI_14", "MACD_Histogram", "BB_Pct_B", "ATR_14",
        "SMA_20_50_cross", "EMA_trend",
        "Returns_1d", "Returns_5d", "Returns_10d",
        "Volume_Change", "Volatility_20d",
        "Price_vs_SMA50", "RSI_Momentum",
        "Day_of_Week", "Month",
    ]

    features_clean = combined[feature_names]
    target_clean = combined["Target"]

    if features_clean.empty:
        raise ValueError(
            "All rows were dropped after NaN removal. "
            "The input data may be too short for feature engineering."
        )

    return features_clean, target_clean, feature_names


# ──────────────────────────────────────────────────────────────
#  2. Walk-Forward (Time-Series) Cross-Validation
# ──────────────────────────────────────────────────────────────

def walk_forward_split(features: pd.DataFrame, target: pd.Series,
                       n_splits: int = 3):
    """
    Generate time-series cross-validation splits (walk-forward).

    WHY walk-forward instead of random K-fold?
        In stock data, time order matters!  If you randomly split,
        you might train on 2024 data and test on 2023 data — that's
        "seeing the future" (data leakage).  Walk-forward ensures:
            - Training data is ALWAYS before test data
            - Each fold uses progressively more training data

    Visual example with 5 splits:
        Fold 1: [TRAIN          ] [TEST ]
        Fold 2: [TRAIN               ] [TEST ]
        Fold 3: [TRAIN                    ] [TEST ]

    Args:
        features: Feature DataFrame with DatetimeIndex (time-ordered).
        target:   Target Series aligned with features.
        n_splits: Number of folds (default 3). More splits = more
                  thorough validation but slower.

    Yields:
        Tuples of (train_indices, test_indices) as numpy arrays.

    Raises:
        ValueError: If there are fewer data points than n_splits + 1.

    Example:
        >>> for train_idx, test_idx in walk_forward_split(X, y, n_splits=3):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        ...     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    """
    n_samples = len(features)

    if n_samples < n_splits + 1:
        raise ValueError(
            f"Need at least {n_splits + 1} samples for {n_splits} splits, "
            f"but got {n_samples}."
        )

    # Each test fold gets an equal share of the data.
    # The minimum training set uses the first portion of the data.
    # fold_size = how many rows each test set gets
    fold_size = n_samples // (n_splits + 1)

    for i in range(n_splits):
        # Training set: everything from the start up to the split point.
        # Each fold, the training set grows by one fold_size.
        train_end = fold_size * (i + 1)

        # Test set: the next fold_size rows after training.
        test_start = train_end
        test_end = test_start + fold_size

        # For the last fold, include any remaining rows in the test set.
        if i == n_splits - 1:
            test_end = n_samples

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        yield train_idx, test_idx


# ──────────────────────────────────────────────────────────────
#  3. Model Training
# ──────────────────────────────────────────────────────────────

def train_model(features: pd.DataFrame, target: pd.Series) -> tuple:
    """
    Train an XGBoost classifier to predict stock direction (UP/DOWN).

    WHY XGBoost?
        XGBoost (eXtreme Gradient Boosting) builds an ensemble of
        decision trees sequentially — each new tree focuses on correcting
        the mistakes of the previous trees.  It's the go-to algorithm
        for tabular data because:
            - Handles non-linear relationships
            - Built-in regularisation (prevents overfitting)
            - Fast training
            - Works well with mixed feature types

    Training Process:
        1. Walk-forward cross-validation to get honest performance metrics
        2. Train a final model on ALL available data for deployment

    Args:
        features: DataFrame of ML features (from engineer_features).
        target:   Series of binary labels (1=UP, 0=DOWN).

    Returns:
        tuple of (model, metrics_dict, feature_names):
            - model: Trained XGBClassifier ready for prediction
            - metrics_dict: Dict with keys 'accuracy', 'precision',
              'recall', 'f1', each containing a list of per-fold scores
              and a 'mean' key with the average.
            - feature_names: List of feature column names

    Raises:
        ValueError: If features or target are empty.

    Example:
        >>> model, metrics, names = train_model(features_df, target_series)
        >>> print(f"Mean accuracy: {metrics['accuracy']['mean']:.2%}")
    """
    if features.empty or target.empty:
        raise ValueError("Cannot train on empty features or target.")

    feature_names = list(features.columns)

    # ── XGBoost hyperparameters ──
    # n_estimators=100: build 100 trees (more = slower but often better)
    # max_depth=4:      each tree can be 4 levels deep (prevents overfitting)
    # learning_rate=0.1: each tree contributes 10% — slow and steady
    # eval_metric='logloss': binary cross-entropy loss for classification
    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,            # Use all CPU cores
        "verbosity": 0,          # Suppress XGBoost's own logging
    }

    # ── Walk-forward cross-validation ──
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for train_idx, test_idx in walk_forward_split(features, target):
        X_train = features.iloc[train_idx]
        X_test = features.iloc[test_idx]
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]

        fold_model = XGBClassifier(**params)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        # Compute classification metrics for this fold.
        # zero_division=0 avoids warnings when a class has no predictions.
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(
            precision_score(y_test, y_pred, zero_division=0)
        )
        metrics["recall"].append(
            recall_score(y_test, y_pred, zero_division=0)
        )
        metrics["f1"].append(
            f1_score(y_test, y_pred, zero_division=0)
        )

    # ── Compute mean scores across folds ──
    for key in metrics:
        scores = metrics[key]
        metrics[key] = {
            "folds": scores,
            "mean": float(np.mean(scores)),
        }

    # ── Train final model on ALL data ──
    # For production predictions, we want the model to learn from
    # every available data point.  The cross-validation metrics above
    # give us an honest estimate of how well this approach generalises.
    final_model = XGBClassifier(**params)
    final_model.fit(features, target)

    return final_model, metrics, feature_names


# ──────────────────────────────────────────────────────────────
#  4. Prediction
# ──────────────────────────────────────────────────────────────

def predict_direction(model: XGBClassifier,
                      latest_features: pd.DataFrame) -> tuple:
    """
    Predict whether the stock will go UP or DOWN tomorrow.

    HOW it works:
        XGBoost outputs a probability P(class=1) = P(UP).
        - If P > 0.5 → predict UP,  confidence = P
        - If P ≤ 0.5 → predict DOWN, confidence = 1 - P

    Args:
        model:           Trained XGBClassifier (from train_model).
        latest_features: DataFrame with exactly ONE row containing the
                         most recent feature values.  Column names must
                         match those used during training.

    Returns:
        tuple of (direction, confidence):
            - direction:  'UP' or 'DOWN'
            - confidence: float in [0.5, 1.0] — how sure the model is

    Raises:
        ValueError: If latest_features is empty.

    Example:
        >>> direction, conf = predict_direction(model, features.iloc[[-1]])
        >>> print(f"Prediction: {direction} ({conf:.1%} confident)")
    """
    if latest_features is None or latest_features.empty:
        raise ValueError("latest_features must contain at least one row.")

    # predict_proba returns [[P(DOWN), P(UP)]] — we want P(UP)
    probabilities = model.predict_proba(latest_features)
    prob_up = probabilities[0, 1]  # Probability of class 1 (UP)

    if prob_up > 0.5:
        return "UP", float(prob_up)
    else:
        # Confidence for DOWN is how sure we are it's NOT up
        return "DOWN", float(1.0 - prob_up)


# ──────────────────────────────────────────────────────────────
#  5. SHAP Explainability
# ──────────────────────────────────────────────────────────────

def explain_prediction(model: XGBClassifier,
                       features: pd.DataFrame,
                       feature_names: list) -> tuple:
    """
    Explain the model's prediction using SHAP (SHapley Additive exPlanations).

    WHY SHAP?
        ML models are often "black boxes".  SHAP borrows a concept from
        cooperative game theory (Shapley values) to answer:
        "How much did each feature contribute to THIS specific prediction?"

        Example interpretation:
            "RSI_14 pushed the prediction toward UP by +0.12"
            "Volume_Change pushed it toward DOWN by -0.08"

    IMPORTANT:
        This function returns raw SHAP values only.  It does NOT create
        any plots — the Streamlit app handles visualisation separately.

    Args:
        model:         Trained XGBClassifier.
        features:      Feature DataFrame (can be multiple rows; SHAP values
                       are computed for ALL rows, but typically you care
                       about the last one — the latest prediction).
        feature_names: List of feature column names in the correct order.

    Returns:
        tuple of (shap_values, expected_value, feature_names):
            - shap_values:    np.ndarray of shape (n_samples, n_features).
                              Each value shows the contribution of that
                              feature to moving the prediction away from
                              the base rate (expected_value).
            - expected_value: float — the model's average prediction across
                              the training data (the "base rate").
            - feature_names:  List[str] — passed through for convenience.

    Raises:
        ValueError: If features DataFrame is empty.

    Example:
        >>> sv, ev, names = explain_prediction(model, features, feature_names)
        >>> # sv[-1] gives SHAP values for the most recent prediction
    """
    if features is None or features.empty:
        raise ValueError("Cannot explain predictions on empty features.")

    # TreeExplainer is optimised for tree-based models (XGBoost, LightGBM).
    # It computes exact Shapley values in polynomial time (fast!).
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values for all rows.
    # For binary classification, shap_values may be a list [class_0, class_1]
    # or a single array depending on the SHAP version.
    shap_values_raw = explainer.shap_values(features[feature_names])

    # Handle both SHAP output formats:
    # - Older shap: returns list of arrays [shap_class_0, shap_class_1]
    # - Newer shap: returns a single array for binary classification
    if isinstance(shap_values_raw, list):
        # Use class 1 (UP) SHAP values — we care about "what drives UP?"
        shap_values = shap_values_raw[1]
    else:
        shap_values = shap_values_raw

    # expected_value is the base prediction before any features are considered.
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    return shap_values, float(expected_value), feature_names


# ──────────────────────────────────────────────────────────────
#  6. Feature Importance (Global)
# ──────────────────────────────────────────────────────────────

def get_feature_importance(model: XGBClassifier,
                           feature_names: list) -> list:
    """
    Get global feature importance scores from the trained model.

    WHAT is feature importance?
        After training, XGBoost knows which features were most useful
        across ALL predictions.  This is different from SHAP (which
        explains a SINGLE prediction).

        XGBoost's default importance = how often a feature was used to
        split the data in decision trees ("weight" or "gain").

    Args:
        model:         Trained XGBClassifier.
        feature_names: List of feature column names.

    Returns:
        List of (feature_name, importance_score) tuples, sorted by
        importance in DESCENDING order.  Importance scores are
        normalised (sum to ~1).

    Example:
        >>> importances = get_feature_importance(model, feature_names)
        >>> for name, score in importances[:5]:
        ...     print(f"  {name}: {score:.4f}")
    """
    # feature_importances_ gives normalised gain-based importance.
    raw_importances = model.feature_importances_

    # Pair each feature name with its importance score.
    paired = list(zip(feature_names, raw_importances))

    # Sort descending — most important features first.
    paired.sort(key=lambda x: x[1], reverse=True)

    return paired
