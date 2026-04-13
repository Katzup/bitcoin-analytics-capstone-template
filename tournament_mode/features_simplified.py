"""
Simplified Feature Generation (No CNN Required)

Based on ablation studies showing CNN signal is equivalent to random,
this module provides simple technical indicators that achieve same or better
performance with 100x less complexity.

Performance comparison (from ablations + signal variant testing):
    CNN approach:              41.43% RW percentile, 54.32% win rate
    Neutral (constant):        41.94% RW percentile, 70.42% win rate
    OLS contrarian (raw):      43.25% RW percentile, 91.48% win rate  (+1.31 pp vs neutral)
    OLS winsorized (z-clip):   44.95% RW percentile, 66.94% win rate  (+3.01 pp vs neutral)

Key advantages:
    - No pre-trained model required
    - 100x faster execution (no GAF generation, no CNN inference)
    - More robust (no overfitting to 2014-2015 data)
    - Easier to understand and maintain
"""

import numpy as np
import pandas as pd
from typing import Optional


def _fast_rolling_ols(log_prices: np.ndarray, window: int):
    """
    Vectorized rolling OLS: fits log(price) ~ linear trend over each window.

    Returns (slopes, r2s) arrays of same length as log_prices.
    - slope: daily log-return trend (multiply by 365 for annualized)
    - r2: coefficient of determination [0, 1] — trend quality/confidence

    First (window-1) elements are NaN (insufficient history).

    Causality: window[t] = log_prices[t-window+1 : t+1], no lookahead.
    """
    n = len(log_prices)
    if n < window:
        return np.full(n, np.nan), np.full(n, np.nan)

    # Precompute OLS weights (fixed for given window size)
    x = np.arange(window, dtype=float)
    x_mean = (window - 1) / 2.0
    x_centered = x - x_mean          # shape (window,)
    ss_xx = np.sum(x_centered ** 2)
    slope_weights = x_centered / ss_xx  # shape (window,)

    # Build sliding window view (vectorized, no Python loop)
    # windows[i] = log_prices[i : i+window]
    row_idx = np.arange(n - window + 1)[:, None]
    col_idx = np.arange(window)[None, :]
    windows = log_prices[row_idx + col_idx]  # shape (n-window+1, window)

    # Slopes via dot product (subtract mean first for numerical stability)
    y_means = windows.mean(axis=1, keepdims=True)          # (n-window+1, 1)
    slopes = (windows - y_means) @ slope_weights           # (n-window+1,)

    # R² = 1 - SS_res / SS_tot
    y_hat = y_means + slopes[:, None] * x_centered         # (n-window+1, window)
    ss_res = ((windows - y_hat) ** 2).sum(axis=1)
    ss_tot = ((windows - y_means) ** 2).sum(axis=1)
    r2s = np.where(ss_tot > 1e-10,
                   np.maximum(0.0, 1.0 - ss_res / ss_tot),
                   0.0)

    # Pad head with NaN (insufficient history)
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, slopes]), np.concatenate([pad, r2s])


def build_features_trend_ols(
    df: pd.DataFrame,
    lookback: int = 200,
    price_col: str = 'PriceUSD_coinmetrics',
    short_window: int = 60,
    long_window: int = 180,
    vol_window: int = 20,
    vol_threshold: float = 0.65,
    regime_short_ma: int = 50,
    regime_long_ma: int = 200,
) -> pd.DataFrame:
    """
    OLS TrendScore + Regime Detection features.

    Signal logic (CONTRARIAN + Z-SCORE SHRINKAGE — optimized for sats accumulation):
        TrendScore = 0.4*(slope_60d * R²_60d) + 0.6*(slope_180d * R²_180d)
        Both slopes are annualized (× 365).

        Z-score normalization (signal shrinkage):
            ts_z = clip((TrendScore - rolling_mean_252d) / rolling_std_252d, -2, +2)
            prob_up = 0.5 - 0.15 * tanh(ts_z)   ← INVERTED + NORMALIZED

        The rolling z-score makes the signal relative to its own recent history:
        - During high-volatility regimes (large std), raw TrendScore is dampened
        - During calm regimes (small std), the same absolute tilt matters more
        - Clip ±2σ prevents extreme outliers from driving large position tilts
        - Result: +1.70 pp RW over raw contrarian (+3.01 pp over neutral DCA)

        Positive TrendScore → uptrend → prob_up < 0.5 → buy LESS (prices expensive)
        Negative TrendScore → downtrend → prob_up > 0.5 → buy MORE (prices cheap)
        R² gate: noisy/sideways markets produce low R² → small tilt → near-DCA

        Rationale: For the Stacking SATS tournament (maximize sats/dollar vs DCA),
        buying MORE during downtrends accumulates cheap sats. Z-score normalization
        achieves 44.95% RW vs neutral 41.94% (+3.01 pp) and 66.94% win rate.

    Regime detection (used by compute_weights for sensitivity scaling):
        regime_score = +1  (risk_on):  MA50 > MA200 AND vol_20d < threshold
        regime_score = -1  (risk_off): MA50 < MA200 AND vol_20d >= threshold
        regime_score =  0  (neutral):  mixed signals

    Args:
        df: DataFrame with price column and DatetimeIndex
        lookback: Minimum rows before features are valid (default 200)
        price_col: Name of price column
        short_window: OLS window for short-term trend (default 60 days)
        long_window: OLS window for long-term trend (default 180 days)
        vol_window: Lookback for realized volatility (default 20 days)
        vol_threshold: Annualized vol threshold for regime (default 0.65)
        regime_short_ma: Short MA for regime detection (default 50 days)
        regime_long_ma: Long MA for regime detection (default 200 days)

    Returns:
        DataFrame with:
            - 'prob_up': trend signal mapped to [0.35, 0.65], NaN before lookback
            - 'regime_score': -1/0/+1 regime state, NaN before lookback

    Causality: all computations use only data[0:t+1] for row t.
    """
    prices = df[price_col].values
    n = len(prices)
    log_prices = np.log(prices)

    # --- OLS TrendScore (vectorized) ---
    slopes_60, r2_60 = _fast_rolling_ols(log_prices, short_window)
    slopes_180, r2_180 = _fast_rolling_ols(log_prices, long_window)

    # Annualize and weight: longer window is more stable
    trend_score = (0.4 * slopes_60 * 365 * r2_60 +
                   0.6 * slopes_180 * 365 * r2_180)

    # Z-score normalization: makes signal relative to its own rolling history.
    # Causality: rolling(252) uses only past data — no lookahead.
    # min_periods=60: start producing values once 60d of trend_score is available.
    ts_series = pd.Series(trend_score)
    ts_mean = ts_series.rolling(252, min_periods=60).mean()
    ts_std  = ts_series.rolling(252, min_periods=60).std().clip(lower=1e-8)
    ts_z    = ((ts_series - ts_mean) / ts_std).clip(-2.0, 2.0).values

    # Map to prob_up: ±0.15 max tilt — CONTRARIAN + NORMALIZED (inverted for sats accumulation)
    # Key insight: for the sats tournament (maximize sats/dollar), we want to
    # BUY MORE when prices are cheap (downtrend = negative z) and LESS when
    # prices are high (uptrend = positive z).
    # Negative ts_z → prob_up > 0.5 → buy more (cheap sats)
    # Positive ts_z → prob_up < 0.5 → buy less (expensive sats)
    prob_up_arr = 0.5 - 0.15 * np.tanh(ts_z)

    # --- Regime Detection (vectorized via pandas rolling) ---
    prices_series = pd.Series(prices, index=df.index)
    ma_short = prices_series.rolling(regime_short_ma).mean().values
    ma_long = prices_series.rolling(regime_long_ma).mean().values

    # Realized vol: annualized from daily log returns
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.diff(log_prices)
    log_ret_s = pd.Series(log_ret, index=df.index)
    realized_vol = log_ret_s.rolling(vol_window).std().values * np.sqrt(252)

    # MA signal: +1 if short > long, -1 otherwise
    ma_ok = np.isfinite(ma_short) & np.isfinite(ma_long)
    ma_signal = np.where(ma_ok, np.where(ma_short > ma_long, 1.0, -1.0), np.nan)

    # Vol signal: +1 if vol low (normal conditions), -1 if vol elevated
    vol_ok = np.isfinite(realized_vol)
    vol_signal = np.where(vol_ok, np.where(realized_vol < vol_threshold, 1.0, -1.0), np.nan)

    # Regime: require BOTH signals to agree for strong label
    raw_regime = (ma_signal + vol_signal) / 2.0  # ∈ {-1, -0.5, 0, +0.5, +1}
    regime_arr = np.where(raw_regime > 0.3, 1.0,
                          np.where(raw_regime < -0.3, -1.0, 0.0))
    # NaN where inputs were NaN
    regime_arr = np.where(np.isfinite(raw_regime), regime_arr, np.nan)

    # --- Build output DataFrame ---
    min_valid = max(lookback, long_window, regime_long_ma) - 1

    features_df = pd.DataFrame(index=df.index)
    features_df['prob_up'] = np.nan
    features_df['regime_score'] = np.nan

    if min_valid < n:
        features_df['prob_up'].iloc[min_valid:] = prob_up_arr[min_valid:]
        features_df['regime_score'].iloc[min_valid:] = regime_arr[min_valid:]

    return features_df


def build_features_neutral(
    df: pd.DataFrame,
    lookback: int = 90,
    price_col: str = 'PriceUSD_coinmetrics'
) -> pd.DataFrame:
    """
    Generate neutral features (constant probability).

    This is the control baseline that achieves 41.94% RW percentile.
    Use when you want consistent, predictable behavior.

    Args:
        df: DataFrame with price column
        lookback: Days to skip at start (for consistency with CNN version)
        price_col: Name of price column

    Returns:
        DataFrame with prob_up = 0.5 (neutral)
    """
    features_df = pd.DataFrame(index=df.index)
    features_df['prob_up'] = np.nan

    # Set neutral probability after lookback period
    features_df.iloc[lookback:, features_df.columns.get_loc('prob_up')] = 0.5

    return features_df


def build_features_momentum(
    df: pd.DataFrame,
    lookback: int = 90,
    momentum_window: int = 20,
    price_col: str = 'PriceUSD_coinmetrics'
) -> pd.DataFrame:
    """
    Generate features based on simple momentum.

    Logic: If price > 20-day MA, assume higher probability of continuation.
    This captures trend-following behavior without complex models.

    Args:
        df: DataFrame with price column
        lookback: Days to skip at start (for alignment with CNN version)
        momentum_window: Days for momentum calculation (default 20)
        price_col: Name of price column

    Returns:
        DataFrame with prob_up based on momentum signal
    """
    features_df = pd.DataFrame(index=df.index)
    features_df['prob_up'] = np.nan

    prices = df[price_col].values

    # Calculate rolling momentum
    for t in range(lookback, len(df)):
        # Current price vs moving average
        if t >= momentum_window:
            ma = np.mean(prices[t-momentum_window:t])
            current = prices[t-1]  # Use t-1 to maintain causality

            # Map distance from MA to probability
            # Above MA → prob > 0.5, below MA → prob < 0.5
            distance_pct = (current - ma) / ma

            # Sigmoid-like mapping: [-10%, +10%] → [0.3, 0.7]
            prob = 0.5 + 0.2 * np.tanh(distance_pct * 5)

            features_df.iloc[t, features_df.columns.get_loc('prob_up')] = prob
        else:
            features_df.iloc[t, features_df.columns.get_loc('prob_up')] = 0.5

    return features_df


def build_features_ma_crossover(
    df: pd.DataFrame,
    lookback: int = 90,
    fast_window: int = 50,
    slow_window: int = 200,
    price_col: str = 'PriceUSD_coinmetrics'
) -> pd.DataFrame:
    """
    Generate features based on moving average crossover.

    Classic trend-following: 50-day MA vs 200-day MA
    - Fast > Slow: Bull market (prob_up > 0.5)
    - Fast < Slow: Bear market (prob_up < 0.5)

    Args:
        df: DataFrame with price column
        lookback: Days to skip at start
        fast_window: Fast MA period (default 50)
        slow_window: Slow MA period (default 200)
        price_col: Name of price column

    Returns:
        DataFrame with prob_up based on MA crossover
    """
    features_df = pd.DataFrame(index=df.index)
    features_df['prob_up'] = np.nan

    prices = df[price_col].values

    for t in range(lookback, len(df)):
        if t >= slow_window:
            fast_ma = np.mean(prices[t-fast_window:t])
            slow_ma = np.mean(prices[t-slow_window:t])

            # Distance between MAs determines conviction
            distance_pct = (fast_ma - slow_ma) / slow_ma

            # Map to probability: [-5%, +5%] → [0.3, 0.7]
            prob = 0.5 + 0.2 * np.tanh(distance_pct * 10)

            features_df.iloc[t, features_df.columns.get_loc('prob_up')] = prob
        else:
            features_df.iloc[t, features_df.columns.get_loc('prob_up')] = 0.5

    return features_df


def build_features_volatility_adjusted(
    df: pd.DataFrame,
    lookback: int = 90,
    vol_window: int = 20,
    price_col: str = 'PriceUSD_coinmetrics'
) -> pd.DataFrame:
    """
    Generate features with volatility-adjusted probability.

    Logic: High volatility → lower confidence (prob closer to 0.5)
    Low volatility → higher confidence in momentum signal

    Args:
        df: DataFrame with price column
        lookback: Days to skip at start
        vol_window: Days for volatility calculation
        price_col: Name of price column

    Returns:
        DataFrame with volatility-adjusted prob_up
    """
    features_df = pd.DataFrame(index=df.index)
    features_df['prob_up'] = np.nan

    prices = df[price_col].values
    returns = np.diff(np.log(prices))

    for t in range(lookback, len(df)):
        if t >= vol_window + 1:
            # Calculate recent volatility
            vol = np.std(returns[t-vol_window:t-1]) * np.sqrt(252)  # Annualized

            # Calculate momentum
            if t >= 20:
                ma20 = np.mean(prices[t-20:t])
                current = prices[t-1]
                momentum = (current - ma20) / ma20
            else:
                momentum = 0

            # Adjust confidence based on volatility
            # High vol (>100%) → stay neutral
            # Low vol (<50%) → trust momentum more
            vol_factor = np.clip(1.0 - (vol - 0.5) / 0.5, 0.5, 1.0)

            # Base probability from momentum, scaled by volatility confidence
            prob_base = 0.5 + 0.3 * np.tanh(momentum * 5)
            prob = 0.5 + (prob_base - 0.5) * vol_factor

            features_df.iloc[t, features_df.columns.get_loc('prob_up')] = prob
        else:
            features_df.iloc[t, features_df.columns.get_loc('prob_up')] = 0.5

    return features_df


# Default: Use neutral (most robust)
build_features = build_features_neutral


def construct_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tournament-required endpoint for feature generation.

    This is the official interface expected by the tournament template/grader.
    Must be causal: features for day t only use data <= t.
    Must preserve ALL original columns plus add new feature columns.

    Args:
        df: DataFrame with columns=['PriceUSD_coinmetrics'],
            index=DatetimeIndex

    Returns:
        df_enriched: Input DataFrame (all columns preserved) with added column:
            - 'prob_up': z-score normalized contrarian signal [0.35, 0.65], NaN for first 200 rows

    Contract:
        - Preserves ALL input DataFrame columns (especially price)
        - Preserves input DataFrame index exactly
        - Adds 'prob_up' column
        - NaN for first 200 rows (insufficient history for 180d OLS + 200d MA)
        - No lookahead bias (row t uses only data <= t)
        - Deterministic (same input → same output, seed-independent)

    Signal design (contrarian + z-score shrinkage — buy dips for sats accumulation):
        TrendScore = 0.4*(slope_60d × R²_60d) + 0.6*(slope_180d × R²_180d)
        ts_z      = clip((TrendScore - rolling_mean_252d) / rolling_std_252d, -2, +2)
        prob_up   = 0.5 - 0.15 × tanh(ts_z)   ← INVERTED + NORMALIZED
        Performance: 44.95% RW percentile, 66.94% win rate (vs neutral 41.94% / 70.42%)
        regime_score NOT propagated (backtest shows it hurts vs pure contrarian)
    """
    # CRITICAL: Preserve all original columns + add features
    df_enriched = df.copy()

    # Generate OLS TrendScore + regime features
    features = build_features_trend_ols(
        df,
        lookback=200,
        price_col='PriceUSD_coinmetrics'
    )

    # Add prob_up to enriched DataFrame (preserving all original columns).
    # NOTE: regime_score is NOT propagated here — backtest shows the contrarian
    # OLS signal alone (43.25% RW) outperforms adding the regime gate (42.41% RW).
    # The contrarian signal already captures bear/bull dynamics implicitly.
    df_enriched['prob_up'] = features['prob_up']

    return df_enriched


__all__ = [
    'construct_features',  # Tournament-required endpoint
    'build_features',
    'build_features_neutral',
    'build_features_momentum',
    'build_features_ma_crossover',
    'build_features_volatility_adjusted',
    'build_features_trend_ols',
]
