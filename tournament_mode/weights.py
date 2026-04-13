"""
Tournament Weight Computation Module

Converts VTS CNN probability signals to normalized daily allocation weights.

VTS Allocation Logic (adapted for tournament):
    1. P(up) → tilt via sensitivity (sensitivity × (P - 0.5))
    2. Bounded multiplier: clip(1 + tilt, min_mult, max_mult)
    3. EMA smoothing with optional turnover governor (α=0.30 for daily granularity)
    4. Normalize to sum=1 (tournament constraint)
    5. Enforce minimum weight: clip to ≥ MIN_WEIGHT

Turnover Governor (Mode A: freeze_then_ema):
    Replaces vectorized EMA with a sequential day-by-day loop.
    On each day t >= 1, compares the new target multiplier to the prior
    smoothed multiplier. If the absolute change is below both thresholds
    (THRESH_L1 and THRESH_MAXDELTA), the multiplier is frozen (carried
    forward). Otherwise the standard EMA update is applied.

    Effective freeze condition (scalar per-day case):
        delta = |target[t] - smoothed[t-1]|
        freeze iff delta < THRESH_L1 AND delta < THRESH_MAXDELTA
              iff delta < min(THRESH_L1, THRESH_MAXDELTA)

    Purpose: damp flip-flopping between adjacent days caused by noisy
    prob_up signals. Reduces needless weight churn without slowing
    genuine regime changes.

CRITICAL: All operations must be causal (w_t depends only on data ≤ t)
"""

import numpy as np
import pandas as pd
from typing import Optional

# Import tournament constants
from .scoring import MIN_WEIGHT, TOLERANCE_RTOL, TOLERANCE_ATOL, validate_weights


# VTS allocation parameters (from config_multi_timeframe BTC_1D)
DEFAULT_SENSITIVITY = 1.5
DEFAULT_MIN_MULT = 0.7
DEFAULT_MAX_MULT = 1.6
DEFAULT_EMA_ALPHA = 0.30

# Turnover governor thresholds (Mode A: freeze_then_ema)
# Both thresholds must be satisfied to freeze. Effective freeze level =
# min(GOV_THRESH_L1, GOV_THRESH_MAXDELTA) = 0.02 multiplier units.
DEFAULT_GOV_ENABLED = True
DEFAULT_GOV_THRESH_L1 = 0.04        # L1 norm threshold on day-over-day change
DEFAULT_GOV_THRESH_MAXDELTA = 0.02  # Max per-element change threshold


def enforce_constraints(
    raw_weights: np.ndarray,
    min_weight: float = MIN_WEIGHT
) -> np.ndarray:
    """
    Enforce tournament weight constraints.

    Three-step deterministic algorithm:
        1. Normalize raw weights to sum=1
        2. Clamp to minimum (may cause sum > 1)
        3. Re-normalize to sum=1

    Args:
        raw_weights: Unnormalized positive weights (e.g., bounded multipliers)
        min_weight: Minimum allowed weight (default 1e-5)

    Returns:
        Constrained weights: w_i >= min_weight, sum(w) = 1.0

    Raises:
        ValueError: If impossible constraint (N * min_weight > 1)
    """
    # Guard against degenerate case
    N = len(raw_weights)
    if N * min_weight > 1.0:
        raise ValueError(
            f"Impossible constraint: {N} weights × {min_weight} = "
            f"{N * min_weight} > 1.0"
        )

    # Step 1: Initial normalization
    weights = raw_weights / raw_weights.sum()

    # Step 2: Clamp to minimum (may cause sum > 1)
    weights = np.maximum(weights, min_weight)

    # Step 3: Re-normalize to exactly 1.0
    weights = weights / weights.sum()

    # Validate final constraints
    validate_weights(weights)

    return weights


def _ema_with_governor(
    targets: np.ndarray,
    alpha: float,
    thresh_l1: float,
    thresh_maxdelta: float,
) -> np.ndarray:
    """
    Sequential EMA with Mode A (freeze_then_ema) turnover governor.

    Processes the sequence of per-day target multipliers one at a time.
    On each day t >= 1, if the absolute change from the prior smoothed
    value is below BOTH thresholds, the value is frozen (no update).
    Otherwise a standard EMA step is applied.

    For this single-asset, single-scalar-per-day case:
        l1      = |targets[t] - smoothed[t-1]|   (same as maxdelta)
        maxdelta = |targets[t] - smoothed[t-1]|
        freeze  iff l1 < thresh_l1 AND maxdelta < thresh_maxdelta
                iff delta < min(thresh_l1, thresh_maxdelta)

    CAUSAL: smoothed[t] depends only on targets[0:t+1] and smoothed[0:t].

    Args:
        targets:         Array of shape (T,) — clipped multipliers [min_mult, max_mult].
        alpha:           EMA decay factor (0 < alpha <= 1).
        thresh_l1:       L1-norm freeze threshold.
        thresh_maxdelta: Max-element freeze threshold.

    Returns:
        smoothed: Array of shape (T,) — governor-smoothed multipliers.
    """
    T = len(targets)
    smoothed = np.empty(T, dtype=np.float64)
    smoothed[0] = targets[0]  # Day 0: initialize to raw target

    for t in range(1, T):
        delta = abs(targets[t] - smoothed[t - 1])
        freeze = (delta < thresh_l1) and (delta < thresh_maxdelta)
        if freeze:
            smoothed[t] = smoothed[t - 1]
        else:
            smoothed[t] = alpha * targets[t] + (1.0 - alpha) * smoothed[t - 1]

    return smoothed


def compute_weights(
    df_window: pd.DataFrame,
    sensitivity: float = DEFAULT_SENSITIVITY,
    min_mult: float = DEFAULT_MIN_MULT,
    max_mult: float = DEFAULT_MAX_MULT,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    prob_col: str = 'prob_up',
    gov_enabled: bool = DEFAULT_GOV_ENABLED,
    gov_thresh_l1: float = DEFAULT_GOV_THRESH_L1,
    gov_thresh_maxdelta: float = DEFAULT_GOV_THRESH_MAXDELTA,
) -> pd.Series:
    """
    Convert CNN probabilities to tournament-compliant allocation weights.

    CRITICAL: This function enforces causality. All operations are element-wise
    or use causal sequential processing (governor loop or pandas.ewm()).

    VTS Allocation Logic:
        1. tilt = sensitivity × (P(up) - 0.5)
        2. multiplier = clip(1 + tilt, min_mult, max_mult)
        3. multiplier_smooth = EMA(multiplier, alpha) with optional governor
        4. weights = normalize(multiplier_smooth)  # Sum to 1
        5. weights = enforce_min_weight(weights)   # Clip and renormalize

    Turnover Governor (gov_enabled=True, default):
        Replaces vectorized EMA with _ema_with_governor(). On each day t >= 1,
        if |target[t] - smoothed[t-1]| < min(thresh_l1, thresh_maxdelta),
        the smoothed value is frozen (no EMA update). This damps day-to-day
        flip-flopping caused by noisy signals without slowing genuine trends.

    Args:
        df_window: 12-month DataFrame with columns:
                  - prob_up: CNN probability of price increase [0, 1]
                  - PriceUSD_coinmetrics: BTC price in USD
        sensitivity: Allocation sensitivity to probability signal (default 1.5)
        min_mult: Minimum allocation multiplier (default 0.7)
        max_mult: Maximum allocation multiplier (default 1.6)
        ema_alpha: EMA smoothing parameter for daily data (default 0.30)
        prob_col: Name of probability column (default 'prob_up')
        gov_enabled: Enable turnover governor (default True)
        gov_thresh_l1: L1-norm freeze threshold (default 0.04)
        gov_thresh_maxdelta: Max-element freeze threshold (default 0.02)

    Returns:
        Series of normalized allocation weights:
            - All weights >= MIN_WEIGHT
            - Sum exactly 1.0
            - Index matches df_window.index

    Raises:
        ValueError: If required columns missing
        AssertionError: If output weights violate constraints
    """
    # Validate input
    if prob_col not in df_window.columns:
        raise ValueError(f"Missing required column: {prob_col}")

    # Extract probabilities (neutral if missing)
    prob_up = df_window[prob_col].fillna(0.5)

    # Step 1: Generate tilt signal
    # CAUSAL: Element-wise operation, no lookahead
    #
    # Dynamic sensitivity: if regime_score column is present (added by
    # build_features_trend_ols), scale sensitivity per day:
    #   regime_score = +1 (risk_on):  factor = 1.5x  → lean in during bull
    #   regime_score =  0 (neutral):  factor = 1.0x  → baseline DCA tilt
    #   regime_score = -1 (risk_off): factor = 0.3x  → near-DCA in bear
    if 'regime_score' in df_window.columns:
        regime = df_window['regime_score'].fillna(0.0).values
        # Asymmetric scaling: stronger reduction in bears than boost in bulls
        regime_factor = np.where(
            regime >= 0,
            1.0 + 0.5 * regime,   # [1.0 → 1.5] for regime in [0, +1]
            1.0 + 0.7 * regime    # [0.3 → 1.0] for regime in [-1, 0]
        )
        effective_sensitivity = sensitivity * regime_factor
        tilt = pd.Series(
            effective_sensitivity * (prob_up.values - 0.5),
            index=df_window.index
        )
    else:
        # Backward-compatible: fixed sensitivity (e.g. neutral baseline)
        tilt = sensitivity * (prob_up - 0.5)

    # Step 2: Bounded multipliers
    # CRITICAL FIX: Use pandas .clip() to keep as Series (not NumPy array)
    # multiplier ∈ [min_mult, max_mult]
    # CAUSAL: Element-wise operation, no lookahead
    multipliers = (1.0 + tilt).clip(lower=min_mult, upper=max_mult)

    # Step 3: EMA smoothing with optional turnover governor
    # CAUSAL: both paths depend only on historical data
    if gov_enabled:
        # Mode A (freeze_then_ema): sequential loop, freezes on tiny changes
        raw_weights = _ema_with_governor(
            targets=multipliers.values,
            alpha=ema_alpha,
            thresh_l1=gov_thresh_l1,
            thresh_maxdelta=gov_thresh_maxdelta,
        )
    else:
        # Legacy path: vectorized pandas EMA (identical to prior behavior)
        raw_weights = multipliers.ewm(alpha=ema_alpha, adjust=False).mean().values

    # Step 4: Enforce tournament constraints
    # normalize → clamp to min_weight → renormalize
    weights = enforce_constraints(raw_weights, min_weight=MIN_WEIGHT)

    # Final validation
    assert abs(weights.sum() - 1.0) < TOLERANCE_RTOL, \
        f"Weight sum violation: {weights.sum():.10f}"
    assert (weights >= MIN_WEIGHT - TOLERANCE_ATOL).all(), \
        f"Min weight violation: {weights.min():.10f} < {MIN_WEIGHT}"

    # Return as Series with proper index
    return pd.Series(weights, index=df_window.index)


def compute_weights_day_by_day(
    df_window: pd.DataFrame,
    sensitivity: float = DEFAULT_SENSITIVITY,
    min_mult: float = DEFAULT_MIN_MULT,
    max_mult: float = DEFAULT_MAX_MULT,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    prob_col: str = 'prob_up'
) -> pd.Series:
    """
    Alternative implementation: Day-by-day processing for strict causality.

    This is the "safest" implementation - processes each day sequentially
    to guarantee w_t depends only on data from days [0, ..., t].

    NOTE: For the VTS allocation logic, this is equivalent to compute_weights()
    because all operations (tilt, clip, EMA) are inherently causal. This function
    is provided for validation and debugging purposes.

    Args:
        Same as compute_weights()

    Returns:
        Same as compute_weights()
    """
    weights = []
    multipliers_history = []

    for t in range(len(df_window)):
        # Extract historical data up to and including day t
        historical_data = df_window.iloc[:t+1]

        # Extract probability for day t
        prob_t = historical_data[prob_col].iloc[-1]
        if pd.isna(prob_t):
            prob_t = 0.5  # Neutral if missing

        # Generate tilt for day t
        tilt_t = sensitivity * (prob_t - 0.5)

        # Bounded multiplier for day t
        mult_t = np.clip(1.0 + tilt_t, min_mult, max_mult)

        # EMA smoothing using historical multipliers
        multipliers_history.append(mult_t)
        if t == 0:
            mult_smooth_t = mult_t
        else:
            # EMA: S_t = α*X_t + (1-α)*S_{t-1}
            mult_smooth_t = (
                ema_alpha * mult_t +
                (1 - ema_alpha) * multipliers_history[t-1]
            )
            multipliers_history[t] = mult_smooth_t

        weights.append(mult_smooth_t)

    # Convert to numpy array
    raw_weights = np.array(weights)

    # Enforce tournament constraints
    weights_final = enforce_constraints(raw_weights, min_weight=MIN_WEIGHT)

    # Return as Series
    return pd.Series(weights_final, index=df_window.index)


def uniform_weights(df_window: pd.DataFrame) -> pd.Series:
    """
    Generate uniform DCA weights (baseline).

    Args:
        df_window: 12-month DataFrame

    Returns:
        Series of uniform weights (1/N for all days)
    """
    N = len(df_window)
    weights = np.ones(N) / N

    return pd.Series(weights, index=df_window.index)
