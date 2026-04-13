"""
Tournament Scoring Module

Implements exact scoring formulas from official tournament template.
All functions verified against official scorer from:
https://github.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025

Key Metrics:
    - SPD (Satoshis Per Dollar): Purchasing power efficiency
    - Percentile: Normalized rank within window (0-100)
    - RW_SPD_PCT: Recency-weighted average percentile (PRIMARY metric)
    - Win Rate: Fraction of windows beating DCA (SECONDARY metric)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Tournament constants (from official template)
MIN_WEIGHT = 1e-5
TOLERANCE_RTOL = 1e-5
TOLERANCE_ATOL = 1e-8
DEFAULT_DECAY = 0.9


def calculate_spd_for_window(
    weights: pd.Series,
    prices: pd.Series
) -> Dict[str, float]:
    """
    Calculate SPD metrics for a single 12-month window.

    Formula (from official scorer):
        inv_price_t = (1.0 / price_t) * 1e8
        uniform_spd = mean(inv_price_t)
        dynamic_spd = sum(weight_t * inv_price_t)

    Args:
        weights: Normalized allocation weights (sum = 1.0)
        prices: BTC prices in USD

    Returns:
        dict with keys:
            - dynamic_spd: Strategy SPD
            - uniform_spd: DCA baseline SPD
            - dynamic_pct: Strategy percentile (0-100)
            - uniform_pct: DCA percentile (0-100)
            - min_spd: Worst-case SPD (all buys at peak price)
            - max_spd: Best-case SPD (all buys at bottom price)

    Raises:
        AssertionError: If weight constraints violated
        ValueError: If inputs invalid
    """
    # Validate inputs
    if len(weights) != len(prices):
        raise ValueError(
            f"Length mismatch: weights={len(weights)}, prices={len(prices)}"
        )

    if not np.isclose(weights.sum(), 1.0, rtol=TOLERANCE_RTOL, atol=TOLERANCE_ATOL):
        raise AssertionError(
            f"Weights must sum to 1.0: got {weights.sum():.10f}"
        )

    if (weights < 0).any():
        raise ValueError("Weights must be non-negative")

    if (prices <= 0).any():
        raise ValueError("Prices must be positive")

    # Step 1: Compute inv_price (satoshis per dollar)
    # Official formula: inv_price = (1.0 / price) * 1e8
    inv_price = (1.0 / prices) * 1e8

    # Step 2: Uniform DCA SPD (baseline)
    # Arithmetic mean because uniform weights imply w_t = 1/N
    uniform_spd = inv_price.mean()

    # Step 3: Strategy SPD
    # Official formula: dynamic_spd = sum(weight_t * inv_price_t)
    # Equivalent to: 1e8 * sum(weight_t / price_t)
    dynamic_spd = (weights.values * inv_price.values).sum()

    # Step 4: Percentile calculations
    # Normalize between worst-case (min SPD) and best-case (max SPD)
    min_spd = inv_price.min()
    max_spd = inv_price.max()
    span = max_spd - min_spd

    # Handle edge case: constant price (all days same price)
    if span == 0 or span < 1e-10:
        # All strategies perform identically (50th percentile)
        uniform_pct = 50.0
        dynamic_pct = 50.0
    else:
        uniform_pct = (uniform_spd - min_spd) / span * 100
        dynamic_pct = (dynamic_spd - min_spd) / span * 100

    return {
        'dynamic_spd': float(dynamic_spd),
        'uniform_spd': float(uniform_spd),
        'dynamic_pct': float(dynamic_pct),
        'uniform_pct': float(uniform_pct),
        'min_spd': float(min_spd),
        'max_spd': float(max_spd)
    }


def calculate_recency_weighted_percentile(
    percentiles: np.ndarray,
    decay: float = DEFAULT_DECAY
) -> float:
    """
    Calculate RW_SPD_PCT (PRIMARY tournament metric).

    Official formula:
        decay = 0.9
        raw_w[i] = decay ** (N - 1 - i)  # oldest=smallest, newest=1.0
        exp_w = raw_w / sum(raw_w)
        RW_SPD_PCT = sum(percentiles[i] * exp_w[i])

    Newest windows receive highest weight (exponential decay backward).

    Args:
        percentiles: Array of dynamic_pct values
                    (ordered chronologically: oldest first, newest last)
        decay: Exponential decay rate (default 0.9 from official template)

    Returns:
        Recency-weighted average percentile (RW_SPD_PCT)

    Example:
        percentiles = [50.0, 60.0, 70.0]  # 3 windows
        decay = 0.9

        raw_w = [0.9^2, 0.9^1, 0.9^0] = [0.81, 0.90, 1.00]
        exp_w = [0.81, 0.90, 1.00] / 2.71 = [0.299, 0.332, 0.369]
        RW = 50*0.299 + 60*0.332 + 70*0.369 = 61.8
    """
    if len(percentiles) == 0:
        raise ValueError("Cannot compute recency weighting for empty array")

    if not (0 < decay < 1):
        raise ValueError(f"Decay must be in (0, 1): got {decay}")

    N = len(percentiles)

    # Official formula: raw_w[i] = decay ** (N - 1 - i)
    # i=0 (oldest): weight = decay^(N-1) = smallest
    # i=N-1 (newest): weight = decay^0 = 1.0 = largest
    raw_w = np.array([decay ** (N - 1 - i) for i in range(N)])

    # Normalize to sum=1
    exp_w = raw_w / raw_w.sum()

    # Weighted average (equivalent to dot product)
    RW_SPD_PCT = (percentiles * exp_w).sum()

    return float(RW_SPD_PCT)


def calculate_win_rate(
    dynamic_percentiles: np.ndarray,
    uniform_percentiles: np.ndarray
) -> float:
    """
    Calculate pass_ratio (SECONDARY tournament metric).

    Official formula:
        wins = count windows where dynamic_pct > uniform_pct
        pass_ratio = wins / total_windows

    Args:
        dynamic_percentiles: Strategy percentiles
        uniform_percentiles: DCA baseline percentiles

    Returns:
        Win rate (fraction of windows where strategy beats DCA)

    Raises:
        ValueError: If arrays have different lengths
    """
    if len(dynamic_percentiles) != len(uniform_percentiles):
        raise ValueError(
            f"Length mismatch: strategy={len(dynamic_percentiles)}, "
            f"dca={len(uniform_percentiles)}"
        )

    if len(dynamic_percentiles) == 0:
        raise ValueError("Cannot compute win rate for empty arrays")

    # Count wins (strategy > DCA)
    wins = (dynamic_percentiles > uniform_percentiles).sum()

    # Pass ratio
    pass_ratio = wins / len(dynamic_percentiles)

    return float(pass_ratio)


def calculate_tournament_metrics(
    dynamic_percentiles: np.ndarray,
    uniform_percentiles: np.ndarray,
    decay: float = DEFAULT_DECAY
) -> Dict[str, float]:
    """
    Calculate complete tournament metrics for all rolling windows.

    Primary metric: RW_SPD_PCT (recency-weighted percentile)
    Secondary metric: Win rate (fraction beating DCA)

    Args:
        dynamic_percentiles: Strategy percentiles across all windows
        uniform_percentiles: DCA percentiles across all windows
        decay: Recency weighting decay parameter (default 0.9)

    Returns:
        dict with keys:
            - RW_SPD_PCT: Recency-weighted average percentile (PRIMARY)
            - win_rate: Fraction of windows beating DCA (SECONDARY)
            - primary_score: Alias for RW_SPD_PCT (main ranking metric)
            - num_windows: Total number of windows evaluated
            - avg_dynamic_pct: Simple average of strategy percentiles
            - avg_uniform_pct: Simple average of DCA percentiles
    """
    # Calculate primary metric (recency-weighted percentile)
    RW_SPD_PCT = calculate_recency_weighted_percentile(
        dynamic_percentiles,
        decay=decay
    )

    # Calculate secondary metric (win rate)
    win_rate = calculate_win_rate(dynamic_percentiles, uniform_percentiles)

    # Additional statistics
    avg_dynamic_pct = float(dynamic_percentiles.mean())
    avg_uniform_pct = float(uniform_percentiles.mean())

    return {
        'RW_SPD_PCT': RW_SPD_PCT,
        'win_rate': win_rate,
        'primary_score': RW_SPD_PCT,  # Alias for main ranking
        'num_windows': len(dynamic_percentiles),
        'avg_dynamic_pct': avg_dynamic_pct,
        'avg_uniform_pct': avg_uniform_pct
    }


def validate_weights(weights: np.ndarray) -> bool:
    """
    Validate tournament weight constraints.

    Constraints (from official template):
        - All weights >= MIN_WEIGHT (1e-5)
        - Sum of weights = 1.0 (within tolerance)

    Args:
        weights: Weight vector to validate

    Returns:
        True if valid

    Raises:
        AssertionError: If constraints violated
    """
    # Check minimum weight
    if (weights < MIN_WEIGHT - TOLERANCE_ATOL).any():
        raise AssertionError(
            f"Weight below minimum: {weights.min():.10f} < {MIN_WEIGHT}"
        )

    # Check sum equals 1.0 (using official tolerance)
    total = weights.sum()
    if not np.isclose(total, 1.0, rtol=TOLERANCE_RTOL, atol=TOLERANCE_ATOL):
        raise AssertionError(
            f"Weights must sum to 1.0: got {total:.10f}"
        )

    return True


# ============================================================================
# Simplified API for Evaluator
# ============================================================================

def calculate_spd(weights: pd.Series, prices: pd.Series) -> float:
    """
    Simplified SPD calculation for evaluator.

    Args:
        weights: Normalized allocation weights
        prices: BTC prices

    Returns:
        SPD value (satoshis per dollar)
    """
    result = calculate_spd_for_window(weights, prices)
    return result['dynamic_spd']


def _DEPRECATED_PLACEHOLDER_calculate_percentile(spd_strategy: float, spd_uniform: float) -> float:
    """
    ⚠️ DEPRECATED PLACEHOLDER - DO NOT USE ⚠️

    This function is INCORRECT and exists only for documentation purposes.

    Percentile calculation requires the full price series to compute min/max SPD.
    You MUST use calculate_spd_for_window() which returns both SPD and percentile correctly.

    Args:
        spd_strategy: Strategy SPD
        spd_uniform: Uniform SPD (unused)

    Returns:
        INCORRECT placeholder value - DO NOT USE THIS FUNCTION
    """
    raise NotImplementedError(
        "This is a deprecated placeholder. Use calculate_spd_for_window() instead."
    )


# Convenience constants for external use
TOLERANCE = {
    'rtol': TOLERANCE_RTOL,
    'atol': TOLERANCE_ATOL
}