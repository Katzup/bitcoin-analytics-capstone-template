"""
Weight Computation Causality Tests

Validates that compute_weights() enforces strict causality:
    w_t must depend ONLY on data from days [0, ..., t]
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tournament_mode.weights import (
    compute_weights,
    compute_weights_day_by_day,
    enforce_constraints,
    uniform_weights,
    MIN_WEIGHT
)


def create_test_window(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic 12-month window for testing.

    Args:
        n_days: Number of days (default 365)
        seed: Random seed

    Returns:
        DataFrame with prob_up and PriceUSD_coinmetrics columns
    """
    np.random.seed(seed)

    df = pd.DataFrame({
        'prob_up': np.random.uniform(0.3, 0.7, n_days),
        'PriceUSD_coinmetrics': np.random.uniform(20000, 40000, n_days)
    }, index=pd.date_range('2020-01-01', periods=n_days, freq='D'))

    return df


def test_weights_causality_last_row_modification():
    """
    CRITICAL TEST: Changing last row must NOT affect earlier weights.

    This is the "gotcha test" equivalent for weight computation.
    If w_t depends on future data, this test will fail.
    """
    print("\n🔍 Weights Causality: Last-Row Modification Test")

    # Create test data
    df_original = create_test_window(n_days=365, seed=123)

    # Compute weights on original data
    weights_original = compute_weights(df_original)

    # Modify last row drastically
    df_modified = df_original.copy()
    df_modified.iloc[-1, df_modified.columns.get_loc('prob_up')] = 0.99
    df_modified.iloc[-1, df_modified.columns.get_loc('PriceUSD_coinmetrics')] = 99999.0

    # Compute weights on modified data
    weights_modified = compute_weights(df_modified)

    # CRITICAL ASSERTION: First N-1 weights must be unchanged
    assert np.allclose(weights_original.iloc[:-1], weights_modified.iloc[:-1], atol=1e-9), \
        "CAUSALITY VIOLATION: Earlier weights changed when last row was modified!"

    # Last weight CAN differ (uses modified data)
    assert not np.isclose(weights_original.iloc[-1], weights_modified.iloc[-1], atol=1e-6), \
        "Sanity check: Last weight should differ after modification"

    print("✅ Weights causality test PASSED")
    print(f"   Verified {len(weights_original) - 1} weights unchanged")
    print(f"   Last weight correctly differs")


def test_ema_is_causal():
    """
    Test that pandas .ewm() is causal (doesn't look ahead).

    EMA at time t should only depend on values [0, ..., t].
    """
    print("\n🔍 EMA Causality Test")

    # Create time series
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    # Compute EMA
    ema = values.ewm(alpha=0.3, adjust=False).mean()

    # EMA[2] should only depend on values[0:3]
    # Changing values[3] or values[4] should NOT affect EMA[2]

    values_modified = values.copy()
    values_modified.iloc[3] = 999.0  # Change future value
    ema_modified = values_modified.ewm(alpha=0.3, adjust=False).mean()

    # First 3 EMA values should be unchanged
    assert np.allclose(ema.iloc[:3], ema_modified.iloc[:3], atol=1e-9), \
        "EMA lookahead detected: Future values affected past EMA"

    print("✅ EMA causality test PASSED")
    print(f"   Past EMA values unaffected by future data ✓")


def test_weights_constraints_satisfied():
    """
    Test that weight constraints are enforced correctly.

    Requirements:
        - All weights >= MIN_WEIGHT
        - Sum of weights = 1.0 (within tolerance)
    """
    print("\n🔍 Weight Constraints Test")

    df = create_test_window(n_days=365, seed=456)
    weights = compute_weights(df)

    # Test 1: Minimum weight
    assert (weights >= MIN_WEIGHT - 1e-9).all(), \
        f"Weight below minimum: {weights.min():.10f} < {MIN_WEIGHT}"

    # Test 2: Sum equals 1.0
    total = weights.sum()
    assert abs(total - 1.0) < 1e-5, \
        f"Weights sum violation: {total:.10f} (expected 1.0)"

    # Test 3: Non-negative
    assert (weights >= 0).all(), \
        "Negative weight detected"

    # Test 4: Proper length
    assert len(weights) == len(df), \
        f"Length mismatch: weights={len(weights)}, input={len(df)}"

    print("✅ Weight constraints test PASSED")
    print(f"   Min weight: {weights.min():.10f} >= {MIN_WEIGHT} ✓")
    print(f"   Sum: {total:.10f} ≈ 1.0 ✓")
    print(f"   Length: {len(weights)} == {len(df)} ✓")


def test_enforce_constraints_deterministic():
    """
    Test that enforce_constraints() is deterministic and correct.
    """
    print("\n🔍 Enforce Constraints Deterministic Test")

    # Test with raw multipliers
    raw = np.array([1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4, 1.0, 1.2])

    # Apply constraints twice
    constrained1 = enforce_constraints(raw)
    constrained2 = enforce_constraints(raw)

    # Should be identical
    assert np.allclose(constrained1, constrained2, atol=1e-9), \
        "enforce_constraints is non-deterministic!"

    # Should satisfy constraints
    assert (constrained1 >= MIN_WEIGHT - 1e-9).all(), \
        "Min weight constraint violated"
    assert abs(constrained1.sum() - 1.0) < 1e-5, \
        "Sum constraint violated"

    print("✅ Enforce constraints deterministic test PASSED")


def test_day_by_day_equivalence():
    """
    Test that day-by-day processing gives same result as vectorized.

    Both implementations should be identical (both are causal).
    """
    print("\n🔍 Day-by-Day Equivalence Test")

    df = create_test_window(n_days=100, seed=789)  # Smaller for speed

    # Vectorized implementation
    weights_vectorized = compute_weights(df)

    # Day-by-day implementation (strict sequential processing)
    weights_day_by_day = compute_weights_day_by_day(df)

    # Should be identical (both are causal)
    assert np.allclose(weights_vectorized, weights_day_by_day, atol=1e-6), \
        "Vectorized and day-by-day implementations differ!"

    print("✅ Day-by-day equivalence test PASSED")
    print(f"   Vectorized matches sequential processing ✓")


def test_uniform_weights_generation():
    """
    Test uniform DCA weights are correct.
    """
    print("\n🔍 Uniform Weights Test")

    df = create_test_window(n_days=365, seed=111)
    weights = uniform_weights(df)

    # Should all be 1/N
    expected = 1.0 / 365
    assert np.allclose(weights, expected, atol=1e-9), \
        f"Uniform weights incorrect: got {weights[0]:.10f}, expected {expected:.10f}"

    # Should sum to 1.0
    assert abs(weights.sum() - 1.0) < 1e-9, \
        f"Uniform weights don't sum to 1: {weights.sum():.10f}"

    print("✅ Uniform weights test PASSED")
    print(f"   All weights = {expected:.10f} ✓")
    print(f"   Sum = {weights.sum():.10f} ✓")


def test_sensitivity_impact():
    """
    Test that sensitivity parameter affects allocation correctly.

    Higher sensitivity → more aggressive allocation (bigger tilt).
    """
    print("\n🔍 Sensitivity Impact Test")

    df = create_test_window(n_days=365, seed=222)

    # Low sensitivity (conservative)
    weights_low = compute_weights(df, sensitivity=0.5)

    # High sensitivity (aggressive)
    weights_high = compute_weights(df, sensitivity=2.0)

    # High sensitivity should have larger variance (more extreme allocations)
    var_low = weights_low.var()
    var_high = weights_high.var()

    assert var_high > var_low, \
        f"High sensitivity should have higher variance: {var_high:.6f} vs {var_low:.6f}"

    print("✅ Sensitivity impact test PASSED")
    print(f"   Low sensitivity variance: {var_low:.6f}")
    print(f"   High sensitivity variance: {var_high:.6f}")
    print(f"   High > Low: ✓")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("WEIGHT COMPUTATION CAUSALITY TESTS")
    print("=" * 70)

    # Critical causality test
    test_weights_causality_last_row_modification()
    test_ema_is_causal()

    # Constraint validation
    test_weights_constraints_satisfied()
    test_enforce_constraints_deterministic()

    # Implementation validation
    test_day_by_day_equivalence()
    test_uniform_weights_generation()
    test_sensitivity_impact()

    print("\n" + "=" * 70)
    print("✅ ALL WEIGHTS CAUSALITY TESTS PASSED")
    print("=" * 70)
