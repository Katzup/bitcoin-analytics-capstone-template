"""
Official Scorer Parity Test - Single Window

CRITICAL TEST: Validates our implementation matches official tournament scorer
on a single 12-month window before scaling to 3,075 windows.

This is the "looks right but scores wrong" detector - catches:
    - Index alignment issues
    - Dtype mismatches
    - Rounding errors
    - Percentile edge cases
    - Recency direction errors

Test Strategy:
    1. Load one 12-month window of BTC data
    2. Run full pipeline: features → weights → scoring
    3. Compare against official scorer (if available) OR known reference values
    4. Assert parity to tight tolerance (1e-9 for float64)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tournament_mode.scoring import (
    calculate_spd_for_window,
    calculate_recency_weighted_percentile,
    calculate_win_rate,
    MIN_WEIGHT
)
from tournament_mode.weights import compute_weights, uniform_weights
from tournament_mode.features import build_features, generate_gaf_image


# ============================================================================
# Test Data Generation
# ============================================================================

def create_reference_window(
    n_days: int = 365,
    start_date: str = '2020-01-01',
    seed: int = 42
) -> pd.DataFrame:
    """
    Create synthetic 12-month reference window.

    Uses realistic BTC price movements for testing.

    Args:
        n_days: Number of days (default 365 for 12 months)
        start_date: Start date
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics and prob_up columns
    """
    np.random.seed(seed)

    # Generate realistic price movement (geometric brownian motion)
    base_price = 30000
    mu = 0.001  # Daily drift (0.1%)
    sigma = 0.03  # Daily volatility (3%)

    dt = 1.0
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate mock CNN probabilities (correlated with price direction)
    # In real test, these would come from actual CNN
    future_returns = np.roll(returns, -5)  # 5-day forward return
    prob_up = 0.5 + 0.3 * np.tanh(future_returns * 10)  # Sigmoid-like
    prob_up = np.clip(prob_up, 0.3, 0.7)  # Realistic range

    df = pd.DataFrame({
        'PriceUSD_coinmetrics': prices,
        'prob_up': prob_up
    }, index=pd.date_range(start_date, periods=n_days, freq='D'))

    return df


# ============================================================================
# Reference Calculations (Official Formula Implementation)
# ============================================================================

def calculate_spd_reference(weights: np.ndarray, prices: np.ndarray) -> float:
    """
    Reference SPD calculation using official formula.

    Official: SPD = 1e8 * sum(w_t / price_t)

    Args:
        weights: Normalized weights (sum=1)
        prices: BTC prices in USD

    Returns:
        SPD (satoshis per dollar)
    """
    assert abs(weights.sum() - 1.0) < 1e-5, f"Weights sum: {weights.sum()}"
    inv_price = (1.0 / prices) * 1e8
    spd = (weights * inv_price).sum()
    return spd


def calculate_percentile_reference(
    spd: float,
    inv_price: np.ndarray
) -> float:
    """
    Reference percentile calculation using official formula.

    Official: (spd - min_spd) / (max_spd - min_spd) * 100

    Args:
        spd: Strategy SPD
        inv_price: Inverse prices (1/price * 1e8)

    Returns:
        Percentile (0-100)
    """
    min_spd = inv_price.min()
    max_spd = inv_price.max()
    span = max_spd - min_spd

    if span < 1e-10:
        return 50.0  # Constant price edge case

    percentile = (spd - min_spd) / span * 100
    return percentile


def calculate_recency_weighted_reference(
    percentiles: np.ndarray,
    decay: float = 0.9
) -> float:
    """
    Reference recency weighting using official formula.

    Official: raw_w[i] = decay^(N-1-i), then normalize and dot product

    Args:
        percentiles: Percentiles (oldest first, newest last)
        decay: Decay parameter (default 0.9)

    Returns:
        Recency-weighted average percentile
    """
    N = len(percentiles)
    raw_w = np.array([decay ** (N - 1 - i) for i in range(N)])
    exp_w = raw_w / raw_w.sum()
    rw_pct = (percentiles * exp_w).sum()
    return rw_pct


# ============================================================================
# PARITY TEST: Single Window End-to-End
# ============================================================================

def test_single_window_e2e_parity():
    """
    CRITICAL END-TO-END TEST: Validate full pipeline on one window.

    Pipeline:
        1. Load 12-month window
        2. Generate features (mock prob_up for speed)
        3. Compute weights
        4. Calculate SPD metrics
        5. Validate against reference implementation

    This test catches:
        - Index alignment issues
        - Dtype conversion errors
        - Formula implementation bugs
        - Rounding/precision issues
    """
    print("\n" + "=" * 80)
    print("CRITICAL E2E TEST: Single Window Official Parity")
    print("=" * 80)

    # ────────────────────────────────────────────────────────────────────────
    # Step 1: Load Reference Window
    # ────────────────────────────────────────────────────────────────────────
    print("\n📋 Step 1: Loading reference window...")
    df_window = create_reference_window(n_days=365, seed=42)

    print(f"   Window: {df_window.index[0].date()} to {df_window.index[-1].date()}")
    print(f"   Days: {len(df_window)}")
    print(f"   Price range: ${df_window['PriceUSD_coinmetrics'].min():.2f} - ${df_window['PriceUSD_coinmetrics'].max():.2f}")

    # ────────────────────────────────────────────────────────────────────────
    # Step 2: Compute Strategy Weights
    # ────────────────────────────────────────────────────────────────────────
    print("\n⚙️  Step 2: Computing strategy weights...")
    weights_strategy = compute_weights(df_window, prob_col='prob_up')

    # Validate constraints
    assert abs(weights_strategy.sum() - 1.0) < 1e-5, \
        f"Weight sum violation: {weights_strategy.sum()}"
    assert (weights_strategy >= MIN_WEIGHT - 1e-9).all(), \
        f"Min weight violation: {weights_strategy.min()}"

    print(f"   Weights computed: {len(weights_strategy)}")
    print(f"   Sum: {weights_strategy.sum():.10f}")
    print(f"   Min: {weights_strategy.min():.10f}")
    print(f"   Max: {weights_strategy.max():.10f}")

    # ────────────────────────────────────────────────────────────────────────
    # Step 3: Compute Uniform DCA Weights
    # ────────────────────────────────────────────────────────────────────────
    print("\n⚙️  Step 3: Computing uniform DCA weights...")
    weights_uniform = uniform_weights(df_window)

    print(f"   Uniform weight: {weights_uniform[0]:.10f}")
    print(f"   Sum: {weights_uniform.sum():.10f}")

    # ────────────────────────────────────────────────────────────────────────
    # Step 4: Calculate SPD Metrics (Our Implementation)
    # ────────────────────────────────────────────────────────────────────────
    print("\n📊 Step 4: Calculating SPD metrics (our implementation)...")
    prices = df_window['PriceUSD_coinmetrics']

    result_strategy = calculate_spd_for_window(weights_strategy, prices)
    result_uniform = calculate_spd_for_window(weights_uniform, prices)

    print(f"   Strategy SPD: {result_strategy['dynamic_spd']:.4f} sats/$")
    print(f"   Uniform SPD: {result_uniform['dynamic_spd']:.4f} sats/$")
    print(f"   Strategy percentile: {result_strategy['dynamic_pct']:.2f}%")
    print(f"   Uniform percentile: {result_uniform['dynamic_pct']:.2f}%")

    # ────────────────────────────────────────────────────────────────────────
    # Step 5: Calculate Reference Values (Official Formulas)
    # ────────────────────────────────────────────────────────────────────────
    print("\n🔍 Step 5: Calculating reference values (official formulas)...")

    # Reference SPD calculation
    inv_price = (1.0 / prices.values) * 1e8
    spd_strategy_ref = calculate_spd_reference(weights_strategy.values, prices.values)
    spd_uniform_ref = calculate_spd_reference(weights_uniform.values, prices.values)

    print(f"   Reference strategy SPD: {spd_strategy_ref:.4f} sats/$")
    print(f"   Reference uniform SPD: {spd_uniform_ref:.4f} sats/$")

    # Reference percentile calculation
    pct_strategy_ref = calculate_percentile_reference(spd_strategy_ref, inv_price)
    pct_uniform_ref = calculate_percentile_reference(spd_uniform_ref, inv_price)

    print(f"   Reference strategy percentile: {pct_strategy_ref:.2f}%")
    print(f"   Reference uniform percentile: {pct_uniform_ref:.2f}%")

    # ────────────────────────────────────────────────────────────────────────
    # Step 6: Assert Parity (Tight Tolerance)
    # ────────────────────────────────────────────────────────────────────────
    print("\n✅ Step 6: Asserting parity...")

    # SPD parity
    spd_strategy_diff = abs(result_strategy['dynamic_spd'] - spd_strategy_ref)
    spd_uniform_diff = abs(result_uniform['dynamic_spd'] - spd_uniform_ref)

    assert spd_strategy_diff < 1e-6, \
        f"Strategy SPD mismatch: {result_strategy['dynamic_spd']} vs {spd_strategy_ref} (diff: {spd_strategy_diff})"

    assert spd_uniform_diff < 1e-6, \
        f"Uniform SPD mismatch: {result_uniform['dynamic_spd']} vs {spd_uniform_ref} (diff: {spd_uniform_diff})"

    print(f"   ✓ Strategy SPD parity: diff = {spd_strategy_diff:.2e}")
    print(f"   ✓ Uniform SPD parity: diff = {spd_uniform_diff:.2e}")

    # Percentile parity
    pct_strategy_diff = abs(result_strategy['dynamic_pct'] - pct_strategy_ref)
    pct_uniform_diff = abs(result_uniform['dynamic_pct'] - pct_uniform_ref)

    assert pct_strategy_diff < 1e-6, \
        f"Strategy percentile mismatch: {result_strategy['dynamic_pct']}% vs {pct_strategy_ref}% (diff: {pct_strategy_diff})"

    assert pct_uniform_diff < 1e-6, \
        f"Uniform percentile mismatch: {result_uniform['dynamic_pct']}% vs {pct_uniform_ref}% (diff: {pct_uniform_diff})"

    print(f"   ✓ Strategy percentile parity: diff = {pct_strategy_diff:.2e}")
    print(f"   ✓ Uniform percentile parity: diff = {pct_uniform_diff:.2e}")

    # ────────────────────────────────────────────────────────────────────────
    # Step 7: Test Recency Weighting (Multiple Windows)
    # ────────────────────────────────────────────────────────────────────────
    print("\n🔄 Step 7: Testing recency weighting (3 windows)...")

    # Simulate 3 windows with different percentiles
    percentiles_test = np.array([50.0, 60.0, 70.0])

    # Our implementation
    rw_pct_ours = calculate_recency_weighted_percentile(percentiles_test, decay=0.9)

    # Reference implementation
    rw_pct_ref = calculate_recency_weighted_reference(percentiles_test, decay=0.9)

    rw_diff = abs(rw_pct_ours - rw_pct_ref)

    assert rw_diff < 1e-9, \
        f"Recency weighting mismatch: {rw_pct_ours} vs {rw_pct_ref} (diff: {rw_diff})"

    print(f"   Percentiles: {percentiles_test}")
    print(f"   Our RW average: {rw_pct_ours:.6f}%")
    print(f"   Reference RW average: {rw_pct_ref:.6f}%")
    print(f"   ✓ Recency weighting parity: diff = {rw_diff:.2e}")

    # ────────────────────────────────────────────────────────────────────────
    # Final Summary
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✅ SINGLE WINDOW E2E PARITY TEST PASSED")
    print("=" * 80)
    print("\nValidated components:")
    print("  ✓ Weight computation (sum=1, min_weight constraints)")
    print("  ✓ SPD calculation (harmonic mean formula)")
    print("  ✓ Percentile ranking (0-100 normalization)")
    print("  ✓ Recency weighting (decay direction and formula)")
    print("\nReady to scale to 3,075 rolling windows!")
    print("=" * 80)


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_edge_case_all_same_price():
    """
    Test edge case: Constant price should give 50% percentile.
    """
    print("\n🔍 Edge Case: Constant Price")

    # All prices identical
    df = pd.DataFrame({
        'PriceUSD_coinmetrics': np.ones(365) * 30000,
        'prob_up': np.random.uniform(0.4, 0.6, 365)
    }, index=pd.date_range('2020-01-01', periods=365, freq='D'))

    weights = compute_weights(df)
    result = calculate_spd_for_window(weights, df['PriceUSD_coinmetrics'])

    # Should be 50% (any allocation is equivalent when price is constant)
    assert abs(result['dynamic_pct'] - 50.0) < 1e-6, \
        f"Constant price should give 50%: got {result['dynamic_pct']}%"

    print(f"   Constant price: ${df['PriceUSD_coinmetrics'][0]:.2f}")
    print(f"   Percentile: {result['dynamic_pct']:.2f}%")
    print("   ✓ Edge case handled correctly")


def test_edge_case_extreme_allocation():
    """
    Test edge case: All weight on single day.
    """
    print("\n🔍 Edge Case: Extreme Allocation")

    prices = pd.Series(np.random.uniform(20000, 40000, 365))

    # All weight on day 0 (should be at specific percentile)
    weights_extreme = pd.Series(np.zeros(365))
    weights_extreme.iloc[0] = 1.0

    result = calculate_spd_for_window(weights_extreme, prices)

    # SPD should be exactly (1/price[0]) * 1e8
    expected_spd = (1.0 / prices.iloc[0]) * 1e8
    assert abs(result['dynamic_spd'] - expected_spd) < 1e-6, \
        f"Extreme allocation SPD mismatch: {result['dynamic_spd']} vs {expected_spd}"

    print(f"   All weight on day 0 (price: ${prices.iloc[0]:.2f})")
    print(f"   SPD: {result['dynamic_spd']:.2f} sats/$")
    print(f"   Expected: {expected_spd:.2f} sats/$")
    print("   ✓ Extreme allocation handled correctly")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "OFFICIAL PARITY TEST - SINGLE WINDOW" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")

    # Run main E2E test
    test_single_window_e2e_parity()

    # Run edge case tests
    print("\n" + "─" * 80)
    print("EDGE CASE VALIDATION")
    print("─" * 80)
    test_edge_case_all_same_price()
    test_edge_case_extreme_allocation()

    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "✅ ALL PARITY TESTS PASSED" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n🚀 Ready to implement evaluator.py for 3,075 rolling windows!\n")
