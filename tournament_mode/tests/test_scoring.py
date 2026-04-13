"""
Scoring Tests

Validates SPD calculation, percentile ranking, and tournament metrics
against official scorer formulas.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tournament_mode.scoring import (
    calculate_spd_for_window,
    calculate_recency_weighted_percentile,
    calculate_win_rate,
    calculate_tournament_metrics,
    MIN_WEIGHT,
    TOLERANCE
)


def test_spd_harmonic_calculation():
    """
    Test SPD uses harmonic mean (sum of w/p), not arithmetic (sum of w*p).

    Official formula: SPD = 1e8 * sum(w_t / price_t)
    """
    print("\n🔍 SPD Harmonic Calculation Test")

    # Simple test case: 2 days
    weights = pd.Series([0.5, 0.5])
    prices = pd.Series([20000.0, 40000.0])

    result = calculate_spd_for_window(weights, prices)

    # Expected (harmonic): (0.5/20k + 0.5/40k) * 1e8 = 3750 sats/$
    expected_spd = (0.5/20000 + 0.5/40000) * 1e8

    assert abs(result['dynamic_spd'] - expected_spd) < 1, \
        f"SPD mismatch: {result['dynamic_spd']} vs {expected_spd}"

    # WRONG (arithmetic would give): 1e8 / (0.5*20k + 0.5*40k) = 3333 sats/$
    wrong_spd = 1e8 / (0.5*20000 + 0.5*40000)
    assert abs(result['dynamic_spd'] - wrong_spd) > 100, \
        "SPD should NOT match arithmetic mean formula"

    print(f"✅ SPD harmonic test PASSED")
    print(f"   Correct (harmonic): {result['dynamic_spd']:.2f} sats/$")
    print(f"   Expected: {expected_spd:.2f} sats/$")
    print(f"   Wrong (arithmetic): {wrong_spd:.2f} sats/$")


def test_uniform_weights_equals_dca():
    """
    Reference equivalence: Uniform weights should match DCA exactly.

    For uniform weights w_t = 1/N:
        dynamic_spd should equal uniform_spd
        dynamic_pct should equal uniform_pct
    """
    print("\n🔍 Uniform Weights = DCA Test")

    N = 10
    weights = pd.Series(np.ones(N) / N)
    prices = pd.Series(np.random.uniform(20000, 40000, N))

    result = calculate_spd_for_window(weights, prices)

    # Should be identical
    assert abs(result['dynamic_spd'] - result['uniform_spd']) < 1e-6, \
        f"Uniform weights don't match DCA: {result['dynamic_spd']} vs {result['uniform_spd']}"

    assert abs(result['dynamic_pct'] - result['uniform_pct']) < 1e-6, \
        f"Percentiles don't match: {result['dynamic_pct']}% vs {result['uniform_pct']}%"

    print("✅ Uniform weights test PASSED")
    print(f"   Dynamic SPD: {result['dynamic_spd']:.2f}")
    print(f"   Uniform SPD: {result['uniform_spd']:.2f}")
    print(f"   Match: ✓")


def test_constant_price_gives_50_percentile():
    """
    Edge case: Constant price should give 50% percentile for any weights.

    When all prices are identical, min_spd = max_spd, span = 0.
    All strategies should be at the 50th percentile.
    """
    print("\n🔍 Constant Price Edge Case Test")

    weights = pd.Series([0.3, 0.4, 0.3])  # Arbitrary weights
    prices = pd.Series([30000.0, 30000.0, 30000.0])  # Constant

    result = calculate_spd_for_window(weights, prices)

    # Both percentiles should be 50%
    assert abs(result['dynamic_pct'] - 50.0) < 1e-6, \
        f"Dynamic percentile should be 50%: got {result['dynamic_pct']}"

    assert abs(result['uniform_pct'] - 50.0) < 1e-6, \
        f"Uniform percentile should be 50%: got {result['uniform_pct']}"

    print("✅ Constant price test PASSED")
    print(f"   Dynamic percentile: {result['dynamic_pct']:.2f}%")
    print(f"   Uniform percentile: {result['uniform_pct']:.2f}%")


def test_percentile_normalization():
    """
    Test percentile is properly normalized between 0-100.

    0% = all buys at worst (peak) price
    100% = all buys at best (bottom) price
    """
    print("\n🔍 Percentile Normalization Test")

    prices = pd.Series([20000, 30000, 40000, 25000, 35000])

    # Worst case: All weight on peak price (40000)
    weights_worst = pd.Series([0.0, 0.0, 1.0, 0.0, 0.0])
    result_worst = calculate_spd_for_window(weights_worst, prices)
    assert abs(result_worst['dynamic_pct'] - 0.0) < 1e-6, \
        f"All-in at peak should be 0%: got {result_worst['dynamic_pct']}"

    # Best case: All weight on bottom price (20000)
    weights_best = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0])
    result_best = calculate_spd_for_window(weights_best, prices)
    assert abs(result_best['dynamic_pct'] - 100.0) < 1e-6, \
        f"All-in at bottom should be 100%: got {result_best['dynamic_pct']}"

    print("✅ Percentile normalization test PASSED")
    print(f"   Worst case (all at peak): {result_worst['dynamic_pct']:.2f}%")
    print(f"   Best case (all at bottom): {result_best['dynamic_pct']:.2f}%")


def test_recency_weighting_direction():
    """
    Test recency weighting gives higher weight to newer windows.

    With decay=0.9:
        oldest (i=0): weight = 0.9^(N-1) = smallest
        newest (i=N-1): weight = 0.9^0 = 1.0 = largest
    """
    print("\n🔍 Recency Weighting Direction Test")

    # 3 windows with increasing percentiles
    percentiles = np.array([50.0, 60.0, 70.0])
    decay = 0.9

    rw_pct = calculate_recency_weighted_percentile(percentiles, decay=decay)

    # Manual calculation:
    # weights = [0.9^2, 0.9^1, 0.9^0] = [0.81, 0.90, 1.00]
    # normalized = [0.81, 0.90, 1.00] / 2.71 = [0.299, 0.332, 0.369]
    # rw_pct = 50*0.299 + 60*0.332 + 70*0.369 = 61.8

    expected = (50*0.81 + 60*0.90 + 70*1.00) / (0.81 + 0.90 + 1.00)

    assert abs(rw_pct - expected) < 1e-6, \
        f"Recency weighting mismatch: {rw_pct} vs {expected}"

    # RW average should be closer to newest value (70) than simple average (60)
    simple_avg = percentiles.mean()
    assert rw_pct > simple_avg, \
        f"RW average ({rw_pct:.2f}) should exceed simple average ({simple_avg:.2f})"

    print("✅ Recency weighting test PASSED")
    print(f"   Expected: {expected:.2f}%")
    print(f"   Calculated: {rw_pct:.2f}%")
    print(f"   Simple avg: {simple_avg:.2f}%")
    print(f"   RW > Simple: ✓ (newest weighted more)")


def test_win_rate_calculation():
    """
    Test win rate correctly counts windows beating DCA.
    """
    print("\n🔍 Win Rate Calculation Test")

    # Strategy beats DCA in 7 out of 10 windows
    dynamic = np.array([60, 55, 70, 45, 65, 75, 50, 80, 55, 60])
    uniform = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50])

    win_rate = calculate_win_rate(dynamic, uniform)

    # Count wins: 7 windows where dynamic > uniform
    expected_wins = 7
    expected_rate = 7 / 10

    assert abs(win_rate - expected_rate) < 1e-6, \
        f"Win rate mismatch: {win_rate} vs {expected_rate}"

    print("✅ Win rate test PASSED")
    print(f"   Wins: {expected_wins}/10")
    print(f"   Win rate: {win_rate:.2%}")


def test_tournament_metrics_complete():
    """
    Integration test: Calculate complete tournament metrics.
    """
    print("\n🔍 Complete Tournament Metrics Test")

    # Simulate 5 rolling windows
    dynamic_pct = np.array([55, 60, 65, 70, 75])
    uniform_pct = np.array([50, 50, 50, 50, 50])

    metrics = calculate_tournament_metrics(dynamic_pct, uniform_pct, decay=0.9)

    # Should have all required keys
    required_keys = ['RW_SPD_PCT', 'win_rate', 'primary_score', 'num_windows',
                     'avg_dynamic_pct', 'avg_uniform_pct']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"

    # Win rate should be 100% (beats DCA in all windows)
    assert abs(metrics['win_rate'] - 1.0) < 1e-6, \
        f"Win rate should be 100%: got {metrics['win_rate']}"

    # RW percentile should be > simple average (recency weight)
    assert metrics['RW_SPD_PCT'] > metrics['avg_dynamic_pct'], \
        "RW percentile should exceed simple average"

    # Primary score should equal RW_SPD_PCT
    assert metrics['primary_score'] == metrics['RW_SPD_PCT'], \
        "Primary score should match RW_SPD_PCT"

    print("✅ Complete metrics test PASSED")
    print(f"   RW SPD PCT: {metrics['RW_SPD_PCT']:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']:.2%}")
    print(f"   Windows: {metrics['num_windows']}")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TOURNAMENT SCORING TESTS")
    print("=" * 70)

    test_spd_harmonic_calculation()
    test_uniform_weights_equals_dca()
    test_constant_price_gives_50_percentile()
    test_percentile_normalization()
    test_recency_weighting_direction()
    test_win_rate_calculation()
    test_tournament_metrics_complete()

    print("\n" + "=" * 70)
    print("✅ ALL SCORING TESTS PASSED")
    print("=" * 70)
