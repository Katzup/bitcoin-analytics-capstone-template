"""
Tournament Mode Test Suite Runner

Runs all validation tests for tournament compliance.

Test Categories:
    1. Feature Generation (causality, alignment)
    2. Scoring (SPD, percentile, recency weighting)
    3. Weight Computation (causality, constraints)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test modules
from tournament_mode.tests import test_features
from tournament_mode.tests import test_scoring
from tournament_mode.tests import test_causality_weights


def run_all_tests():
    """
    Run complete tournament test suite.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "TOURNAMENT MODE VALIDATION SUITE")
    print("=" * 80)

    all_passed = True

    # ========================================================================
    # CATEGORY 1: Feature Generation Tests
    # ========================================================================
    print("\n" + "─" * 80)
    print("CATEGORY 1: FEATURE GENERATION")
    print("─" * 80)

    try:
        # Critical gotcha tests
        test_features.test_features_causality_last_row_modification()
        test_features.test_features_alignment()

        # Additional validation
        test_features.test_gaf_normalization()
        test_features.test_gaf_image_shape()
        test_features.test_features_reproducibility()

        print("\n✅ FEATURE GENERATION: ALL TESTS PASSED")

    except AssertionError as e:
        print(f"\n❌ FEATURE GENERATION: TEST FAILED")
        print(f"   Error: {e}")
        all_passed = False

    except Exception as e:
        print(f"\n❌ FEATURE GENERATION: UNEXPECTED ERROR")
        print(f"   Error: {e}")
        all_passed = False

    # ========================================================================
    # CATEGORY 2: Scoring Tests
    # ========================================================================
    print("\n" + "─" * 80)
    print("CATEGORY 2: SCORING CORRECTNESS")
    print("─" * 80)

    try:
        test_scoring.test_spd_harmonic_calculation()
        test_scoring.test_uniform_weights_equals_dca()
        test_scoring.test_constant_price_gives_50_percentile()
        test_scoring.test_percentile_normalization()
        test_scoring.test_recency_weighting_direction()
        test_scoring.test_win_rate_calculation()
        test_scoring.test_tournament_metrics_complete()

        print("\n✅ SCORING: ALL TESTS PASSED")

    except AssertionError as e:
        print(f"\n❌ SCORING: TEST FAILED")
        print(f"   Error: {e}")
        all_passed = False

    except Exception as e:
        print(f"\n❌ SCORING: UNEXPECTED ERROR")
        print(f"   Error: {e}")
        all_passed = False

    # ========================================================================
    # CATEGORY 3: Weight Computation Tests
    # ========================================================================
    print("\n" + "─" * 80)
    print("CATEGORY 3: WEIGHT COMPUTATION CAUSALITY")
    print("─" * 80)

    try:
        # Critical causality tests
        test_causality_weights.test_weights_causality_last_row_modification()
        test_causality_weights.test_ema_is_causal()

        # Constraint and implementation tests
        test_causality_weights.test_weights_constraints_satisfied()
        test_causality_weights.test_enforce_constraints_deterministic()
        test_causality_weights.test_day_by_day_equivalence()
        test_causality_weights.test_uniform_weights_generation()
        test_causality_weights.test_sensitivity_impact()

        print("\n✅ WEIGHT COMPUTATION: ALL TESTS PASSED")

    except AssertionError as e:
        print(f"\n❌ WEIGHT COMPUTATION: TEST FAILED")
        print(f"   Error: {e}")
        all_passed = False

    except Exception as e:
        print(f"\n❌ WEIGHT COMPUTATION: UNEXPECTED ERROR")
        print(f"   Error: {e}")
        all_passed = False

    # ========================================================================
    # CATEGORY 4: Evaluator Smoke Test
    # ========================================================================
    print("\n" + "─" * 80)
    print("CATEGORY 4: EVALUATOR SMOKE TEST")
    print("─" * 80)

    try:
        import pandas as pd
        import numpy as np
        from tournament_mode.evaluator import evaluate_rolling_windows, summarize_results

        print("\n🔍 Creating synthetic test data...")
        # Create 400 days of synthetic data (allows for 365-day windows)
        dates = pd.date_range("2020-01-01", periods=400, freq="D")
        prices = np.linspace(10000, 20000, len(dates))
        prob_up = np.random.uniform(0.4, 0.6, len(dates))  # Add prob_up to skip CNN

        df = pd.DataFrame({
            "PriceUSD_coinmetrics": prices,
            "prob_up": prob_up
        }, index=dates)

        print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"   Days: {len(df)}")

        # Evaluate with larger step size for speed (30 days instead of 1)
        print("\n🔍 Running rolling window evaluation (step=30 days)...")
        res = evaluate_rolling_windows(df, window_days=365, step_days=30, verbose=False)

        print(f"   Windows evaluated: {len(res)}")

        # Summarize
        summ = summarize_results(res)

        print(f"   RW SPD Percentile: {summ['rw_spd_percentile']:.2f}%")
        print(f"   Win Rate: {summ['win_rate']:.2%}")

        # Basic sanity checks
        assert len(res) > 0, "No windows evaluated"
        assert summ['n_windows'] > 0, "No windows in summary"

        # Check percentile is in valid range
        if not np.isnan(summ["rw_spd_percentile"]):
            assert 0.0 <= summ["rw_spd_percentile"] <= 100.0, \
                f"Percentile out of range: {summ['rw_spd_percentile']}"

        # Check win rate is in valid range
        if not np.isnan(summ["win_rate"]):
            assert 0.0 <= summ["win_rate"] <= 1.0, \
                f"Win rate out of range: {summ['win_rate']}"

        print("\n✅ EVALUATOR: SMOKE TEST PASSED")

    except AssertionError as e:
        print(f"\n❌ EVALUATOR: TEST FAILED")
        print(f"   Error: {e}")
        all_passed = False

    except Exception as e:
        print(f"\n❌ EVALUATOR: UNEXPECTED ERROR")
        print(f"   Error: {e}")
        all_passed = False

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    if all_passed:
        print(" " * 25 + "✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nTournament mode is VALIDATED and ready for submission.")
        print("\nKey validations:")
        print("  ✓ Causality enforced (no lookahead bias)")
        print("  ✓ Alignment correct (features match input index)")
        print("  ✓ Scoring matches official formulas (SPD, percentile, RW)")
        print("  ✓ Weight constraints satisfied (min_weight, sum=1)")
        print("=" * 80)
    else:
        print(" " * 25 + "❌ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease review error messages above and fix issues before proceeding.")
        print("=" * 80)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
