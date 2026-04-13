#!/usr/bin/env python3
"""
Simplified Tournament Evaluation - No CNN Required

Based on ablation studies showing CNN signal equivalent to random (neutral),
this version uses simple technical indicators that achieve same or better
performance with 100x less complexity and execution time.

Performance comparison:
    CNN approach:        41.43% RW percentile, 54.32% win rate (10-15 min runtime)
    Neutral (constant):  41.94% RW percentile, 70.42% win rate (2-3 min runtime)
    Momentum (MA-based): ~42-45% RW percentile (estimated, 2-3 min runtime)

Key advantages:
    - No pre-trained model artifacts required
    - 5-10x faster execution (no GAF generation, no CNN inference)
    - More robust (no overfitting to historical data)
    - Simpler to understand and maintain
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tournament_mode.features_simplified import (
    build_features_neutral,
    build_features_momentum,
    build_features_ma_crossover,
    build_features_volatility_adjusted,
    build_features_trend_ols,
)
from tournament_mode.evaluator import evaluate_rolling_windows, summarize_results

# Configuration
RANDOM_SEED = 42
DATA_URL = "https://raw.githubusercontent.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025/main/data/stacking_sats_data.parquet"
BACKTEST_START = '2016-01-01'
BACKTEST_END = '2025-06-01'


def main():
    print("=" * 80)
    print("SIMPLIFIED TOURNAMENT EVALUATION (NO CNN)")
    print("=" * 80)

    # Set seed
    np.random.seed(RANDOM_SEED)
    print(f"\n✅ Deterministic setup (seed={RANDOM_SEED})")

    # Load data
    print(f"\n📊 Loading tournament data...")
    df = pd.read_parquet(DATA_URL)
    df = df.loc[BACKTEST_START:BACKTEST_END]

    print(f"   Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Total days: {len(df)}")
    print(f"   Price range: ${df['PriceUSD_coinmetrics'].min():.2f} - ${df['PriceUSD_coinmetrics'].max():.2f}")

    # Test all feature approaches
    approaches = {
        'Neutral (constant)': build_features_neutral,
        'Momentum (20-day MA)': build_features_momentum,
        'MA Crossover (50/200)': build_features_ma_crossover,
        'Volatility-Adjusted': build_features_volatility_adjusted,
        'OLS TrendScore + Regime': build_features_trend_ols,
    }

    results_all = {}

    for name, feature_fn in approaches.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {name}")
        print(f"{'=' * 80}")

        # Generate features (fast - no CNN!)
        print(f"\n🔧 Generating features...")
        df_features = df.copy()
        features_only = feature_fn(df)

        # Copy ALL feature columns (supports both prob_up-only and prob_up+regime_score)
        for col in features_only.columns:
            df_features[col] = features_only[col]

        valid_features = df_features['prob_up'].notna().sum()
        print(f"   ✅ Features generated: {valid_features} valid rows")

        # Evaluate
        print(f"\n🎯 Running rolling window evaluation...")
        print(f"   Window size: 365 days")
        print(f"   Expected windows: ~{len(df) - 365 + 1:,}")

        results = evaluate_rolling_windows(
            df_features,
            model=None,
            window_days=365,
            step_days=1,
            verbose=True
        )

        summary = summarize_results(results, decay=0.9)
        results_all[name] = summary

        # Print results
        print(f"\n📈 Results for {name}:")
        print(f"   RW SPD percentile: {summary['rw_spd_percentile']:.2f}%")
        print(f"   Win rate:          {summary['win_rate']:.2%}")
        print(f"   Mean percentile:   {summary['mean_spd_percentile']:.2f}%")
        print(f"   Windows:           {summary['n_windows']:.0f}")

    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: ALL APPROACHES")
    print("=" * 80)

    comparison = pd.DataFrame(results_all).T
    print("\n" + comparison.to_string())

    # Find best
    best_approach = comparison['rw_spd_percentile'].idxmax()
    best_rw = comparison.loc[best_approach, 'rw_spd_percentile']

    print(f"\n🏆 Best approach: {best_approach}")
    print(f"   RW percentile: {best_rw:.2f}%")
    print(f"   Win rate: {comparison.loc[best_approach, 'win_rate']:.2%}")

    # Compare to original CNN
    cnn_rw = 41.43
    cnn_win = 0.5432

    print(f"\n📊 Comparison to CNN approach:")
    print(f"   CNN:           {cnn_rw:.2f}% RW, {cnn_win:.2%} win rate")
    print(f"   Best simple:   {best_rw:.2f}% RW, {comparison.loc[best_approach, 'win_rate']:.2%} win rate")
    print(f"   Improvement:   {best_rw - cnn_rw:.2f} pp RW, {comparison.loc[best_approach, 'win_rate'] - cnn_win:.2%} win rate")

    # Save results
    output_path = PROJECT_ROOT / 'tournament_results_simplified.csv'
    comparison.to_csv(output_path)
    print(f"\n💾 Results saved to: {output_path.name}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if best_rw > cnn_rw + 1:
        print(f"\n✅ SIMPLIFIED APPROACH IS BETTER")
        print(f"   Use '{best_approach}' instead of CNN")
        print(f"   Gains: {best_rw - cnn_rw:.1f} pp RW percentile")
        print(f"   Bonus: 5-10x faster execution, no model artifacts needed")
    elif best_rw > cnn_rw:
        print(f"\n✅ SIMPLIFIED APPROACH IS EQUIVALENT")
        print(f"   Use '{best_approach}' for simplicity")
        print(f"   Similar performance but much simpler")
    else:
        print(f"\n➡️  CNN APPROACH IS SLIGHTLY BETTER")
        print(f"   But difference is marginal ({cnn_rw - best_rw:.1f} pp)")
        print(f"   Consider simplified for robustness/speed")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
