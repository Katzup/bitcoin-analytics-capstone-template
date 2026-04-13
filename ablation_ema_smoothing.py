#!/usr/bin/env python3
"""
Ablation 3: EMA Smoothing Parameter Sweep

Tests whether smoothing can improve the "small wins, big losses" pattern.

Following ChatGPT's recommendation:
1. Run on Neutral baseline first (better signal than CNN)
2. Use 7-point grid: [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
3. If improvement found, optionally test best 1-2 alphas on CNN

Expected runtime: ~15-20 minutes (7 runs × 2-3 min each)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tournament_mode.evaluator import evaluate_rolling_windows, summarize_results
from tournament_mode.weights import compute_weights

print("=" * 80)
print("ABLATION 3: EMA SMOOTHING PARAMETER SWEEP")
print("=" * 80)

# Load data
print("\n📊 Loading tournament data...")
df = pd.read_parquet("https://raw.githubusercontent.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025/main/data/stacking_sats_data.parquet")
df = df.loc['2016-01-01':'2025-06-01']

print(f"   Data range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"   Total days: {len(df)}")

# Create neutral features
df_neutral = df.copy()
df_neutral['prob_up'] = 0.5

# EMA alpha grid (7 points spanning smooth → reactive)
ema_alphas = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]

print(f"\n🔧 Testing {len(ema_alphas)} EMA alpha values on Neutral baseline...")
print(f"   Grid: {ema_alphas}")
print(f"   Alpha interpretation:")
print(f"     - Low (0.05-0.20):  Very smooth, slow to react (dampens whipsaws)")
print(f"     - Medium (0.30):     Current baseline")
print(f"     - High (0.50-0.90):  Fast reaction, minimal smoothing")
print(f"\n   Expected runtime: ~15-20 minutes")

results_by_alpha = {}

for alpha in ema_alphas:
    print(f"\n{'=' * 80}")
    print(f"Testing ema_alpha = {alpha:.2f}")
    print(f"{'=' * 80}")

    # Create custom weight function with this alpha
    def compute_weights_custom(df_window):
        return compute_weights(
            df_window,
            prob_col='prob_up',
            sensitivity=1.5,
            min_mult=0.7,
            max_mult=1.6,
            ema_alpha=alpha  # Variable we're testing
        )

    # Evaluate
    results = evaluate_rolling_windows(
        df_neutral,
        model=None,
        window_days=365,
        step_days=1,
        weights_fn=compute_weights_custom,
        verbose=True
    )

    summary = summarize_results(results, decay=0.9)
    results_by_alpha[alpha] = summary

    print(f"\n📊 Results for alpha={alpha:.2f}:")
    print(f"   RW percentile: {summary['rw_spd_percentile']:.2f}%")
    print(f"   Win rate:      {summary['win_rate']:.2%}")
    print(f"   Mean pct:      {summary['mean_spd_percentile']:.2f}%")

# Summary comparison
print("\n" + "=" * 80)
print("EMA SMOOTHING ABLATION RESULTS")
print("=" * 80)

df_compare = pd.DataFrame(results_by_alpha).T
df_compare.index.name = 'ema_alpha'
print("\n" + df_compare.to_string())

# Analysis
best_alpha = df_compare['rw_spd_percentile'].idxmax()
best_rw = df_compare.loc[best_alpha, 'rw_spd_percentile']
baseline_rw = df_compare.loc[0.30, 'rw_spd_percentile']  # Current baseline

print(f"\n{'=' * 80}")
print("ANALYSIS")
print(f"{'=' * 80}")

print(f"\nBest EMA alpha:     {best_alpha:.2f}")
print(f"  RW percentile:    {best_rw:.2f}%")
print(f"  Win rate:         {df_compare.loc[best_alpha, 'win_rate']:.2%}")

print(f"\nBaseline (α=0.30):  {baseline_rw:.2f}%")
print(f"Improvement:        {best_rw - baseline_rw:.2f} pp")

# Interpret results
delta = best_rw - baseline_rw

print(f"\n{'=' * 80}")
print("INTERPRETATION")
print(f"{'=' * 80}")

if delta > 2.0:
    print("\n✅ SIGNIFICANT IMPROVEMENT")
    print(f"   Smoothing helps: +{delta:.1f} pp improvement")
    print(f"   Recommendation: Use alpha={best_alpha:.2f}")
    print(f"   Next step: Test this alpha on CNN signal too")
elif delta > 0.5:
    print("\n➡️  MODERATE IMPROVEMENT")
    print(f"   Smoothing helps slightly: +{delta:.1f} pp")
    print(f"   Recommendation: Use alpha={best_alpha:.2f}")
    print(f"   Impact: Marginal but positive")
elif delta > -0.5:
    print("\n⚠️  MINIMAL EFFECT")
    print(f"   Smoothing doesn't matter much: {delta:+.1f} pp")
    print(f"   Recommendation: Keep current alpha=0.30 (no benefit to change)")
    print(f"   Conclusion: Allocator tuning can't fix signal quality issue")
else:
    print("\n❌ SMOOTHING HURTS")
    print(f"   More smoothing reduces performance: {delta:.1f} pp")
    print(f"   Recommendation: Keep current alpha=0.30")
    print(f"   Conclusion: Fast reaction is better than dampening")

# Check if we should test on CNN
print(f"\n{'=' * 80}")
print("NEXT STEP RECOMMENDATION")
print(f"{'=' * 80}")

if delta > 1.0:
    print(f"\n✅ Test best alpha ({best_alpha:.2f}) on CNN signal")
    print(f"   Rationale: {delta:.1f} pp improvement suggests smoothing helps")
    print(f"   Expected: May improve CNN's 41.43% RW percentile")
else:
    print(f"\n❌ Skip testing on CNN signal")
    print(f"   Rationale: <1pp improvement on better baseline (Neutral)")
    print(f"   Won't meaningfully improve CNN (worse signal)")
    print(f"   Conclusion: Signal quality, not smoothing, is the core issue")

# Save results
output_path = PROJECT_ROOT / 'ablation_ema_results.csv'
df_compare.to_csv(output_path)
print(f"\n💾 Results saved to: {output_path.name}")

print("\n" + "=" * 80)
print("ABLATION COMPLETE")
print("=" * 80)
