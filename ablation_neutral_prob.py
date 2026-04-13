#!/usr/bin/env python3
"""
Ablation 2: Neutral probability control (prob_up = 0.5 constant).
Goal: Determine if CNN signal is helping or hurting.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tournament_mode.evaluator import evaluate_rolling_windows, summarize_results

print("=" * 80)
print("ABLATION 2: NEUTRAL PROBABILITY CONTROL")
print("=" * 80)

# Load data
print("\n📊 Loading tournament data...")
df = pd.read_parquet("https://raw.githubusercontent.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025/main/data/stacking_sats_data.parquet")
df = df.loc['2016-01-01':'2025-06-01']

print(f"   Data range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"   Total days: {len(df)}")

# Create neutral features (prob_up = 0.5 everywhere)
df_neutral = df.copy()
df_neutral['prob_up'] = 0.5  # Constant neutral probability

print("\n🔧 Testing neutral probabilities (prob_up = 0.5 constant)...")
print("   This simulates: 'What if CNN gave no signal at all?'")
print("   Expected result: ~50% percentile (uniform DCA-like)")

# Evaluate with current sensitivity (1.5)
print("\n📊 Evaluation 1: Neutral prob with sensitivity=1.5 (current)")
results_neutral_15 = evaluate_rolling_windows(
    df_neutral,
    model=None,
    window_days=365,
    step_days=1,
    verbose=False
)

summary_neutral_15 = summarize_results(results_neutral_15, decay=0.9)

print(f"   RW percentile: {summary_neutral_15['rw_spd_percentile']:.2f}%")
print(f"   Win rate: {summary_neutral_15['win_rate']:.2%}")

# Also test with reduced sensitivity for comparison
from tournament_mode.weights import compute_weights

def compute_weights_08(df_window):
    return compute_weights(
        df_window,
        sensitivity=0.8,
        min_mult=0.7,
        max_mult=1.6,
        ema_alpha=0.30
    )

print("\n📊 Evaluation 2: Neutral prob with sensitivity=0.8 (conservative)")
results_neutral_08 = evaluate_rolling_windows(
    df_neutral,
    model=None,
    window_days=365,
    step_days=1,
    weights_fn=compute_weights_08,
    verbose=False
)

summary_neutral_08 = summarize_results(results_neutral_08, decay=0.9)

print(f"   RW percentile: {summary_neutral_08['rw_spd_percentile']:.2f}%")
print(f"   Win rate: {summary_neutral_08['win_rate']:.2%}")

# Compare to actual CNN results
print("\n" + "=" * 80)
print("NEUTRAL PROBABILITY CONTROL RESULTS")
print("=" * 80)

comparison = pd.DataFrame({
    'CNN (sens=1.5)': {
        'rw_spd_percentile': 41.43,  # From previous run
        'win_rate': 0.5432
    },
    'Neutral (sens=1.5)': {
        'rw_spd_percentile': summary_neutral_15['rw_spd_percentile'],
        'win_rate': summary_neutral_15['win_rate']
    },
    'Neutral (sens=0.8)': {
        'rw_spd_percentile': summary_neutral_08['rw_spd_percentile'],
        'win_rate': summary_neutral_08['win_rate']
    }
}).T

print("\n" + comparison.to_string())

# Calculate deltas
neutral_best = max(summary_neutral_15['rw_spd_percentile'], summary_neutral_08['rw_spd_percentile'])
cnn_actual = 41.43

delta = neutral_best - cnn_actual

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"\nCNN actual:        {cnn_actual:.2f}%")
print(f"Neutral best:      {neutral_best:.2f}%")
print(f"Delta:             {delta:.2f} pp")

if delta > 5:
    print("\n❌ CNN SIGNAL IS HARMFUL")
    print(f"   Neutral probabilities beat CNN by {delta:.1f} pp")
    print("   Root cause: CNN predictions are WORSE than random")
    print("   Recommendation: MUST retrain CNN on broader/better data")
    print("   Priority: HIGH - current model actively hurts performance")
elif delta > 2:
    print("\n⚠️  CNN SIGNAL IS WEAK")
    print(f"   Neutral probabilities beat CNN by {delta:.1f} pp")
    print("   Root cause: CNN predictions add little value")
    print("   Recommendation: Retrain CNN or simplify to technical indicators")
    print("   Priority: MEDIUM - model provides marginal value")
elif delta > -2:
    print("\n➡️  CNN SIGNAL IS NEUTRAL")
    print(f"   CNN vs neutral: {abs(delta):.1f} pp difference (negligible)")
    print("   Root cause: CNN predictions neither help nor hurt significantly")
    print("   Recommendation: Focus on allocation tuning, not model retraining")
    print("   Priority: LOW - model is adequate, fix allocator instead")
else:
    print("\n✅ CNN SIGNAL IS VALUABLE")
    print(f"   CNN beats neutral by {abs(delta):.1f} pp")
    print("   Root cause: CNN provides useful signal, problem is elsewhere")
    print("   Recommendation: Keep model, tune allocation parameters")
    print("   Priority: Focus on sensitivity/smoothing, not retraining")

# Save results
output_path = PROJECT_ROOT / 'ablation_neutral_results.csv'
comparison.to_csv(output_path)
print(f"\n💾 Results saved to: {output_path.name}")
