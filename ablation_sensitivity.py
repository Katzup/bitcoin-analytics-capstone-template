#!/usr/bin/env python3
"""
Ablation 1: Test reduced sensitivity without retraining.
Goal: Determine if losses come from overbetting vs bad probabilities.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tournament_mode.evaluator import evaluate_rolling_windows, summarize_results

print("=" * 80)
print("ABLATION 1: SENSITIVITY REDUCTION")
print("=" * 80)

# Load pre-computed features from previous run
print("\n📊 Loading pre-computed features...")
df = pd.read_parquet("https://raw.githubusercontent.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025/main/data/stacking_sats_data.parquet")
df = df.loc['2016-01-01':'2025-06-01']

# Load the features we already generated (saved from previous run would be ideal, but we'll regenerate if needed)
# For speed, we'll use the CSV output weights which already have features baked in
# Actually, let me just load the detailed results and analyze the sensitivity impact

print("\n🔧 Testing multiple sensitivity values...")
print("   Current: sensitivity=1.5")
print("   Testing: sensitivity in [0.5, 0.8, 1.0, 1.2, 1.5]")

# We need to regenerate with different sensitivities
# Import what we need
import numpy as np
import torch
import json
from model import DeepTradingCNNClassifier
from calibration import TemperatureScaling
from tournament_mode.features import build_features
from tournament_mode.weights import compute_weights

# Set seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Add volume placeholder
if 'volume' not in df.columns:
    df['volume'] = 1.0

# Load model artifacts
MODEL_DIR = PROJECT_ROOT / 'models'
with open(MODEL_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

with open(MODEL_DIR / 'temperature.json', 'r') as f:
    temp_data = json.load(f)
    temperature = temp_data['temperature']

temp_scaler = TemperatureScaling()
temp_scaler.temperature.data = torch.tensor([temperature])

model = DeepTradingCNNClassifier()
model.load_state_dict(torch.load(MODEL_DIR / 'btc_cnn_2014_2015.pt', map_location='cpu'))
model.to('cpu')
model.eval()

# Generate features once (reuse for all sensitivity tests)
print("\n🔧 Generating features (reusing for all sensitivity tests)...")
features_only = build_features(
    df,
    model,
    temp_scaler=temp_scaler,
    lookback=metadata['lookback'],
    device='cpu',
    volume_col='volume'
)

df_features = df.copy()
df_features['prob_up'] = features_only['prob_up']

print(f"   ✅ Features ready: {df_features['prob_up'].notna().sum()} valid rows")

# Test different sensitivities
sensitivities = [0.5, 0.8, 1.0, 1.2, 1.5]
results_by_sensitivity = {}

for sens in sensitivities:
    print(f"\n📊 Testing sensitivity={sens}...")

    # Create custom weight function with this sensitivity
    def compute_weights_custom(df_window):
        from tournament_mode.weights import compute_weights
        return compute_weights(
            df_window,
            sensitivity=sens,
            min_mult=0.7,
            max_mult=1.6,
            ema_alpha=0.30
        )

    # Evaluate
    results = evaluate_rolling_windows(
        df_features,
        model=None,
        window_days=365,
        step_days=1,
        weights_fn=compute_weights_custom,
        verbose=False
    )

    summary = summarize_results(results, decay=0.9)
    results_by_sensitivity[sens] = summary

    print(f"   RW percentile: {summary['rw_spd_percentile']:.2f}%")
    print(f"   Win rate: {summary['win_rate']:.2%}")

# Summary comparison
print("\n" + "=" * 80)
print("SENSITIVITY ABLATION RESULTS")
print("=" * 80)

df_compare = pd.DataFrame(results_by_sensitivity).T
df_compare.index.name = 'sensitivity'
print("\n" + df_compare.to_string())

# Find best
best_sens = df_compare['rw_spd_percentile'].idxmax()
best_rw = df_compare.loc[best_sens, 'rw_spd_percentile']
baseline_rw = df_compare.loc[1.5, 'rw_spd_percentile']

print(f"\n📈 Best sensitivity: {best_sens}")
print(f"   RW percentile: {best_rw:.2f}%")
print(f"   Improvement over baseline: {best_rw - baseline_rw:.2f} pp")

# Save results
output_path = PROJECT_ROOT / 'ablation_sensitivity_results.csv'
df_compare.to_csv(output_path)
print(f"\n💾 Results saved to: {output_path.name}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if best_rw - baseline_rw > 5:
    print("✅ MAJOR IMPROVEMENT: Reducing sensitivity significantly helps")
    print(f"   Recommendation: Use sensitivity={best_sens} (gains {best_rw - baseline_rw:.1f} pp)")
    print("   Root cause: Allocator too aggressive (overbetting on signals)")
elif best_rw - baseline_rw > 2:
    print("✅ MODERATE IMPROVEMENT: Reducing sensitivity helps somewhat")
    print(f"   Recommendation: Use sensitivity={best_sens} (gains {best_rw - baseline_rw:.1f} pp)")
    print("   Root cause: Partial overbetting issue")
else:
    print("⚠️  MINIMAL IMPROVEMENT: Sensitivity adjustment doesn't help much")
    print("   Implication: Problem is CNN signal quality, not allocator aggressiveness")
    print("   Next step: Proceed to Ablation 2 (neutral prob control)")
