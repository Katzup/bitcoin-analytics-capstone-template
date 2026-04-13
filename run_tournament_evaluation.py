#!/usr/bin/env python3
"""
Run full tournament evaluation: VTS vs DCA across all rolling windows.

This script:
1. Loads tournament data (2016-2025)
2. Loads pre-trained CNN model + calibration
3. Evaluates all rolling 12-month windows
4. Reports tournament metrics: RW percentile, win rate, distribution

Expected runtime: 10-20 minutes depending on CPU
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import DeepTradingCNNClassifier
from calibration import TemperatureScaling
from tournament_mode.features import build_features
from tournament_mode.evaluator import evaluate_rolling_windows, summarize_results

# Configuration
RANDOM_SEED = 42
DEVICE = 'cpu'
DATA_URL = "https://raw.githubusercontent.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025/main/data/stacking_sats_data.parquet"
BACKTEST_START = '2016-01-01'
BACKTEST_END = '2025-06-01'

# Model paths
MODEL_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'btc_cnn_2014_2015.pt'
TEMPERATURE_PATH = MODEL_DIR / 'temperature.json'
METADATA_PATH = MODEL_DIR / 'metadata.json'


def main():
    print("=" * 80)
    print("TOURNAMENT EVALUATION: VTS vs DCA")
    print("=" * 80)

    # Set seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print(f"\n✅ Deterministic setup (seed={RANDOM_SEED}, device={DEVICE})")

    # Load data
    print(f"\n📊 Loading tournament data from {DATA_URL}")
    df = pd.read_parquet(DATA_URL)
    df = df.loc[BACKTEST_START:BACKTEST_END]

    # Add volume placeholder if missing
    if 'volume' not in df.columns:
        df['volume'] = 1.0

    print(f"   Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Total days: {len(df)}")
    print(f"   Price range: ${df['PriceUSD_coinmetrics'].min():.2f} - ${df['PriceUSD_coinmetrics'].max():.2f}")

    # Load model artifacts
    print(f"\n🧠 Loading pre-trained CNN model from {MODEL_PATH}")

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    print(f"   Training period: {metadata['train_start']} to {metadata['train_end']}")
    print(f"   Lookback: {metadata['lookback']} days")

    with open(TEMPERATURE_PATH, 'r') as f:
        temp_data = json.load(f)
        temperature = temp_data['temperature']

    temp_scaler = TemperatureScaling()
    temp_scaler.temperature.data = torch.tensor([temperature])

    model = DeepTradingCNNClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Temperature: {temp_scaler.temperature.item():.4f}")

    # Generate features ONCE for entire dataset (more efficient)
    print(f"\n🔧 Generating features for entire dataset...")
    print(f"   (This will take 5-10 minutes - GAF generation + CNN inference)")

    features_only = build_features(
        df,
        model,
        temp_scaler=temp_scaler,
        lookback=metadata['lookback'],
        device=DEVICE,
        volume_col='volume'
    )

    # Merge features with original data (evaluator needs price column)
    df_features = df.copy()
    df_features['prob_up'] = features_only['prob_up']

    valid_features = df_features['prob_up'].notna().sum()
    print(f"   ✅ Features generated: {valid_features} valid rows")

    # Run rolling window evaluation
    print(f"\n🎯 Running rolling window evaluation...")
    print(f"   Window size: 365 days (12 months)")
    print(f"   Step size: 1 day")

    expected_windows = len(df) - 365 + 1
    print(f"   Expected windows: ~{expected_windows:,}")
    print(f"   (This will take 5-10 minutes)")

    results = evaluate_rolling_windows(
        df_features,
        model=None,  # Features already computed
        window_days=365,
        step_days=1,
        verbose=True
    )

    # Summarize results
    summary = summarize_results(results, decay=0.9)

    # Print tournament results
    print("\n" + "=" * 80)
    print("TOURNAMENT RESULTS")
    print("=" * 80)

    print(f"\n📈 Primary Metric (Recency-Weighted SPD Percentile):")
    print(f"   VTS Strategy: {summary['rw_spd_percentile']:.2f}%")
    print(f"   Target: >50% (beat DCA)")

    print(f"\n🏆 Secondary Metric (Win Rate):")
    print(f"   VTS wins: {summary['win_rate']:.2%} of windows")
    print(f"   Target: >50% (majority of windows beat DCA)")

    print(f"\n📊 Additional Metrics:")
    print(f"   Mean SPD percentile: {summary['mean_spd_percentile']:.2f}%")
    print(f"   Windows evaluated: {summary['n_windows']:.0f}")

    # Distribution analysis
    print(f"\n📉 Performance Distribution:")
    print(f"   Min percentile: {results['pct_strategy'].min():.2f}%")
    print(f"   25th percentile: {results['pct_strategy'].quantile(0.25):.2f}%")
    print(f"   Median: {results['pct_strategy'].median():.2f}%")
    print(f"   75th percentile: {results['pct_strategy'].quantile(0.75):.2f}%")
    print(f"   Max percentile: {results['pct_strategy'].max():.2f}%")

    # VTS vs DCA comparison
    vts_better = (results['pct_strategy'] > results['pct_uniform']).sum()
    vts_worse = (results['pct_strategy'] < results['pct_uniform']).sum()
    vts_tie = (results['pct_strategy'] == results['pct_uniform']).sum()

    print(f"\n⚔️  Window-by-Window Comparison:")
    print(f"   VTS > DCA: {vts_better} windows ({vts_better/len(results)*100:.1f}%)")
    print(f"   VTS < DCA: {vts_worse} windows ({vts_worse/len(results)*100:.1f}%)")
    print(f"   VTS = DCA: {vts_tie} windows ({vts_tie/len(results)*100:.1f}%)")

    # Sample windows
    print(f"\n📋 Sample Windows (first 5):")
    sample = results[['start', 'end', 'spd_strategy', 'spd_uniform', 'pct_strategy', 'pct_uniform']].head(5)
    print(sample.to_string(index=False))

    print(f"\n📋 Sample Windows (last 5):")
    sample = results[['start', 'end', 'spd_strategy', 'spd_uniform', 'pct_strategy', 'pct_uniform']].tail(5)
    print(sample.to_string(index=False))

    # Save detailed results
    output_path = PROJECT_ROOT / 'tournament_results_detailed.csv'
    results.to_csv(output_path, index=False)
    print(f"\n💾 Detailed results saved to: {output_path.name}")

    # Summary CSV
    summary_df = pd.DataFrame([summary])
    summary_path = PROJECT_ROOT / 'tournament_results_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Summary saved to: {summary_path.name}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
