"""
Quick Validation Test - Trained CNN + Tournament Pipeline

Validates end-to-end integration before full 3,075-window run:
1. Load trained CNN + temperature artifacts
2. Run single 12-month window through full pipeline
3. Assert all constraints and sanity checks

CRITICAL: This catches serialization/device/shape mismatches before notebook.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add parent for VTS imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import DeepTradingCNNClassifier
from calibration import TemperatureScaling
from tournament_mode.features import build_features
from tournament_mode.weights import compute_weights, MIN_WEIGHT
from tournament_mode.scoring import calculate_spd_for_window
from data_adapter import DataAdapter
from config_multi_timeframe import BarConfig
import json


# Paths to trained artifacts
MODEL_DIR = Path(__file__).parent.parent / 'models'
MODEL_PATH = MODEL_DIR / 'btc_cnn_2014_2015.pt'
TEMPERATURE_PATH = MODEL_DIR / 'temperature.json'
METADATA_PATH = MODEL_DIR / 'metadata.json'


def load_trained_model():
    """
    Load trained CNN + temperature scaler exactly as submission will.

    Returns:
        (model, temp_scaler, metadata)
    """
    print("=" * 60)
    print("LOADING TRAINED MODEL ARTIFACTS")
    print("=" * 60)

    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    print(f"Metadata:")
    print(f"  Training period: {metadata['train_start']} to {metadata['train_end']}")
    print(f"  Lookback: {metadata['lookback']} days")
    print(f"  Horizon: {metadata['horizon']} days")
    print(f"  Random seed: {metadata['random_seed']}")
    print(f"  Trained: {metadata['trained_at']}")

    # Load temperature and create temp_scaler
    with open(TEMPERATURE_PATH, 'r') as f:
        temp_data = json.load(f)
        temperature = temp_data['temperature']

    print(f"  Temperature: {temperature:.4f}")

    # Create temperature scaler with saved temperature
    temp_scaler = TemperatureScaling()
    temp_scaler.temperature.data = torch.tensor([temperature])

    # Load model weights
    model = DeepTradingCNNClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Temperature scaler created: T = {temp_scaler.temperature.item():.4f}")

    return model, temp_scaler, metadata


def load_test_window():
    """
    Load 12-month window of BTC data from tournament start.

    Returns:
        DataFrame with 365 days starting 2016-01-01
    """
    print("\n" + "=" * 60)
    print("LOADING TEST WINDOW (2016-01-01 + 365 days)")
    print("=" * 60)

    adapter = DataAdapter(ticker='BTC-USD', source='yfinance')
    bar_config = BarConfig(timespan="day", multiplier=1, bars_per_day=1)

    # Fetch with extra buffer for lookback (90 days before window)
    df = adapter.fetch_bars(
        bar_config=bar_config,
        start_date='2015-10-03',  # 90 days before 2016-01-01
        end_date='2017-01-01'     # ~365 days after 2016-01-01
    )

    # Ensure price column exists
    if 'close' in df.columns and 'PriceUSD_coinmetrics' not in df.columns:
        df['PriceUSD_coinmetrics'] = df['close']

    # Add volume column if missing (model expects 2 channels: price + volume)
    if 'volume' not in df.columns:
        df['volume'] = 1.0  # Placeholder - model was trained with this

    print(f"✅ Loaded {len(df)} days")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Price range: ${df['PriceUSD_coinmetrics'].min():.2f} - ${df['PriceUSD_coinmetrics'].max():.2f}")

    # Extract exactly 365-day window starting 2016-01-01
    # Make timezone-aware to match df.index
    window_start = pd.Timestamp('2016-01-01', tz='UTC')
    window_end = window_start + pd.Timedelta(days=365)

    mask = (df.index >= window_start) & (df.index < window_end)
    df_window = df[mask].copy()

    print(f"   Window extracted: {len(df_window)} days")
    print(f"   Window range: {df_window.index[0].date()} to {df_window.index[-1].date()}")

    return df, df_window


def validate_features(features_df, df_window, lookback=90):
    """
    Validate feature generation output.

    Assertions:
    - Index alignment
    - First lookback rows are NaN
    - prob_up range [0, 1] where not NaN
    - Not all ~0.5 (model should have signal)
    """
    print("\n" + "=" * 60)
    print("VALIDATING FEATURES")
    print("=" * 60)

    # Index alignment
    assert features_df.index.equals(df_window.index), "Feature index mismatch!"
    print("✅ Index alignment: features match df_window")

    # First lookback rows NaN
    assert features_df['prob_up'].iloc[:lookback].isna().all(), f"First {lookback} rows should be NaN!"
    print(f"✅ Causality: First {lookback} rows are NaN (lookback period)")

    # prob_up valid range
    prob_valid = features_df['prob_up'].iloc[lookback:]
    assert (prob_valid >= 0.0).all() and (prob_valid <= 1.0).all(), "prob_up outside [0, 1]!"
    print(f"✅ prob_up range: [{prob_valid.min():.4f}, {prob_valid.max():.4f}]")

    # Not all ~0.5 (model should have signal)
    prob_std = prob_valid.std()
    assert prob_std > 0.01, f"prob_up has no variance (std={prob_std:.4f})!"
    print(f"✅ Model signal: prob_up std = {prob_std:.4f} (not flat)")

    # Show distribution
    print(f"   prob_up distribution: mean={prob_valid.mean():.4f}, median={prob_valid.median():.4f}")


def validate_causality_last_row(df_full, df_window, model, temp_scaler, lookback=90):
    """
    GOTCHA TEST: Verify changing last row doesn't affect earlier features.

    This is the critical test from planning - ensures no future leakage.
    """
    print("\n" + "=" * 60)
    print("VALIDATING CAUSALITY (LAST-ROW MODIFICATION TEST)")
    print("=" * 60)

    # Generate features with original data
    features_original = build_features(
        df_window,
        model,
        temp_scaler=temp_scaler,
        lookback=lookback,
        volume_col='volume'
    )

    # Modify last row
    df_window_modified = df_window.copy()
    df_window_modified.iloc[-1, df_window_modified.columns.get_loc('PriceUSD_coinmetrics')] = 999999.0

    # Generate features with modified data
    features_modified = build_features(
        df_window_modified,
        model,
        temp_scaler=temp_scaler,
        lookback=lookback,
        volume_col='volume'
    )

    # First N-1 features must be identical
    n_to_check = len(features_original) - 1
    prob_up_orig = features_original['prob_up'].iloc[:n_to_check]
    prob_up_mod = features_modified['prob_up'].iloc[:n_to_check]

    # Allow for numerical precision
    max_diff = np.abs(prob_up_orig - prob_up_mod).max()

    assert max_diff < 1e-6, f"Causality violation! First {n_to_check} rows differ by {max_diff}"
    print(f"✅ Causality verified: First {n_to_check} features identical (max diff: {max_diff:.2e})")


def validate_weights(weights, min_weight=MIN_WEIGHT):
    """
    Validate weight computation output.

    Assertions:
    - All weights >= MIN_WEIGHT
    - Weights sum to 1.0 (within tolerance)
    - Proper length
    """
    print("\n" + "=" * 60)
    print("VALIDATING WEIGHTS")
    print("=" * 60)

    # Minimum weight constraint
    assert (weights >= min_weight - 1e-9).all(), f"Weight below minimum: {weights.min()}"
    print(f"✅ Min weight constraint: all >= {min_weight:.2e} (min={weights.min():.6f})")

    # Sum to 1.0
    weight_sum = weights.sum()
    assert abs(weight_sum - 1.0) < 1e-5, f"Weights sum to {weight_sum}, not 1.0"
    print(f"✅ Sum constraint: {weight_sum:.10f} (diff from 1.0: {abs(weight_sum-1.0):.2e})")

    # Show distribution
    print(f"   Weight distribution:")
    print(f"     Mean: {weights.mean():.6f}")
    print(f"     Median: {weights.median():.6f}")
    print(f"     Std: {weights.std():.6f}")
    print(f"     Min: {weights.min():.6f}, Max: {weights.max():.6f}")


def validate_scoring(result_strategy, result_uniform, prices):
    """
    Validate scoring output.

    Assertions:
    - All values finite (no inf/NaN)
    - SPD > 0
    - Percentile in [0, 100]
    """
    print("\n" + "=" * 60)
    print("VALIDATING SCORING")
    print("=" * 60)

    # Check all values finite
    for key, val in result_strategy.items():
        assert np.isfinite(val), f"Strategy {key} is not finite: {val}"

    for key, val in result_uniform.items():
        assert np.isfinite(val), f"Uniform {key} is not finite: {val}"

    print("✅ All scoring values finite (no inf/NaN)")

    # SPD > 0
    assert result_strategy['dynamic_spd'] > 0, f"Strategy SPD <= 0: {result_strategy['dynamic_spd']}"
    assert result_uniform['uniform_spd'] > 0, f"Uniform SPD <= 0: {result_uniform['uniform_spd']}"
    print(f"✅ SPD values positive:")
    print(f"   Strategy: {result_strategy['dynamic_spd']:.2f} sats/dollar")
    print(f"   Uniform:  {result_uniform['uniform_spd']:.2f} sats/dollar")

    # Percentile in [0, 100]
    assert 0 <= result_strategy['dynamic_pct'] <= 100, f"Percentile out of range: {result_strategy['dynamic_pct']}"
    assert 0 <= result_uniform['uniform_pct'] <= 100, f"Percentile out of range: {result_uniform['uniform_pct']}"
    print(f"✅ Percentiles in [0, 100]:")
    print(f"   Strategy: {result_strategy['dynamic_pct']:.2f}%")
    print(f"   Uniform:  {result_uniform['uniform_pct']:.2f}%")

    # Show comparison
    if result_strategy['dynamic_pct'] > result_uniform['uniform_pct']:
        delta = result_strategy['dynamic_pct'] - result_uniform['uniform_pct']
        print(f"   🎯 Strategy beats uniform by {delta:.2f} percentile points")
    else:
        delta = result_uniform['uniform_pct'] - result_strategy['dynamic_pct']
        print(f"   ⚠️  Uniform beats strategy by {delta:.2f} percentile points")


def main():
    """Run quick validation test."""
    print("\n" + "=" * 80)
    print("TOURNAMENT PIPELINE VALIDATION - SINGLE WINDOW TEST")
    print("=" * 80)
    print("Purpose: Validate trained CNN + pipeline integration before full run")
    print("=" * 80)

    # Step 1: Load trained model
    model, temp_scaler, metadata = load_trained_model()
    lookback = metadata['lookback']

    # Step 2: Load test window
    df_full, df_window = load_test_window()

    # Step 3: Generate features
    print("\n" + "=" * 60)
    print("GENERATING FEATURES")
    print("=" * 60)
    features_df = build_features(
        df_window,
        model,
        temp_scaler=temp_scaler,
        lookback=lookback,
        volume_col='volume'  # Use volume channel (model expects 2 channels)
    )
    print(f"✅ Features generated: {len(features_df)} rows")
    print(f"   Columns: {list(features_df.columns)}")

    # Step 4: Validate features
    validate_features(features_df, df_window, lookback=lookback)

    # Step 5: Validate causality (CRITICAL GOTCHA TEST)
    validate_causality_last_row(df_full, df_window, model, temp_scaler, lookback=lookback)

    # Step 6: Compute weights
    print("\n" + "=" * 60)
    print("COMPUTING WEIGHTS")
    print("=" * 60)
    weights = compute_weights(features_df, prob_col='prob_up')
    print(f"✅ Weights computed: {len(weights)} values")

    # Step 7: Validate weights
    validate_weights(weights)

    # Step 8: Score with official function
    print("\n" + "=" * 60)
    print("SCORING WINDOW")
    print("=" * 60)
    prices = df_window['PriceUSD_coinmetrics']

    result_strategy = calculate_spd_for_window(weights, prices)

    # Uniform weights for comparison
    uniform_weights = pd.Series(1.0 / len(df_window), index=df_window.index)
    result_uniform = calculate_spd_for_window(uniform_weights, prices)

    print(f"✅ Scoring complete")

    # Step 9: Validate scoring
    validate_scoring(result_strategy, result_uniform, prices)

    # Final summary
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE - ALL CHECKS PASSED")
    print("=" * 80)
    print("Pipeline is ready for:")
    print("  1. Submission notebook creation")
    print("  2. Full 3,075-window tournament evaluation")
    print("=" * 80)

    return {
        'model': model,
        'temp_scaler': temp_scaler,
        'metadata': metadata,
        'result_strategy': result_strategy,
        'result_uniform': result_uniform
    }


if __name__ == '__main__':
    results = main()
