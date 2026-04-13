"""
Feature Generation Tests

Critical tests for causality and alignment in feature generation.

GOTCHA TESTS (from user requirements):
    1. Last-row modification test: Changing last day shouldn't affect earlier features
    2. Alignment test: features.index.equals(df.index) and len match
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tournament_mode.features import (
    generate_gaf_image,
    normalize_for_gaf,
    build_features
)
from model import DeepTradingCNNClassifier


def create_test_dataframe(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic test DataFrame with price data.

    Args:
        n_days: Number of days
        seed: Random seed for reproducibility

    Returns:
        DataFrame with PriceUSD_coinmetrics column
    """
    np.random.seed(seed)

    # Generate synthetic price data (random walk)
    base_price = 30000
    returns = np.random.normal(0, 0.02, n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'PriceUSD_coinmetrics': prices
    }, index=pd.date_range('2020-01-01', periods=n_days, freq='D'))

    return df


def create_mock_model(num_channels: int = 2, image_size: int = 90) -> DeepTradingCNNClassifier:
    """
    Create mock CNN model for testing.

    Args:
        num_channels: Number of input channels
        image_size: GAF image size

    Returns:
        Initialized model (random weights, not trained)
    """
    model = DeepTradingCNNClassifier(
        num_channels=num_channels,
        image_size=image_size,
        num_classes=2,
        dropout=0.5
    )
    model.eval()
    return model


# ============================================================================
# GOTCHA TEST #1: Last-Row Modification Test
# ============================================================================

def test_features_causality_last_row_modification():
    """
    CRITICAL TEST: Changing the last row must NOT affect earlier features.

    If features[t] depends on data from t+1 or later, this test will fail.

    This is the most important causality test - if this passes, we have
    strong evidence that no lookahead bias exists.
    """
    print("\n🔍 GOTCHA TEST #1: Last-Row Modification")

    # Create test data
    df_original = create_test_dataframe(n_days=200, seed=42)
    model = create_mock_model(num_channels=1, image_size=90)

    # Generate features on original data
    features_original = build_features(
        df_original,
        model,
        lookback=90,
        volume_col=None,  # Single channel for speed
        device='cpu'
    )

    # Modify last row drastically
    df_modified = df_original.copy()
    df_modified.iloc[-1, df_modified.columns.get_loc('PriceUSD_coinmetrics')] = 999999.0

    # Generate features on modified data
    features_modified = build_features(
        df_modified,
        model,
        lookback=90,
        volume_col=None,
        device='cpu'
    )

    # CRITICAL ASSERTION: First N-1 features must be identical
    # (Only last feature can differ, since it uses the modified data)
    prob_up_original = features_original['prob_up'].iloc[:-1]
    prob_up_modified = features_modified['prob_up'].iloc[:-1]

    # Remove NaN values (from lookback period)
    valid_mask = ~prob_up_original.isna()
    prob_up_original_valid = prob_up_original[valid_mask]
    prob_up_modified_valid = prob_up_modified[valid_mask]

    assert np.allclose(prob_up_original_valid, prob_up_modified_valid, atol=1e-6), \
        "CAUSALITY VIOLATION: Earlier features changed when last row was modified!"

    # Last feature SHOULD differ (uses modified data)
    assert not np.isclose(
        features_original['prob_up'].iloc[-1],
        features_modified['prob_up'].iloc[-1],
        atol=1e-6
    ), "Sanity check failed: Last feature should differ after modification"

    print("✅ Last-row modification test PASSED")
    print(f"   Verified {len(prob_up_original_valid)} features unchanged")
    print(f"   Last feature correctly differs")


# ============================================================================
# GOTCHA TEST #2: Alignment Test
# ============================================================================

def test_features_alignment():
    """
    CRITICAL TEST: Feature index must exactly match input DataFrame index.

    This ensures:
        1. No rows dropped or added during feature generation
        2. Index alignment preserved for joining with prices/weights
        3. Length consistency for all downstream operations
    """
    print("\n🔍 GOTCHA TEST #2: Alignment Test")

    # Create test data with specific date range
    df = create_test_dataframe(n_days=300, seed=123)
    model = create_mock_model(num_channels=1, image_size=90)

    # Generate features
    features = build_features(
        df,
        model,
        lookback=90,
        volume_col=None,
        device='cpu'
    )

    # CRITICAL ASSERTIONS: Perfect alignment required

    # Test 1: Index equality (exact same dates in same order)
    assert features.index.equals(df.index), \
        "ALIGNMENT FAILURE: Feature index doesn't match input index"

    # Test 2: Length equality
    assert len(features) == len(df), \
        f"LENGTH MISMATCH: features={len(features)}, input={len(df)}"

    # Test 3: Column existence
    assert 'prob_up' in features.columns, \
        "Missing required column: prob_up"

    # Test 4: NaN pattern (first lookback days should be NaN)
    lookback = 90
    assert features['prob_up'].iloc[:lookback].isna().all(), \
        "Lookback period should contain NaN values"

    # Test 5: Non-NaN values after lookback
    assert not features['prob_up'].iloc[lookback:].isna().all(), \
        "Features should exist after lookback period"

    print("✅ Alignment test PASSED")
    print(f"   Index match: ✓")
    print(f"   Length match: {len(features)} == {len(df)} ✓")
    print(f"   NaN lookback: {lookback} days ✓")
    print(f"   Features generated: {len(features) - lookback} days ✓")


# ============================================================================
# Additional Causality Tests
# ============================================================================

def test_gaf_normalization():
    """
    Test GAF normalization produces values in [-1, 1].
    """
    print("\n🔍 GAF Normalization Test")

    # Test case 1: Normal data
    data1 = np.array([10.0, 20.0, 15.0, 25.0, 30.0])
    normalized1 = normalize_for_gaf(data1)
    assert normalized1.min() >= -1.0, "Normalization below -1"
    assert normalized1.max() <= 1.0, "Normalization above 1"
    assert np.isclose(normalized1.min(), -1.0, atol=0.01), "Min should be near -1"
    assert np.isclose(normalized1.max(), 1.0, atol=0.01), "Max should be near 1"

    # Test case 2: Constant data (edge case)
    data2 = np.array([30.0, 30.0, 30.0, 30.0])
    normalized2 = normalize_for_gaf(data2)
    assert np.allclose(normalized2, 0.0), "Constant data should normalize to 0"

    print("✅ GAF normalization test PASSED")


def test_gaf_image_shape():
    """
    Test GAF image generation produces correct shape.
    """
    print("\n🔍 GAF Image Shape Test")

    lookback = 90
    price_window = np.random.uniform(20000, 40000, lookback)
    volume_window = np.random.uniform(1e6, 1e9, lookback)

    # Single channel (price only)
    gaf_single = generate_gaf_image(price_window, volume_window=None, image_size=lookback)
    assert gaf_single.shape == (1, lookback, lookback), \
        f"Single channel shape mismatch: {gaf_single.shape}"

    # Dual channel (price + volume)
    gaf_dual = generate_gaf_image(price_window, volume_window=volume_window, image_size=lookback)
    assert gaf_dual.shape == (2, lookback, lookback), \
        f"Dual channel shape mismatch: {gaf_dual.shape}"

    print("✅ GAF image shape test PASSED")


def test_features_reproducibility():
    """
    Test that same seed produces same features (determinism).
    """
    print("\n🔍 Reproducibility Test")

    # Create identical test data
    df1 = create_test_dataframe(n_days=150, seed=999)
    df2 = create_test_dataframe(n_days=150, seed=999)

    # Create identical models (same seed)
    torch.manual_seed(42)
    model1 = create_mock_model(num_channels=1, image_size=90)

    torch.manual_seed(42)
    model2 = create_mock_model(num_channels=1, image_size=90)

    # Generate features
    features1 = build_features(df1, model1, lookback=90, volume_col=None, device='cpu')
    features2 = build_features(df2, model2, lookback=90, volume_col=None, device='cpu')

    # Should be identical
    valid_mask = ~features1['prob_up'].isna()
    assert np.allclose(
        features1['prob_up'][valid_mask],
        features2['prob_up'][valid_mask],
        atol=1e-6
    ), "Non-reproducible features with same seed"

    print("✅ Reproducibility test PASSED")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TOURNAMENT FEATURE GENERATION TESTS")
    print("=" * 70)

    # Run critical gotcha tests first
    test_features_causality_last_row_modification()
    test_features_alignment()

    # Run additional validation tests
    test_gaf_normalization()
    test_gaf_image_shape()
    test_features_reproducibility()

    print("\n" + "=" * 70)
    print("✅ ALL FEATURE TESTS PASSED")
    print("=" * 70)
