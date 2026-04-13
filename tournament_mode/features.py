"""
Tournament Feature Generation Module

Wraps VTS CNN components for tournament-compliant feature generation.

CRITICAL CAUSALITY REQUIREMENT:
    Features for day t must ONLY use data from days [0, ..., t]
    No lookahead allowed - validated via last-row modification test

VTS Components Used:
    - EnhancedBinaryDataset: GAF image generation from price history
    - DeepTradingCNNClassifier: Binary classification CNN
    - TemperatureScaling: Probability calibration (optional)

Tournament Feature Schema:
    - prob_up: CNN probability of price increase [0, 1]
    - (Optional) logits_raw: Raw model logits before calibration
    - (Optional) prob_calibrated: Temperature-scaled probabilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from pathlib import Path

# Import VTS components (assuming they're in parent directory)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import EnhancedBinaryDataset
from model import DeepTradingCNNClassifier
from calibration import TemperatureScaling
from pyts.image import GramianAngularField


# Default parameters (from VTS config_multi_timeframe BTC_1D)
DEFAULT_LOOKBACK = 90  # 90 days for GAF image
DEFAULT_IMAGE_SIZE = 90
DEFAULT_NUM_CHANNELS = 2  # price + volume (can expand with indicators)


def load_model(
    model_path: str,
    num_channels: int = DEFAULT_NUM_CHANNELS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: str = 'cpu'
) -> DeepTradingCNNClassifier:
    """
    Load pre-trained CNN model for tournament.

    Args:
        model_path: Path to saved model checkpoint (.pth file)
        num_channels: Number of input channels (default 2: price + volume)
        image_size: GAF image size (default 90)
        device: 'cpu' or 'cuda'

    Returns:
        Loaded CNN model in eval mode

    Raises:
        FileNotFoundError: If model checkpoint doesn't exist
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Initialize model architecture
    model = DeepTradingCNNClassifier(
        num_channels=num_channels,
        image_size=image_size,
        num_classes=2,  # Binary classification
        dropout=0.5
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Set to eval mode (disable dropout, batchnorm updates)
    model.eval()
    model.to(device)

    print(f"✅ Loaded model from {model_path}")
    print(f"   Channels: {num_channels}, Image size: {image_size}")

    return model


def generate_gaf_image(
    price_window: np.ndarray,
    volume_window: Optional[np.ndarray] = None,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """
    Generate multi-channel GAF image from price history.

    CRITICAL: This function is CAUSAL by construction.
    Input window contains only historical data (no future information).

    Args:
        price_window: Historical prices (lookback-length array)
        volume_window: Optional historical volume (same length)
        image_size: Output GAF image size (default 90)

    Returns:
        GAF image of shape (num_channels, image_size, image_size)
    """
    gaf = GramianAngularField(image_size=image_size, method='summation')
    channels = []

    # Process price channel
    price_normalized = normalize_for_gaf(price_window)
    gaf_price = gaf.transform(price_normalized.reshape(1, -1))
    channels.append(gaf_price[0])

    # Process volume channel (if available)
    if volume_window is not None:
        volume_normalized = normalize_for_gaf(volume_window)
        gaf_volume = gaf.transform(volume_normalized.reshape(1, -1))
        channels.append(gaf_volume[0])

    # Stack channels: shape (num_channels, image_size, image_size)
    gaf_image = np.stack(channels, axis=0)

    return gaf_image


def normalize_for_gaf(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to [-1, 1] for GAF transformation.

    Args:
        data: Raw time series data

    Returns:
        Normalized data in [-1, 1]
    """
    data_min = data.min()
    data_max = data.max()

    if data_max > data_min:
        # Scale to [0, 1] then to [-1, 1]
        normalized = 2 * (data - data_min) / (data_max - data_min) - 1
    else:
        # Constant data → neutral normalization
        normalized = np.zeros_like(data)

    return normalized


def predict_probabilities(
    model: DeepTradingCNNClassifier,
    gaf_image: np.ndarray,
    temp_scaler: Optional[TemperatureScaling] = None,
    device: str = 'cpu'
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Run CNN inference to get P(up) probability.

    Args:
        model: Trained CNN model
        gaf_image: GAF image (num_channels, image_size, image_size)
        temp_scaler: Optional temperature scaling for calibration
        device: 'cpu' or 'cuda'

    Returns:
        Tuple of (prob_up, logits):
            - prob_up: Probability of price increase [0, 1]
            - logits: Raw model logits (if temp_scaler provided)
    """
    model.eval()

    with torch.no_grad():
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(gaf_image).float().unsqueeze(0).to(device)

        # Get logits from model
        logits = model(image_tensor)

        # Apply temperature scaling if available
        if temp_scaler is not None:
            temp_scaler.eval()
            probs = temp_scaler(logits)
        else:
            # Standard softmax
            probs = torch.softmax(logits, dim=1)

        # Extract P(up) for class 1
        prob_up = probs[0, 1].item()

    return prob_up, logits.cpu().numpy() if temp_scaler is not None else None


def build_features(
    df: pd.DataFrame,
    model: DeepTradingCNNClassifier,
    lookback: int = DEFAULT_LOOKBACK,
    temp_scaler: Optional[TemperatureScaling] = None,
    device: str = 'cpu',
    price_col: str = 'PriceUSD_coinmetrics',
    volume_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate tournament-compliant features from price data.

    CRITICAL CAUSALITY GUARANTEE:
        Features for day t use ONLY data from days [t-lookback, ..., t]
        First `lookback` rows will have NaN features (insufficient history)

    Args:
        df: Input DataFrame with columns:
            - PriceUSD_coinmetrics (or specified price_col): BTC price in USD
            - (Optional) volume: Trading volume
        model: Pre-trained CNN model
        lookback: Window size for GAF generation (default 90 days)
        temp_scaler: Optional temperature scaling for calibration
        device: 'cpu' or 'cuda'
        price_col: Name of price column (default 'PriceUSD_coinmetrics')
        volume_col: Optional name of volume column

    Returns:
        DataFrame with added feature columns:
            - prob_up: CNN probability of price increase [0, 1]
            - (If volume_col provided) uses 2-channel GAF
            - Index matches input df.index exactly

    Raises:
        ValueError: If required columns missing
        AssertionError: If output alignment fails
    """
    # Validate inputs
    if price_col not in df.columns:
        raise ValueError(f"Missing required column: {price_col}")

    if volume_col is not None and volume_col not in df.columns:
        raise ValueError(f"Volume column specified but not found: {volume_col}")

    # Initialize output DataFrame (aligned with input)
    features_df = pd.DataFrame(index=df.index)
    features_df['prob_up'] = np.nan  # NaN for insufficient history

    # Extract price and volume arrays
    prices = df[price_col].values
    volumes = df[volume_col].values if volume_col is not None else None

    # CAUSALITY-PRESERVING LOOP:
    # Process each day sequentially, using only historical data
    print(f"Generating features for {len(df)} days (lookback={lookback})...")

    for t in range(lookback, len(df)):
        # Extract historical window [t-lookback, ..., t]
        # CRITICAL: This window contains ONLY past data, no future leakage
        price_window = prices[t-lookback:t]
        volume_window = volumes[t-lookback:t] if volumes is not None else None

        # Generate GAF image from historical window
        gaf_image = generate_gaf_image(
            price_window,
            volume_window=volume_window,
            image_size=lookback  # Use lookback as image size
        )

        # Run CNN inference
        prob_up, _ = predict_probabilities(
            model,
            gaf_image,
            temp_scaler=temp_scaler,
            device=device
        )

        # Store feature for day t
        features_df.iloc[t, features_df.columns.get_loc('prob_up')] = prob_up

        # Progress indicator
        if (t - lookback + 1) % 100 == 0:
            print(f"  Processed {t - lookback + 1}/{len(df) - lookback} days...")

    # ALIGNMENT VALIDATION: Critical for tournament compliance
    assert features_df.index.equals(df.index), \
        "ALIGNMENT FAILURE: Feature index doesn't match input index"
    assert len(features_df) == len(df), \
        f"LENGTH MISMATCH: features={len(features_df)}, input={len(df)}"

    print(f"✅ Feature generation complete:")
    print(f"   Total days: {len(df)}")
    print(f"   With features: {len(df) - lookback}")
    print(f"   NaN (lookback): {lookback}")

    return features_df


def build_features_with_indicators(
    df: pd.DataFrame,
    model: DeepTradingCNNClassifier,
    lookback: int = DEFAULT_LOOKBACK,
    temp_scaler: Optional[TemperatureScaling] = None,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Build features with supplementary technical indicators.

    In addition to prob_up, adds:
        - price_to_ma20: Ratio of current price to 20-day MA
        - volatility_20d: 20-day rolling volatility

    CAUSAL: All indicators use expanding/rolling windows (no lookahead)

    Args:
        df: Input DataFrame with price data
        model: Pre-trained CNN model
        lookback: Window for GAF generation
        temp_scaler: Optional temperature calibration
        device: Compute device

    Returns:
        DataFrame with prob_up + technical indicators
    """
    # First generate CNN features
    features_df = build_features(
        df,
        model,
        lookback=lookback,
        temp_scaler=temp_scaler,
        device=device
    )

    # Add technical indicators (all causal - use only historical data)
    price = df['PriceUSD_coinmetrics']

    # 20-day moving average (NaN for first 20 days - causal)
    ma_20 = price.rolling(window=20, min_periods=20).mean()
    features_df['price_to_ma20'] = price / ma_20

    # 20-day volatility (NaN for first 20 days - causal)
    returns = price.pct_change()
    vol_20d = returns.rolling(window=20, min_periods=20).std()
    features_df['volatility_20d'] = vol_20d

    return features_df


# Convenience function for tournament notebooks
def construct_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tournament-required function signature.

    This is the exact function expected by tournament template.
    Loads model and generates features in one step.

    Args:
        df: DataFrame with columns ['PriceUSD_coinmetrics']

    Returns:
        DataFrame with added feature columns
    """
    # Load pre-trained model (path should be configured)
    model_path = Path(__file__).parent.parent / 'models' / 'tournament_cnn.pth'

    if not model_path.exists():
        raise FileNotFoundError(
            f"Tournament CNN model not found at {model_path}\n"
            f"Please run tournament_mode/train_tournament_cnn.py first"
        )

    model = load_model(str(model_path))

    # Generate features
    return build_features(df, model)
