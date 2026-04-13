"""
Tournament CNN Pre-Training Script

Train CNN on 2014-2015 BTC data for tournament submission.

CRITICAL REQUIREMENTS:
1. Train ONLY on pre-2016 data (no lookahead bias)
2. Deterministic training (fixed seeds for reproducibility)
3. Save model weights, temperature calibration, and metadata
4. Validate model works before tournament run

Training Data: 2014-01-01 to 2015-12-31
Tournament Starts: 2016-01-01 (strict temporal separation)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path for VTS imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_adapter import DataAdapter
from dataset import EnhancedBinaryDataset
from model import DeepTradingCNNClassifier
from train_binary import create_dataloaders, train_binary_classifier
from calibration import calibrate_model
from config_multi_timeframe import BarConfig
import config

# Tournament constants
TRAIN_START = '2014-01-01'
TRAIN_END = '2015-12-31'
LOOKBACK = 90
HORIZON = 10
RANDOM_SEED = 42

# Model save paths
MODEL_DIR = Path(__file__).parent.parent / 'models'
MODEL_WEIGHTS_PATH = MODEL_DIR / 'btc_cnn_2014_2015.pt'
TEMPERATURE_PATH = MODEL_DIR / 'temperature.json'
METADATA_PATH = MODEL_DIR / 'metadata.json'


def set_deterministic_seeds(seed: int = RANDOM_SEED):
    """
    Set all random seeds for reproducible training.

    Tournament requirement: Same seed = same trained model
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"✅ Set deterministic seeds: {seed}")


def load_training_data():
    """
    Load BTC data for 2014-2015 (pre-tournament period).

    Returns:
        DataFrame with daily BTC prices
    """
    print(f"\n📊 Loading BTC data: {TRAIN_START} to {TRAIN_END}")

    adapter = DataAdapter(ticker='BTC-USD', source='yfinance')

    # Create BarConfig for daily data
    bar_config = BarConfig(
        timespan="day",
        multiplier=1,
        bars_per_day=1
    )

    # Fetch daily bars for training period
    df = adapter.fetch_bars(
        bar_config=bar_config,
        start_date=TRAIN_START,
        end_date=TRAIN_END
    )

    if df.empty:
        raise ValueError("No training data loaded!")

    # Ensure we have PriceUSD_coinmetrics column
    if 'close' in df.columns and 'PriceUSD_coinmetrics' not in df.columns:
        df['PriceUSD_coinmetrics'] = df['close']

    print(f"✅ Loaded {len(df)} days of BTC data")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Price range: ${df['PriceUSD_coinmetrics'].min():.2f} - ${df['PriceUSD_coinmetrics'].max():.2f}")

    return df


def prepare_training_dataset(df: pd.DataFrame):
    """
    Create EnhancedBinaryDataset for CNN training.

    Args:
        df: DataFrame with price data

    Returns:
        EnhancedBinaryDataset ready for training
    """
    print(f"\n🔨 Preparing GAF-based training dataset...")
    print(f"   Lookback: {LOOKBACK} days")
    print(f"   Horizon: {HORIZON} days")

    # Rename columns to match dataset expectations
    if 'PriceUSD_coinmetrics' in df.columns and 'close' not in df.columns:
        df = df.copy()
        df['close'] = df['PriceUSD_coinmetrics']

    # Add volume if missing (required by dataset)
    if 'volume' not in df.columns:
        df['volume'] = 1.0  # Placeholder if not available

    # Create dataset (without indicators for simpler training)
    dataset = EnhancedBinaryDataset(
        price_data=df,
        lookback=LOOKBACK,
        horizon=HORIZON,
        use_indicators=False  # Simpler: just price + volume channels
    )

    print(f"✅ Created dataset with {len(dataset)} samples")
    print(f"   Image shape: {dataset[0][0].shape}")

    return dataset


def train_model(dataset, num_epochs: int = 50, device: str = 'cpu'):
    """
    Train CNN classifier with deterministic behavior.

    Args:
        dataset: EnhancedBinaryDataset
        num_epochs: Number of training epochs
        device: 'cpu', 'cuda', or 'mps'

    Returns:
        Trained model
    """
    print(f"\n🚀 Training CNN classifier...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Device: {device}")

    # Create dataloaders - actual signature: create_dataloaders(dataset, batch_size=32)
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=32
    )

    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")

    # Initialize model
    model = DeepTradingCNNClassifier()

    # Train with deterministic settings - returns history dict
    history = train_binary_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        save_path=None  # Don't save intermediate checkpoints
    )

    # Extract loss lists from history
    train_losses = history['train_loss']
    val_losses = history['val_loss']

    # Validate training improved
    if train_losses[-1] >= train_losses[0] * 0.9:
        print(f"⚠️  Warning: Training loss did not decrease significantly")
        print(f"   Initial loss: {train_losses[0]:.4f}")
        print(f"   Final loss: {train_losses[-1]:.4f}")
    else:
        print(f"✅ Training successful")
        print(f"   Initial loss: {train_losses[0]:.4f}")
        print(f"   Final loss: {train_losses[-1]:.4f}")
        print(f"   Best val loss: {min(val_losses):.4f}")

    return model


def calibrate_temperature(model, dataset, device: str = 'cpu'):
    """
    Calibrate temperature for probability outputs.

    Temperature scaling improves calibration (Brier score).

    Args:
        model: Trained CNN (already on device)
        dataset: Validation dataset
        device: Device for calibration (model's device)

    Returns:
        Optimal temperature scalar
    """
    print(f"\n🌡️  Calibrating temperature...")
    print(f"   Model device: {device}")
    print(f"   Calibration will run on CPU (calibration.py requires CPU)")

    # Move model to CPU for calibration (calibration.py doesn't support MPS/CUDA)
    model_cpu = model.to('cpu')
    model_cpu.eval()

    # Use 15% of data for calibration
    val_size = int(len(dataset) * 0.15)
    val_indices = range(len(dataset) - val_size, len(dataset))

    val_subset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32)

    # Calibrate on CPU - returns (temp_scaler, temp_value)
    temp_scaler, temperature = calibrate_model(model_cpu, val_loader, device='cpu')

    # Move model back to original device
    model.to(device)

    print(f"✅ Optimal temperature: {temperature:.4f}")

    return temperature


def save_model_artifacts(model, temperature: float):
    """
    Save model weights, temperature, and metadata.

    Args:
        model: Trained CNN
        temperature: Calibrated temperature scalar
    """
    print(f"\n💾 Saving model artifacts...")

    # Create models directory if needed
    MODEL_DIR.mkdir(exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"   ✅ Weights: {MODEL_WEIGHTS_PATH}")

    # Save temperature
    with open(TEMPERATURE_PATH, 'w') as f:
        json.dump({'temperature': float(temperature)}, f, indent=2)
    print(f"   ✅ Temperature: {TEMPERATURE_PATH}")

    # Save metadata
    metadata = {
        'train_start': TRAIN_START,
        'train_end': TRAIN_END,
        'lookback': LOOKBACK,
        'horizon': HORIZON,
        'random_seed': RANDOM_SEED,
        'trained_at': datetime.now().isoformat(),
        'model_class': 'DeepTradingCNNClassifier',
        'temperature': float(temperature)
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata: {METADATA_PATH}")


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("TOURNAMENT CNN PRE-TRAINING")
    print("=" * 80)
    print(f"Training Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Tournament Starts: 2016-01-01 (no data leakage)")
    print(f"Random Seed: {RANDOM_SEED} (deterministic)")
    print("=" * 80)

    # Step 1: Set seeds for reproducibility
    set_deterministic_seeds(RANDOM_SEED)

    # Step 2: Load training data
    df = load_training_data()

    # Step 3: Prepare dataset
    dataset = prepare_training_dataset(df)

    # Step 4: Train model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = train_model(dataset, num_epochs=50, device=device)

    # Step 5: Calibrate temperature
    temperature = calibrate_temperature(model, dataset, device=device)

    # Step 6: Save artifacts
    save_model_artifacts(model, temperature)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model ready for tournament evaluation")
    print(f"Load with: torch.load('{MODEL_WEIGHTS_PATH}')")
    print("=" * 80)

    return {
        'model': model,
        'temperature': temperature,
        'metadata_path': str(METADATA_PATH)
    }


if __name__ == '__main__':
    results = main()
