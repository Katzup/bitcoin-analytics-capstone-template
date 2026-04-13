"""
Quick start script for Visual Trading System.

This is the fastest way to get started. Simply run:
    python quick_start.py

It will:
1. Generate sample data
2. Train a simple model
3. Show results

For production use, replace the sample data with your actual OHLC data.
"""

import torch
import pandas as pd
import numpy as np
import config
from dataset import StockImageDataset
from model import TradingCNN
from train import create_dataloaders, train


def generate_sample_data(num_days=400):
    """Generate synthetic price data for quick testing."""
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')

    # Trending price with noise
    trend = np.linspace(0, 0.25, num_days)
    noise = np.random.randn(num_days) * 0.015
    returns = trend + noise
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, num_days),
    })

    data.set_index('date', inplace=True)
    return data


def main():
    print("\n" + "=" * 80)
    print("VISUAL TRADING SYSTEM - QUICK START")
    print("=" * 80)

    # Configuration
    print(f"\n📋 Configuration:")
    print(f"   Lookback window: {config.LOOKBACK_WINDOW} days")
    print(f"   Prediction horizon: {config.PREDICTION_HORIZON} days")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    print(f"   Max epochs: {config.NUM_EPOCHS}")
    print(f"   Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"   Device: {config.DEVICE}")

    # Generate data
    print(f"\n📊 Generating sample data...")
    price_data = generate_sample_data(num_days=400)
    print(f"   Created {len(price_data)} days of OHLC data")

    # Create dataset
    print(f"\n🖼️  Creating image dataset...")
    dataset = StockImageDataset(
        price_data,
        lookback=config.LOOKBACK_WINDOW,
        horizon=config.PREDICTION_HORIZON
    )
    print(f"   Created {len(dataset)} training samples")
    print(f"   Image shape: {dataset.images.shape}")

    # Create dataloaders
    print(f"\n📦 Creating train/val/test splits...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config.BATCH_SIZE
    )
    print(f"   Training: {len(train_loader)} batches")
    print(f"   Validation: {len(val_loader)} batches")
    print(f"   Test: {len(test_loader)} batches")

    # Create model
    print(f"\n🧠 Creating model...")
    model = TradingCNN(num_channels=config.NUM_CHANNELS)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: TradingCNN")
    print(f"   Parameters: {total_params:,}")

    # Train
    print(f"\n🚀 Training model...")
    print(f"   (This may take a few minutes)")
    print()

    history = train(
        model,
        train_loader,
        val_loader,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE
    )

    # Results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n✅ Best model at epoch {history['best_epoch'] + 1}")
    print(f"✅ Best validation loss: {history['best_val_loss']:.6f}")
    print(f"\n📁 Model saved in: {config.CHECKPOINT_DIR}/")

    # Next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Check your saved model in the checkpoints/ directory

2. Load your own data:
   - From CSV: pd.read_csv('your_data.csv')
   - From PostgreSQL: Use load_from_postgres.py

3. Experiment with configurations in config.py:
   - Try LOOKBACK_WINDOW = 30 or 120
   - Try MODEL_TYPE = 'deep' for better performance
   - Adjust LEARNING_RATE if needed

4. Add technical indicators:
   - See MultiChannelStockDataset in dataset.py
   - Calculate RSI, MACD, Bollinger Bands

5. Backtest your model:
   - See backtest.py for framework
   - Integrate model predictions
   - Evaluate trading performance

6. For full workflow, see: example_usage.py
""")


if __name__ == '__main__':
    main()
