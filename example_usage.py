"""
Example script demonstrating complete workflow of the visual trading system.

This script shows how to:
1. Load OHLC data
2. Create image dataset
3. Train a model
4. Backtest the model
5. Evaluate performance

For demonstration purposes, this uses synthetic data.
Replace with your actual data source.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import config
from dataset import StockImageDataset
from model import TradingCNN, DeepTradingCNN
from train import create_dataloaders, train
from backtest import Backtester


def generate_sample_data(num_days=500, start_price=100):
    """
    Generate synthetic OHLC data for demonstration.

    In production, replace this with your actual data loading:
    - Read from CSV: pd.read_csv('data.csv')
    - Query from PostgreSQL: pd.read_sql(query, connection)
    - Fetch from API: your data provider
    """
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')

    # Generate realistic-looking price data with trend and noise
    trend = np.linspace(0, 0.3, num_days)
    noise = np.random.randn(num_days) * 0.02
    returns = trend + noise
    prices = start_price * np.exp(np.cumsum(returns))

    # Create OHLC from close prices
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, num_days),
    })

    data.set_index('date', inplace=True)
    return data


def main():
    """Main demonstration workflow."""

    print("=" * 80)
    print("VISUAL TRADING SYSTEM - COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Data Preparation
    # ========================================================================
    print("\n📊 STEP 1: Loading Data")
    print("-" * 80)

    # Generate sample data (replace with your data loading)
    price_data = generate_sample_data(num_days=500)

    print(f"Loaded {len(price_data)} days of data")
    print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
    print(f"Price range: ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f}")

    # ========================================================================
    # STEP 2: Create Dataset
    # ========================================================================
    print("\n🖼️  STEP 2: Creating Image Dataset")
    print("-" * 80)

    dataset = StockImageDataset(
        price_data,
        lookback=config.LOOKBACK_WINDOW,
        horizon=config.PREDICTION_HORIZON
    )

    print(f"Created {len(dataset)} image samples")
    print(f"Image shape: {dataset.images.shape}")
    print(f"Channels: {config.NUM_CHANNELS} (price + volume)")

    # ========================================================================
    # STEP 3: Create Dataloaders
    # ========================================================================
    print("\n📦 STEP 3: Creating Train/Val/Test Splits")
    print("-" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config.BATCH_SIZE
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # ========================================================================
    # STEP 4: Create Model
    # ========================================================================
    print("\n🧠 STEP 4: Creating Model")
    print("-" * 80)

    # Choose model type
    if config.MODEL_TYPE == 'simple':
        model = TradingCNN(num_channels=config.NUM_CHANNELS)
    elif config.MODEL_TYPE == 'deep':
        model = DeepTradingCNN(
            num_channels=config.NUM_CHANNELS,
            dropout=config.DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model type: {config.MODEL_TYPE}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ========================================================================
    # STEP 5: Train Model
    # ========================================================================
    print("\n🚀 STEP 5: Training Model")
    print("-" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print()

    history = train(
        model,
        train_loader,
        val_loader,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE
    )

    print(f"\n✅ Training complete!")
    print(f"Best epoch: {history['best_epoch'] + 1}")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")

    # ========================================================================
    # STEP 6: Backtest Model
    # ========================================================================
    print("\n📈 STEP 6: Backtesting Model")
    print("-" * 80)

    # Note: In production, you would:
    # 1. Load the best saved checkpoint
    # 2. Use separate out-of-sample test data
    # 3. Run walk-forward analysis

    print("\n⚠️  Note: Backtesting requires model prediction integration.")
    print("The backtest.py framework is ready, but needs actual model predictions.")
    print("\nTo complete backtesting:")
    print("1. Load trained model checkpoint")
    print("2. Integrate model predictions in Backtester.run()")
    print("3. Run walk-forward simulation on test data")

    # Placeholder backtest demonstration
    backtester = Backtester(model, price_data, initial_capital=100000)
    print(f"\nBacktester initialized with ${backtester.portfolio.initial_capital:,.2f}")
    print(f"Position size: {config.POSITION_SIZE * 100}% per trade")
    print(f"Stop loss: {config.STOP_LOSS * 100}%")
    print(f"Take profit: {config.TAKE_PROFIT * 100}%")

    # ========================================================================
    # STEP 7: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Replace synthetic data with your actual OHLC data
2. Experiment with different configurations in config.py:
   - Adjust LOOKBACK_WINDOW (default: 60 days)
   - Try different MODEL_TYPE ('simple' vs 'deep')
   - Tune LEARNING_RATE and other hyperparameters

3. Integrate model predictions in backtest.py (line 175)
   - Extract image window from historical data
   - Run through trained model
   - Use prediction for trading signals

4. Run full walk-forward backtesting
5. Analyze performance metrics

6. Extension ideas:
   - Add technical indicators as additional channels
   - Try different image transformations (MTF, Recurrence Plots)
   - Experiment with ensemble methods
   - Implement multi-timeframe analysis
""")


if __name__ == '__main__':
    main()
