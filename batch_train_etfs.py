#!/usr/bin/env python3
"""
Batch Training Script for Top 25 ETFs
Trains individual CNN models for each top-performing ETF from Stock Recommender
"""

import torch
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

from load_etf_data import ETFDataLoader
from dataset import StockImageDataset
from model import TradingCNN
from train import create_dataloaders, train
import config


def train_single_etf(ticker, df, output_dir):
    """
    Train a CNN model for a single ETF

    Args:
        ticker: ETF ticker symbol
        df: pandas DataFrame with OHLC data
        output_dir: Directory to save checkpoints and results

    Returns:
        dict: Training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training: {ticker}")
    print(f"{'='*60}")

    try:
        # Create dataset with GAF transformation
        print(f"📊 Creating dataset (lookback={config.LOOKBACK_WINDOW}, horizon={config.PREDICTION_HORIZON})...")
        dataset = StockImageDataset(
            df,
            lookback=config.LOOKBACK_WINDOW,
            horizon=config.PREDICTION_HORIZON
        )

        if len(dataset) == 0:
            print(f"⚠️  Insufficient data for {ticker} - skipping")
            return {
                'ticker': ticker,
                'status': 'skipped',
                'error': 'Insufficient data for training'
            }

        print(f"   Dataset size: {len(dataset)} samples")

        # Create dataloaders
        print(f"🔄 Creating dataloaders (train/val/test: {config.TRAIN_SPLIT}/{config.VAL_SPLIT}/{config.TEST_SPLIT})...")
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=config.BATCH_SIZE
        )

        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

        # Initialize model
        print(f"🏗️  Initializing {config.MODEL_TYPE} CNN model...")
        model = TradingCNN(num_channels=config.NUM_CHANNELS)

        # Train model
        print(f"🚀 Starting training (max epochs: {config.NUM_EPOCHS}, patience: {config.EARLY_STOPPING_PATIENCE})...")
        start_time = time.time()

        history = train(
            model,
            train_loader,
            val_loader,
            num_epochs=config.NUM_EPOCHS,
            device=config.DEVICE
        )

        training_time = time.time() - start_time

        # Save checkpoint
        checkpoint_dir = Path(output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{ticker}_model.pt"

        torch.save({
            'model_state_dict': model.state_dict(),
            'ticker': ticker,
            'config': {
                'lookback': config.LOOKBACK_WINDOW,
                'horizon': config.PREDICTION_HORIZON,
                'num_channels': config.NUM_CHANNELS,
                'model_type': config.MODEL_TYPE
            },
            'history': history,
            'training_time': training_time
        }, checkpoint_path)

        print(f"💾 Saved checkpoint to {checkpoint_path}")

        # Compile results
        result = {
            'ticker': ticker,
            'status': 'success',
            'dataset_size': len(dataset),
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val_loss'],
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'training_time_sec': training_time,
            'checkpoint_path': str(checkpoint_path)
        }

        print(f"\n✅ Training complete:")
        print(f"   Best epoch: {result['best_epoch']}")
        print(f"   Best val loss: {result['best_val_loss']:.6f}")
        print(f"   Training time: {training_time/60:.1f} minutes")

        return result

    except Exception as e:
        print(f"❌ Error training {ticker}: {str(e)}")
        return {
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }


def main():
    """Main batch training workflow"""

    print("=" * 80)
    print("BATCH TRAINING: TOP 25 ETFs FROM STOCK RECOMMENDER")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration summary
    print("📋 Configuration:")
    print(f"   Lookback window: {config.LOOKBACK_WINDOW} days")
    print(f"   Prediction horizon: {config.PREDICTION_HORIZON} days")
    print(f"   Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"   GAF method: {config.GAF_METHOD}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Max epochs: {config.NUM_EPOCHS}")
    print(f"   Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"   Device: {config.DEVICE}")
    print()

    # Load all ETF data from PostgreSQL
    print("🔄 Loading ETF data from PostgreSQL...")
    loader = ETFDataLoader()

    # Request 400 days to ensure sufficient data after lookback window
    etf_data = loader.get_all_etf_data(days=400)

    if not etf_data:
        print("❌ No ETF data loaded - exiting")
        return

    print(f"\n✅ Loaded {len(etf_data)} ETFs")
    print()

    # Create output directory
    output_dir = Path(config.MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train each ETF
    all_results = []
    total_start_time = time.time()

    for idx, (ticker, df) in enumerate(etf_data.items(), 1):
        print(f"\n[{idx}/{len(etf_data)}] Processing {ticker}...")
        print(f"   Data range: {df.index[0]} to {df.index[-1]} ({len(df)} days)")

        result = train_single_etf(ticker, df, config.MODEL_DIR)
        all_results.append(result)

        # Progress update
        elapsed = time.time() - total_start_time
        avg_time_per_etf = elapsed / idx
        remaining_etfs = len(etf_data) - idx
        eta_minutes = (remaining_etfs * avg_time_per_etf) / 60

        print(f"\n📊 Progress: {idx}/{len(etf_data)} ETFs complete")
        print(f"   Average time per ETF: {avg_time_per_etf/60:.1f} minutes")
        print(f"   ETA: {eta_minutes:.1f} minutes remaining")

    total_time = time.time() - total_start_time

    # Generate summary report
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    successful = [r for r in all_results if r['status'] == 'success']
    skipped = [r for r in all_results if r['status'] == 'skipped']
    errors = [r for r in all_results if r['status'] == 'error']

    print(f"\n✅ Successful: {len(successful)}/{len(all_results)} ETFs")
    print(f"⚠️  Skipped: {len(skipped)} ETFs")
    print(f"❌ Errors: {len(errors)} ETFs")
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")

    if successful:
        print(f"\n📈 Performance Statistics (Successful Models):")
        val_losses = [r['best_val_loss'] for r in successful]
        print(f"   Best validation loss - Min: {min(val_losses):.6f}, Max: {max(val_losses):.6f}, Avg: {sum(val_losses)/len(val_losses):.6f}")

        epochs = [r['best_epoch'] for r in successful]
        print(f"   Best epochs - Min: {min(epochs)}, Max: {max(epochs)}, Avg: {sum(epochs)/len(epochs):.1f}")

        times = [r['training_time_sec'] for r in successful]
        print(f"   Training time - Min: {min(times)/60:.1f}min, Max: {max(times)/60:.1f}min, Avg: {sum(times)/len(times)/60:.1f}min")

    # Save detailed results to JSON
    results_json_path = results_dir / f"batch_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'lookback_window': config.LOOKBACK_WINDOW,
                'prediction_horizon': config.PREDICTION_HORIZON,
                'batch_size': config.BATCH_SIZE,
                'num_epochs': config.NUM_EPOCHS,
                'device': config.DEVICE
            },
            'summary': {
                'total_etfs': len(all_results),
                'successful': len(successful),
                'skipped': len(skipped),
                'errors': len(errors),
                'total_time_minutes': total_time / 60
            },
            'results': all_results
        }, f, indent=2)

    print(f"\n💾 Saved detailed results to {results_json_path}")

    # Save summary CSV
    if successful:
        df_results = pd.DataFrame(successful)
        csv_path = results_dir / f"batch_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"💾 Saved summary CSV to {csv_path}")

    # List errors if any
    if errors:
        print(f"\n⚠️  Errors encountered:")
        for err in errors:
            print(f"   {err['ticker']}: {err['error']}")

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
