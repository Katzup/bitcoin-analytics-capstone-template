#!/usr/bin/env python3
"""
Test Batch Training Workflow
Validates the complete pipeline with 3 ETFs before running full batch
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


def test_single_etf(ticker, df):
    """
    Test training workflow for a single ETF

    Returns:
        dict: Test results with status and metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing: {ticker}")
    print(f"{'='*60}")

    try:
        # Create dataset
        print(f"📊 Creating dataset...")
        dataset = StockImageDataset(
            df,
            lookback=config.LOOKBACK_WINDOW,
            horizon=config.PREDICTION_HORIZON
        )

        if len(dataset) == 0:
            return {
                'ticker': ticker,
                'status': 'skipped',
                'reason': 'Insufficient data'
            }

        print(f"   ✅ Dataset size: {len(dataset)} samples")

        # Create dataloaders
        print(f"🔄 Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=config.BATCH_SIZE
        )

        print(f"   ✅ Train: {len(train_loader.dataset)} samples")
        print(f"   ✅ Val: {len(val_loader.dataset)} samples")
        print(f"   ✅ Test: {len(test_loader.dataset)} samples")

        # Initialize model
        print(f"🏗️  Initializing model...")
        model = TradingCNN(num_channels=config.NUM_CHANNELS)
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Run limited training (just 5 epochs for testing)
        print(f"🚀 Running test training (5 epochs)...")
        start_time = time.time()

        history = train(
            model,
            train_loader,
            val_loader,
            num_epochs=5,  # Limited for testing
            device=config.DEVICE
        )

        training_time = time.time() - start_time

        print(f"\n✅ Test complete:")
        print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"   Training time: {training_time:.1f} seconds")

        return {
            'ticker': ticker,
            'status': 'success',
            'dataset_size': len(dataset),
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'training_time_sec': training_time
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }


def main():
    """Test workflow with 3 ETFs"""

    print("=" * 80)
    print("BATCH TRAINING PIPELINE TEST (3 ETFs)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    print("📋 Test Configuration:")
    print(f"   Lookback window: {config.LOOKBACK_WINDOW} days")
    print(f"   Prediction horizon: {config.PREDICTION_HORIZON} days")
    print(f"   Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Device: {config.DEVICE}")
    print(f"   Test epochs: 5 (limited for validation)")
    print()

    # Load ETF data (limit to 3 for testing)
    print("🔄 Loading ETF data from PostgreSQL...")
    loader = ETFDataLoader()
    etf_data = loader.get_all_etf_data(days=400, limit=3)

    if not etf_data:
        print("❌ No ETF data loaded - exiting")
        return

    print(f"✅ Loaded {len(etf_data)} ETFs for testing")
    print()

    # Test each ETF
    results = []
    total_start = time.time()

    for idx, (ticker, df) in enumerate(etf_data.items(), 1):
        print(f"\n[{idx}/{len(etf_data)}] Testing {ticker}...")
        print(f"   Data range: {df.index[0]} to {df.index[-1]} ({len(df)} days)")

        result = test_single_etf(ticker, df)
        results.append(result)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    errors = [r for r in results if r['status'] == 'error']

    print(f"\n✅ Successful: {len(successful)}/{len(results)} ETFs")
    print(f"⚠️  Skipped: {len(skipped)} ETFs")
    print(f"❌ Errors: {len(errors)} ETFs")
    print(f"\n⏱️  Total time: {total_time:.1f} seconds")

    if successful:
        print(f"\n📈 Performance Statistics:")
        val_losses = [r['final_val_loss'] for r in successful]
        print(f"   Validation loss - Min: {min(val_losses):.6f}, Max: {max(val_losses):.6f}, Avg: {sum(val_losses)/len(val_losses):.6f}")

        times = [r['training_time_sec'] for r in successful]
        print(f"   Training time - Min: {min(times):.1f}s, Max: {max(times):.1f}s, Avg: {sum(times)/len(times):.1f}s")

    # Save test results
    results_path = Path('test_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'num_etfs': len(etf_data),
                'test_epochs': 5,
                'lookback': config.LOOKBACK_WINDOW,
                'horizon': config.PREDICTION_HORIZON
            },
            'summary': {
                'successful': len(successful),
                'skipped': len(skipped),
                'errors': len(errors),
                'total_time_seconds': total_time
            },
            'results': results
        }, f, indent=2)

    print(f"\n💾 Saved test results to {results_path}")

    # Final verdict
    print("\n" + "=" * 80)
    if len(successful) == len(results):
        print("✅ PIPELINE VALIDATION SUCCESSFUL")
        print("   All components working correctly")
        print("   Ready for full 25-ETF batch training")
    elif len(successful) > 0:
        print("⚠️  PIPELINE PARTIALLY VALIDATED")
        print(f"   {len(successful)}/{len(results)} ETFs completed successfully")
        print("   Review errors before full batch run")
    else:
        print("❌ PIPELINE VALIDATION FAILED")
        print("   Fix errors before proceeding")
    print("=" * 80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
