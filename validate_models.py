#!/usr/bin/env python3
"""
Model Validation Script
Verifies trained model checkpoints can be loaded and perform inference
Tests inference speed and output validation
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from model import TradingCNN
from load_etf_data import ETFDataLoader
from dataset import StockImageDataset
import config


def validate_single_model(checkpoint_path, ticker, etf_data=None):
    """
    Validate a single trained model checkpoint

    Args:
        checkpoint_path: Path to model .pt file
        ticker: ETF ticker symbol
        etf_data: Optional pre-loaded ETF data dict

    Returns:
        dict: Validation results
    """
    print(f"\n{'='*60}")
    print(f"Validating: {ticker}")
    print(f"{'='*60}")

    try:
        # Load checkpoint
        print(f"📂 Loading checkpoint from {checkpoint_path}...")
        start_load = time.time()
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        load_time = time.time() - start_load
        print(f"   ✅ Loaded in {load_time:.3f}s")

        # Verify checkpoint structure
        required_keys = ['model_state_dict', 'ticker', 'config', 'history']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            return {
                'ticker': ticker,
                'status': 'error',
                'error': f'Missing checkpoint keys: {missing_keys}'
            }

        print(f"   ✅ Checkpoint structure valid")
        print(f"   Config: lookback={checkpoint['config']['lookback']}, "
              f"horizon={checkpoint['config']['horizon']}, "
              f"channels={checkpoint['config']['num_channels']}")

        # Initialize model
        print(f"🏗️  Initializing model...")
        model = TradingCNN(num_channels=checkpoint['config']['num_channels'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.DEVICE)
        model.eval()
        print(f"   ✅ Model initialized and set to eval mode")

        # Test inference if data provided
        if etf_data and ticker in etf_data:
            print(f"🔄 Testing inference with real data...")
            df = etf_data[ticker]

            # Create dataset
            dataset = StockImageDataset(
                df,
                lookback=checkpoint['config']['lookback'],
                horizon=checkpoint['config']['horizon']
            )

            if len(dataset) == 0:
                print(f"   ⚠️  No valid samples in dataset")
                inference_time = None
                output_shape = None
                output_range = None
            else:
                # Get a sample
                sample_image, sample_target = dataset[0]
                sample_batch = sample_image.unsqueeze(0).to(config.DEVICE)

                # Run inference
                with torch.no_grad():
                    start_inference = time.time()
                    output = model(sample_batch)
                    inference_time = time.time() - start_inference

                output_shape = tuple(output.shape)
                output_value = output.item()
                target_value = sample_target.item()

                print(f"   ✅ Inference successful")
                print(f"   Input shape: {tuple(sample_batch.shape)}")
                print(f"   Output shape: {output_shape}")
                print(f"   Output value: {output_value:.6f}")
                print(f"   Target value: {target_value:.6f}")
                print(f"   Inference time: {inference_time*1000:.2f}ms")

                output_range = (output_value, target_value)
        else:
            print(f"   ℹ️  No data provided - skipping inference test")
            inference_time = None
            output_shape = None
            output_range = None

        # Compile results
        result = {
            'ticker': ticker,
            'status': 'success',
            'checkpoint_path': str(checkpoint_path),
            'load_time_sec': load_time,
            'model_params': sum(p.numel() for p in model.parameters()),
            'config': checkpoint['config'],
            'best_val_loss': checkpoint['history']['best_val_loss'],
            'best_epoch': checkpoint['history']['best_epoch'],
            'inference_time_ms': inference_time * 1000 if inference_time else None,
            'output_shape': output_shape,
            'output_range': output_range
        }

        print(f"\n✅ Validation complete for {ticker}")
        return result

    except Exception as e:
        print(f"❌ Error validating {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }


def main():
    """Main validation workflow"""

    print("=" * 80)
    print("MODEL VALIDATION: BATCH TRAINING CHECKPOINTS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Find all checkpoint files
    checkpoint_dir = Path(config.MODEL_DIR) / 'checkpoints'
    checkpoint_files = sorted(checkpoint_dir.glob('*.pt'))

    if not checkpoint_files:
        print("❌ No checkpoint files found")
        return

    print(f"📦 Found {len(checkpoint_files)} checkpoint files")
    print()

    # Load ETF data for inference testing (optional)
    print("🔄 Loading ETF data for inference testing...")
    loader = ETFDataLoader()
    etf_data = loader.get_all_etf_data(days=400)
    print(f"   ✅ Loaded {len(etf_data)} ETFs")
    print()

    # Validate each model
    all_results = []
    total_start_time = time.time()

    for idx, checkpoint_path in enumerate(checkpoint_files, 1):
        ticker = checkpoint_path.stem.replace('_model', '')
        print(f"\n[{idx}/{len(checkpoint_files)}] Processing {ticker}...")

        result = validate_single_model(checkpoint_path, ticker, etf_data)
        all_results.append(result)

        # Progress update
        elapsed = time.time() - total_start_time
        avg_time_per_model = elapsed / idx
        remaining_models = len(checkpoint_files) - idx
        eta_seconds = remaining_models * avg_time_per_model

        print(f"\n📊 Progress: {idx}/{len(checkpoint_files)} models validated")
        print(f"   Average time per model: {avg_time_per_model:.1f}s")
        print(f"   ETA: {eta_seconds:.1f}s remaining")

    total_time = time.time() - total_start_time

    # Generate summary report
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    successful = [r for r in all_results if r['status'] == 'success']
    errors = [r for r in all_results if r['status'] == 'error']

    print(f"\n✅ Successful: {len(successful)}/{len(all_results)} models")
    print(f"❌ Errors: {len(errors)} models")
    print(f"\n⏱️  Total time: {total_time:.1f}s")

    if successful:
        print(f"\n📈 Performance Statistics:")

        load_times = [r['load_time_sec'] for r in successful]
        print(f"   Checkpoint load time - Min: {min(load_times):.3f}s, "
              f"Max: {max(load_times):.3f}s, "
              f"Avg: {sum(load_times)/len(load_times):.3f}s")

        inference_times = [r['inference_time_ms'] for r in successful if r['inference_time_ms'] is not None]
        if inference_times:
            print(f"   Inference time - Min: {min(inference_times):.2f}ms, "
                  f"Max: {max(inference_times):.2f}ms, "
                  f"Avg: {sum(inference_times)/len(inference_times):.2f}ms")

        params = [r['model_params'] for r in successful]
        print(f"   Model parameters - {params[0]:,} per model")

    # Save detailed results
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_json_path = results_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models': len(all_results),
                'successful': len(successful),
                'errors': len(errors),
                'total_time_seconds': total_time
            },
            'results': all_results
        }, f, indent=2)

    print(f"\n💾 Saved detailed results to {results_json_path}")

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
