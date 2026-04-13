"""
Trilemma DCA System - Complete Runner

End-to-end pipeline for practicum-ready Bitcoin DCA allocation strategy.
Compares model-driven dynamic DCA against naive baseline with equal budgets.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from dataclasses import dataclass
from data_store import DataStore
from indicators import add_technical_indicators
from dataset import EnhancedBinaryDataset
from model import DeepTradingCNNClassifier
from train_binary import create_dataloaders, train_binary_classifier
from calibration import calibrate_model, get_calibrated_probabilities
from trilemma_dca import TrilemmaAllocator, get_allocation_multiplier
import config


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward evaluation."""
    train_days: int = 365
    val_days: int = 120  # needs lookback(90) + horizon(10) + margin
    test_days: int = 60
    step_days: int = 30
    min_train_samples: int = 50  # safety threshold


def run_trilemma_evaluation(
    ticker='BTC',
    lookback=90,
    horizon=10,
    base_dca_amount=100.0,
    dca_frequency='weekly',
    sensitivity=1.5,
    min_mult=0.7,
    max_mult=1.6,
    smoothing_alpha=0.3,
    num_epochs=50,
    device='cpu'
):
    """
    Run complete Trilemma DCA evaluation on Bitcoin.

    Args:
        ticker: Asset ticker (default: 'BTC' for BTC-USD)
        lookback: Days of history for predictions (default: 90)
        horizon: Days ahead to predict (default: 10)
        base_dca_amount: Base dollar amount per DCA period (default: $100)
        dca_frequency: 'daily', 'weekly', 'biweekly', or 'monthly'
        sensitivity: Allocation tilt sensitivity 1.0-3.0 (default: 1.5)
        min_mult: Minimum allocation multiplier (default: 0.7)
        max_mult: Maximum allocation multiplier (default: 1.6)
        smoothing_alpha: EMA smoothing factor (default: 0.3)
        num_epochs: Training epochs (default: 50)
        device: 'cpu', 'cuda', or 'mps'

    Returns:
        Dictionary with results and metrics
    """
    print("=" * 80)
    print("TRILEMMA DCA SYSTEM - BITCOIN ALLOCATION STRATEGY")
    print("=" * 80)
    print(f"Asset: {ticker}")
    print(f"DCA Frequency: {dca_frequency}")
    print(f"Base Amount: ${base_dca_amount} per period")
    print(f"Lookback: {lookback} days | Horizon: {horizon} days")
    print(f"Sensitivity: {sensitivity} | Range: [{min_mult}, {max_mult}]")
    print(f"Smoothing: α = {smoothing_alpha}")
    print("=" * 80)

    # Step 1: Load data
    print("\n[Step 1/6] Loading Bitcoin data...")
    store = DataStore()
    price_data = store.get_bars(ticker)

    if price_data.empty:
        print(f"❌ No data found for {ticker}")
        return None

    print(f"  ✅ Loaded {len(price_data)} days")
    print(f"  Date range: {price_data['date'].min()} to {price_data['date'].max()}")
    print(f"  Price range: ${price_data['close'].min():.2f} to ${price_data['close'].max():.2f}")

    # Step 2: Add technical indicators
    print("\n[Step 2/6] Adding technical indicators...")
    price_data_enhanced = add_technical_indicators(price_data)
    print(f"  ✅ Added {len(price_data_enhanced.columns) - len(price_data.columns)} indicators")

    # Step 3: Train model on historical data
    print("\n[Step 3/6] Training prediction model...")
    train_size = int(len(price_data_enhanced) * 0.7)
    val_size = int(len(price_data_enhanced) * 0.15)

    train_data = price_data_enhanced.iloc[:train_size]
    val_data = price_data_enhanced.iloc[train_size:train_size+val_size]
    test_data = price_data_enhanced.iloc[train_size+val_size:]

    print(f"  Train: {len(train_data)} days | Val: {len(val_data)} days | Test: {len(test_data)} days")

    # Create datasets
    train_dataset = EnhancedBinaryDataset(train_data, lookback=lookback, horizon=horizon, use_indicators=True)
    val_dataset = EnhancedBinaryDataset(val_data, lookback=lookback, horizon=horizon, use_indicators=True)
    test_dataset = EnhancedBinaryDataset(test_data, lookback=lookback, horizon=horizon, use_indicators=True)

    num_channels = len(train_dataset.channels)
    print(f"  Channels: {num_channels} - {train_dataset.channels}")

    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(train_dataset, batch_size=config.BATCH_SIZE)

    # Train model
    model = DeepTradingCNNClassifier(num_channels=num_channels, num_classes=2, dropout=config.DROPOUT)
    print(f"  Model: DeepTradingCNNClassifier ({sum(p.numel() for p in model.parameters()):,} params)")

    # Suppress training output
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    history = train_binary_classifier(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        device=device,
        save_path=None
    )

    sys.stdout = old_stdout
    print(f"  ✅ Training complete (Best F1: {history['best_val_f1']:.4f})")

    # Step 4: Calibrate probabilities
    print("\n[Step 4/6] Calibrating probabilities...")
    # Use the existing val_loader from training
    temp_scaler, temp_value = calibrate_model(model, val_loader, device=device)

    # Step 5: Generate predictions for evaluation period
    print("\n[Step 5/6] Generating allocation schedule...")

    model.eval()
    predictions = []

    # Key fix: Allow lookback to use train/val data, only require prediction date in test
    test_start_idx = train_size + val_size
    test_end_idx = len(price_data_enhanced)

    # We can predict for dates where we have:
    # 1. Enough history for lookback (can include train/val)
    # 2. Enough future data for horizon (must stay within available data)
    from pyts.image import GramianAngularField
    gaf = GramianAngularField(image_size=lookback, method='summation')

    with torch.no_grad():
        for prediction_idx in range(test_start_idx, test_end_idx - horizon):
            # Lookback window: can reach into train/val data
            lookback_start = prediction_idx - lookback
            if lookback_start < 0:
                continue  # Need at least lookback days of history

            window = price_data_enhanced.iloc[lookback_start:prediction_idx]

            # Generate multi-channel GAF image (same logic as EnhancedBinaryDataset)
            channels = []
            for channel_name in train_dataset.channels:
                if channel_name in window.columns:
                    channel_data = window[channel_name].values.reshape(1, -1)

                    # Normalize to [-1, 1] for GAF
                    channel_min = channel_data.min()
                    channel_max = channel_data.max()

                    if channel_max > channel_min:
                        normalized = 2 * (channel_data - channel_min) / (channel_max - channel_min) - 1
                    else:
                        normalized = np.zeros_like(channel_data)

                    # Transform to GAF
                    gaf_image = gaf.transform(normalized)
                    channels.append(gaf_image[0])

            # Stack channels and convert to tensor
            img = np.stack(channels, axis=0)
            image = torch.FloatTensor(img).unsqueeze(0).to(device)

            # Get calibrated probabilities
            logits = model(image)
            calibrated_probs = temp_scaler(logits)
            prob_up = calibrated_probs[0, 1].item()

            # Get prediction date
            date = price_data_enhanced.iloc[prediction_idx]['date']

            predictions.append({
                'date': date,
                'prob_up': prob_up
            })

    print(f"  ✅ Generated {len(predictions)} predictions")

    # Step 6: Calculate DCA allocations
    print("\n[Step 6/6] Calculating DCA allocations...")

    allocator = TrilemmaAllocator(
        base_dca_amount=base_dca_amount,
        dca_frequency=dca_frequency,
        sensitivity=sensitivity,
        min_mult=min_mult,
        max_mult=max_mult,
        smoothing_alpha=smoothing_alpha
    )

    schedule, daily_valuation = allocator.generate_allocation_schedule(
        predictions=predictions,
        price_data=price_data_enhanced
    )

    print(f"  ✅ Generated {len(schedule)} allocation periods")

    # Calculate metrics with daily valuation for accurate drawdown
    metrics = allocator.calculate_metrics(schedule, daily_valuation=daily_valuation)

    # Print report
    print()
    allocator.print_report(metrics)

    # Save results
    print("\n[Saving Results]")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    schedule_file = f"{config.RESULTS_DIR}/{ticker}_trilemma_schedule_{timestamp}.csv"
    metrics_file = f"{config.RESULTS_DIR}/{ticker}_trilemma_metrics_{timestamp}.txt"

    schedule.to_csv(schedule_file, index=False)
    print(f"  ✅ Schedule saved to {schedule_file}")

    with open(metrics_file, 'w') as f:
        f.write("TRILEMMA DCA SYSTEM - PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Evaluation Period: {metrics['date_range']}\n")
        f.write(f"Number of Periods: {metrics['num_periods']}\n")
        f.write(f"Base DCA Amount: ${base_dca_amount}\n\n")

        f.write("DYNAMIC DCA (Model-Driven)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Invested: ${metrics['total_invested_dynamic']:,.2f}\n")
        f.write(f"Units Accumulated: {metrics['total_units_accumulated_dynamic']:.8f}\n")
        f.write(f"Avg Unit Cost: ${metrics['avg_unit_cost_dynamic']:,.2f}\n")
        f.write(f"Final Value: ${metrics['final_value_dynamic']:,.2f}\n")
        f.write(f"Total Return: {metrics['total_return_pct_dynamic']:+.2f}%\n\n")

        f.write("NAIVE DCA (Baseline)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Invested: ${metrics['total_invested_naive']:,.2f}\n")
        f.write(f"Units Accumulated: {metrics['total_units_accumulated_naive']:.8f}\n")
        f.write(f"Avg Unit Cost: ${metrics['avg_unit_cost_naive']:,.2f}\n")
        f.write(f"Final Value: ${metrics['final_value_naive']:,.2f}\n")
        f.write(f"Total Return: {metrics['total_return_pct_naive']:+.2f}%\n\n")

        f.write("COMPARATIVE PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Alpha vs Naive: {metrics['alpha_vs_naive']:+.2f}%\n")
        f.write(f"Unit Accumulation Improvement: {metrics['unit_accumulation_improvement']:+.2f}%\n")
        f.write(f"Unit Cost Improvement: {metrics['unit_cost_improvement']:+.2f}%\n\n")

        f.write("RISK METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n")
        f.write(f"Information Ratio: {metrics['information_ratio']:.2f}\n\n")

        f.write("EFFICIENCY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Avg Allocation Turnover: {metrics['avg_allocation_turnover']:.4f}\n")
        f.write(f"Total Turnover: {metrics['total_allocation_turnover']:.2f}\n")

    print(f"  ✅ Metrics saved to {metrics_file}")

    print("\n" + "=" * 80)
    print("TRILEMMA EVALUATION COMPLETE")
    print("=" * 80)

    return {
        'schedule': schedule,
        'metrics': metrics,
        'model': model,
        'temp_scaler': temp_scaler,
        'history': history
    }


def run_trilemma_walkforward(
    ticker='BTC',
    lookback=90,
    horizon=10,
    base_dca_amount=100.0,
    dca_frequency='weekly',
    sensitivity=1.5,
    min_mult=0.7,
    max_mult=1.6,
    smoothing_alpha=0.3,
    num_epochs=20,
    device='mps',
    wf: WalkForwardConfig = None
):
    """
    Run walk-forward evaluation for robust out-of-sample validation.

    Args:
        ticker: Asset ticker
        lookback: Days of history for predictions
        horizon: Days ahead to predict
        base_dca_amount: Base dollar amount per DCA period
        dca_frequency: 'daily', 'weekly', 'biweekly', or 'monthly'
        sensitivity: Allocation tilt sensitivity
        min_mult: Minimum allocation multiplier
        max_mult: Maximum allocation multiplier
        smoothing_alpha: EMA smoothing factor
        num_epochs: Training epochs per block
        device: 'cpu', 'cuda', or 'mps'
        wf: WalkForwardConfig instance

    Returns:
        Dictionary with blocks, stitched schedule, and aggregate metrics
    """
    if wf is None:
        wf = WalkForwardConfig()

    # Step 1: Load data
    print("\n[WF] Loading data...")
    store = DataStore()
    price_data = store.get_bars(ticker)
    if price_data.empty:
        print(f"❌ No data found for {ticker}")
        return None

    price_data = price_data.sort_values("date").reset_index(drop=True)
    price_data["date"] = pd.to_datetime(price_data["date"])

    print(f"  ✅ Loaded {len(price_data)} days: {price_data['date'].min().date()} → {price_data['date'].max().date()}")

    # Step 2: Add technical indicators (once for whole series)
    print("\n[WF] Adding technical indicators...")
    price_data_enhanced = add_technical_indicators(price_data)

    n = len(price_data_enhanced)
    window_len = wf.train_days + wf.val_days + wf.test_days

    start_idx = 0
    end_idx = n

    # Define block start positions
    block_starts = list(range(start_idx, end_idx - window_len + 1, wf.step_days))
    if not block_starts:
        print("❌ Not enough data for the requested walk-forward windows.")
        return None

    print(f"\n[WF] Running {len(block_starts)} walk-forward blocks "
          f"(train={wf.train_days}, val={wf.val_days}, test={wf.test_days}, step={wf.step_days})")

    allocator = TrilemmaAllocator(
        base_dca_amount=base_dca_amount,
        dca_frequency=dca_frequency,
        sensitivity=sensitivity,
        min_mult=min_mult,
        max_mult=max_mult,
        smoothing_alpha=smoothing_alpha
    )

    all_block_results = []
    stitched_weekly_rows = []
    stitched_daily_values = []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for b, block_start in enumerate(block_starts, 1):
        train_start = block_start
        train_end = train_start + wf.train_days
        val_end = train_end + wf.val_days
        test_end = val_end + wf.test_days

        train_data = price_data_enhanced.iloc[train_start:train_end].copy()
        val_data = price_data_enhanced.iloc[train_end:val_end].copy()
        test_data = price_data_enhanced.iloc[val_end:test_end].copy()

        # Skip blocks too early for lookback/horizon usage
        if (val_end - lookback) < 0:
            continue

        # Build datasets/loaders
        train_dataset = EnhancedBinaryDataset(train_data, lookback=lookback, horizon=horizon, use_indicators=True)
        val_dataset = EnhancedBinaryDataset(val_data, lookback=lookback, horizon=horizon, use_indicators=True)

        if len(train_dataset) < wf.min_train_samples or len(val_dataset) < 10:
            print(f"[WF] Block {b}: skipped (insufficient samples). train={len(train_dataset)}, val={len(val_dataset)}")
            continue

        num_channels = len(train_dataset.channels)
        train_loader, val_loader, _ = create_dataloaders(train_dataset, batch_size=config.BATCH_SIZE)

        # Train model
        model = DeepTradingCNNClassifier(num_channels=num_channels, num_classes=2, dropout=config.DROPOUT)

        # Suppress training output
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        history = train_binary_classifier(
            model, train_loader, val_loader,
            num_epochs=num_epochs,
            device=device,
            save_path=None
        )

        sys.stdout = old_stdout

        # Calibrate
        temp_scaler, temp_value = calibrate_model(model, val_loader, device=device)

        # Predict only within this block's test window
        from pyts.image import GramianAngularField
        gaf = GramianAngularField(image_size=lookback, method='summation')
        model.eval()
        predictions = []

        test_start_idx = val_end
        test_end_idx = test_end

        with torch.no_grad():
            for prediction_idx in range(test_start_idx, test_end_idx - horizon):
                lookback_start = prediction_idx - lookback
                if lookback_start < 0:
                    continue

                window = price_data_enhanced.iloc[lookback_start:prediction_idx]

                # Build multi-channel GAF image
                channels = []
                for channel_name in train_dataset.channels:
                    if channel_name in window.columns:
                        channel_data = window[channel_name].values.reshape(1, -1)
                        cmin, cmax = channel_data.min(), channel_data.max()
                        if cmax > cmin:
                            normalized = 2 * (channel_data - cmin) / (cmax - cmin) - 1
                        else:
                            normalized = np.zeros_like(channel_data)
                        gaf_image = gaf.transform(normalized)
                        channels.append(gaf_image[0])

                if len(channels) != num_channels:
                    continue

                img = np.stack(channels, axis=0)
                image = torch.FloatTensor(img).unsqueeze(0).to(device)

                logits = model(image)
                calibrated_probs = temp_scaler(logits)
                prob_up = calibrated_probs[0, 1].item()

                date = price_data_enhanced.iloc[prediction_idx]['date']
                predictions.append({'date': date, 'prob_up': prob_up})

        if len(predictions) < 10:
            print(f"[WF] Block {b}: skipped (too few predictions: {len(predictions)})")
            continue

        # Allocate + metrics for this block
        schedule, daily_valuation = allocator.generate_allocation_schedule(
            predictions=predictions,
            price_data=price_data_enhanced
        )

        # Walk-forward purity checks (prevent data leakage)
        assert schedule['date'].min() >= test_data['date'].iloc[0], \
            f"Block {b}: schedule starts before test window ({schedule['date'].min()} < {test_data['date'].iloc[0]})"
        assert schedule['date'].max() <= test_data['date'].iloc[-1], \
            f"Block {b}: schedule ends after test window ({schedule['date'].max()} > {test_data['date'].iloc[-1]})"

        metrics = allocator.calculate_metrics(schedule, daily_valuation=daily_valuation)

        # Regime classification for test window
        test_prices = test_data['close'].values
        # Trend: simple price direction over test window
        price_change_pct = (test_prices[-1] - test_prices[0]) / test_prices[0]
        trend = "up" if price_change_pct > 0 else "down"

        # Volatility: realized vol over test window
        returns = np.diff(np.log(test_prices))
        realized_vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0  # annualized
        # Store raw vol for percentile calculation later
        vol_regime_raw = realized_vol

        # Attach block metadata (with correct keys)
        block_meta = {
            "block": b,
            "train_start": train_data["date"].iloc[0],
            "train_end": train_data["date"].iloc[-1],
            "val_end": val_data["date"].iloc[-1],
            "test_start": test_data["date"].iloc[0],
            "test_end": test_data["date"].iloc[-1],
            "n_periods": len(schedule),  # number of DCA buys in this test block
            "trend": trend,  # up/down based on test window price change
            "realized_vol": float(realized_vol),  # annualized volatility
            "temp": float(temp_value),
            "best_val_f1": float(history.get("best_val_f1", np.nan)),
            "alpha_vs_naive": float(metrics.get("alpha_vs_naive", np.nan)),
            "unit_accumulation_improvement": float(metrics.get("unit_accumulation_improvement", np.nan)),
            "unit_cost_improvement": float(metrics.get("unit_cost_improvement", np.nan)),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", np.nan)),
            "information_ratio": float(metrics.get("information_ratio", np.nan)),
        }
        all_block_results.append(block_meta)

        # Save per-block artifacts
        block_id_str = f"{b:03d}"
        block_sched_file = f"{config.RESULTS_DIR}/{ticker}_block_{block_id_str}_schedule_{timestamp}.csv"
        block_metrics_file = f"{config.RESULTS_DIR}/{ticker}_block_{block_id_str}_metrics_{timestamp}.txt"

        schedule_block = schedule.copy()
        schedule_block["block"] = b
        schedule_block.to_csv(block_sched_file, index=False)

        with open(block_metrics_file, "w") as f:
            f.write(f"BLOCK {b} METRICS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test window: {test_data['date'].iloc[0].date()} → {test_data['date'].iloc[-1].date()}\n\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        # Stitch schedules (keep only dates inside this test window)
        schedule_block["test_window_start"] = test_data["date"].iloc[0]
        schedule_block["test_window_end"] = test_data["date"].iloc[-1]
        stitched_weekly_rows.append(schedule_block)

        # Stitch daily valuation
        dv = daily_valuation.copy()
        dv = dv.loc[pd.to_datetime(test_data["date"].iloc[0]):pd.to_datetime(test_data["date"].iloc[-1])]
        dv["block"] = b
        stitched_daily_values.append(dv)

        print(f"[WF] Block {b}/{len(block_starts)} "
              f"Test {test_data['date'].iloc[0].date()}→{test_data['date'].iloc[-1].date()} | "
              f"Alpha {block_meta['alpha_vs_naive']:.2f}% | IR {block_meta['information_ratio']:.2f} | "
              f"F1 {block_meta['best_val_f1']:.3f} | T {block_meta['temp']:.3f}")

    if not all_block_results:
        print("❌ No completed walk-forward blocks.")
        return None

    blocks_df = pd.DataFrame(all_block_results)

    stitched_schedule = pd.concat(stitched_weekly_rows, ignore_index=True).sort_values("date")
    # Dedupe overlapping buy dates (keep most recent block's decision)
    stitched_schedule = stitched_schedule.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # Build a continuous DAILY valuation for the stitched OOS schedule
    # (rebuilding from stitched_schedule prevents spurious block-reset artifacts)
    stitched_daily = allocator._build_daily_valuation(
        price_data=price_data_enhanced,      # full daily prices
        weekly_schedule=stitched_schedule    # stitched weekly buys
    )

    # Add sanity checks
    min_v = min(stitched_daily["value_dynamic"].min(), stitched_daily["value_naive"].min())
    max_v = max(stitched_daily["value_dynamic"].max(), stitched_daily["value_naive"].max())
    assert min_v >= 0, f"Aggregate valuation has negative value: {min_v}"
    assert max_v > 0, f"Aggregate valuation looks empty/zero: max={max_v}"

    # Aggregate metrics over stitched schedule
    agg_metrics = allocator.calculate_metrics(
        stitched_schedule,
        daily_valuation=stitched_daily
    )

    # Save artifacts
    blocks_file = f"{config.RESULTS_DIR}/{ticker}_wf_blocks_{timestamp}.csv"
    sched_file = f"{config.RESULTS_DIR}/{ticker}_wf_schedule_{timestamp}.csv"
    metrics_file = f"{config.RESULTS_DIR}/{ticker}_wf_metrics_{timestamp}.txt"

    blocks_df.to_csv(blocks_file, index=False)
    stitched_schedule.to_csv(sched_file, index=False)

    with open(metrics_file, "w") as f:
        f.write("TRILEMMA WALK-FORWARD AGGREGATE METRICS\n")
        f.write("=" * 50 + "\n\n")
        for k, v in agg_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n\nBLOCK-LEVEL SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(blocks_df.describe(include="all").to_string())
        f.write("\n")

    print(f"\n[WF] Saved:\n  {blocks_file}\n  {sched_file}\n  {metrics_file}")

    return {
        "blocks": blocks_df,
        "stitched_schedule": stitched_schedule,
        "stitched_daily": stitched_daily,
        "aggregate_metrics": agg_metrics
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Trilemma DCA System for Bitcoin')
    parser.add_argument('--ticker', type=str, default='BTC', help='Ticker symbol (default: BTC)')
    parser.add_argument('--lookback', type=int, default=90, help='Lookback days (default: 90)')
    parser.add_argument('--horizon', type=int, default=10, help='Prediction horizon (default: 10)')
    parser.add_argument('--amount', type=float, default=100.0, help='Base DCA amount (default: $100)')
    parser.add_argument('--sensitivity', type=float, default=1.5, help='Tilt sensitivity (default: 1.5)')
    parser.add_argument('--min-mult', type=float, default=0.7, help='Min multiplier (default: 0.7)')
    parser.add_argument('--max-mult', type=float, default=1.6, help='Max multiplier (default: 1.6)')
    parser.add_argument('--alpha', type=float, default=0.3, help='Smoothing alpha (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--frequency', type=str, default='weekly',
                        choices=['daily', 'weekly', 'biweekly', 'monthly'],
                        help='DCA frequency (default: weekly)')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device for training (default: auto-detect)')

    # Walk-forward evaluation flags
    parser.add_argument('--walkforward', action='store_true', help='Enable walk-forward evaluation')
    parser.add_argument('--train-days', type=int, default=365, help='WF: Train window days (default: 365)')
    parser.add_argument('--val-days', type=int, default=120, help='WF: Val window days (default: 120)')
    parser.add_argument('--test-days', type=int, default=60, help='WF: Test window days (default: 60)')
    parser.add_argument('--step-days', type=int, default=30, help='WF: Step size days (default: 30)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"🎲 Random seed set to {args.seed}")

    if args.walkforward:
        # Walk-forward evaluation
        wf_config = WalkForwardConfig(
            train_days=args.train_days,
            val_days=args.val_days,
            test_days=args.test_days,
            step_days=args.step_days
        )

        results = run_trilemma_walkforward(
            ticker=args.ticker,
            lookback=args.lookback,
            horizon=args.horizon,
            base_dca_amount=args.amount,
            dca_frequency=args.frequency,
            sensitivity=args.sensitivity,
            min_mult=args.min_mult,
            max_mult=args.max_mult,
            smoothing_alpha=args.alpha,
            num_epochs=args.epochs,
            device=args.device,
            wf=wf_config
        )

        if results is None:
            print("\n❌ Walk-forward evaluation failed")
            sys.exit(1)
        else:
            # Print aggregate summary (with correct metric keys)
            print("\n" + "=" * 80)
            print("WALK-FORWARD AGGREGATE RESULTS")
            print("=" * 80)
            print("(Alpha = total return difference vs naive DCA under identical cash invested)")
            print()
            agg = results["aggregate_metrics"]

            # Concrete dollar and unit differences (all using dynamic - naive convention)
            total_invested = agg.get('total_invested_dynamic', 0.0)
            num_periods = agg.get('num_periods', 0)

            final_value_dyn = agg.get('final_value_dynamic', 0.0)
            final_value_naive = agg.get('final_value_naive', 0.0)
            value_delta = final_value_dyn - final_value_naive

            units_dyn = agg.get('total_units_accumulated_dynamic', 0.0)
            units_naive = agg.get('total_units_accumulated_naive', 0.0)
            units_delta = units_dyn - units_naive

            avg_cost_dyn = agg.get('avg_unit_cost_dynamic', 0.0)
            avg_cost_naive = agg.get('avg_unit_cost_naive', 0.0)
            cost_delta = avg_cost_dyn - avg_cost_naive  # Fixed: dynamic - naive (positive = dynamic paid MORE)

            print(f"Total invested:        ${total_invested:,.2f} across {num_periods} deduped buy dates")
            print(f"                       (buy amounts vary due to budget normalization; total matched to naive)")
            print(f"Final value delta:     ${value_delta:+.2f} (dynamic - naive)")
            print(f"Units delta:           {units_delta:+.8f} BTC (dynamic - naive)")
            print(f"Avg cost basis delta:  ${cost_delta:+.2f} per BTC (dynamic - naive, + = paid more)")
            print()
            print(f"Scale note: ${total_invested:,.0f} is a small test notional; purpose is robustness + regime signal, not absolute profit.")
            print()
            print(f"Alpha vs Naive:        {agg.get('alpha_vs_naive', 0.0):+.2f}%")
            print(f"Unit Accumulation:     {agg.get('unit_accumulation_improvement', 0.0):+.2f}%")
            print(f"Unit Cost Improvement: {agg.get('unit_cost_improvement', 0.0):+.2f}%")
            print(f"Max Drawdown:          {agg.get('max_drawdown_pct', 0.0):.2f}%")
            print(f"Information Ratio:     {agg.get('information_ratio', 0.0):.2f}")

            print("\n" + "=" * 80)
            print("BLOCK-LEVEL STATISTICS")
            print("=" * 80)
            blocks = results["blocks"]
            total_scheduled = blocks['n_periods'].sum()
            actual_unique = agg.get('num_periods', total_scheduled)

            print(f"Total blocks:          {len(blocks)}")
            print(f"Scheduled buys:        {total_scheduled} (before dedupe)")
            print(f"Unique buys:           {actual_unique} (after dedupe)")
            print(f"  → Aggregate results use {actual_unique} deduped buys; block results use {total_scheduled} block-buys before dedupe")

            # Weighted vs unweighted alpha
            weighted_alpha = np.average(blocks['alpha_vs_naive'], weights=blocks['n_periods'])
            print(f"\nUnweighted avg alpha:  {blocks['alpha_vs_naive'].mean():+.2f}% (typical block)")
            print(f"Weighted avg alpha:    {weighted_alpha:+.2f}% (period-weighted, pre-dedupe)")
            print(f"Aggregate alpha:       {agg.get('alpha_vs_naive', 0.0):+.2f}% (post-dedupe stitched OOS)")
            print(f"Median alpha:          {blocks['alpha_vs_naive'].median():+.2f}%")
            print(f"Positive blocks:       {(blocks['alpha_vs_naive'] > 0).sum()} ({(blocks['alpha_vs_naive'] > 0).mean():.1%})")

            print(f"\nAvg IR:                {blocks['information_ratio'].mean():.2f}")
            print(f"Avg F1:                {blocks['best_val_f1'].mean():.3f}")
            print(f"Avg temp:              {blocks['temp'].mean():.3f}")

            # Regime analysis (2x2 table: trend × vol)
            vol_median = blocks['realized_vol'].median()
            blocks['vol_regime'] = blocks['realized_vol'].apply(lambda x: 'high' if x > vol_median else 'low')

            print("\n" + "=" * 80)
            print("REGIME SENSITIVITY (Trend × Volatility)")
            print("=" * 80)
            for trend in ['up', 'down']:
                for vol in ['low', 'high']:
                    regime_blocks = blocks[(blocks['trend'] == trend) & (blocks['vol_regime'] == vol)]
                    if len(regime_blocks) > 0:
                        avg_alpha = regime_blocks['alpha_vs_naive'].mean()
                        count = len(regime_blocks)
                        print(f"{trend.capitalize()}/{vol.capitalize()} Vol:  {avg_alpha:+.2f}% avg alpha ({count} blocks)")
                    else:
                        print(f"{trend.capitalize()}/{vol.capitalize()} Vol:  N/A (0 blocks)")

            # Dual IR reporting: scheduled (pre-dedupe) vs stitched (post-dedupe)
            print("\n" + "=" * 80)
            print("IR VALIDATION (Scheduled vs Stitched)")
            print("=" * 80)

            # IR_scheduled: average of block IRs (pre-dedupe, 56 scheduled buys)
            ir_scheduled = blocks['information_ratio'].mean()
            print(f"IR_scheduled:          {ir_scheduled:.4f} (avg of {len(blocks)} block IRs, pre-dedupe)")

            # IR_stitched: computed on deduped 34-buy stitched path
            stitched_schedule = results["stitched_schedule"]
            if 'tracking_error' in stitched_schedule.columns:
                te_mean = stitched_schedule['tracking_error'].mean()
                te_std = stitched_schedule['tracking_error'].std()
                ann_factor = np.sqrt(52)  # weekly cadence

                # Scale verification: TE should be in decimal returns (e.g., 0.001 = 0.1%)
                # If TE mean/std are ~0.02-0.10, they're likely decimal returns (2%-10%)
                scale_warning = ""
                if abs(te_mean) > 0.01 or te_std > 0.05:
                    scale_warning = " ⚠️ (check if TE is in decimal or % units)"

                print(f"\nIR_stitched ({actual_unique} deduped buys):")
                print(f"  TE mean:             {te_mean:+.8f} (dynamic - naive){scale_warning}")
                print(f"  TE std:              {te_std:.8f}")
                print(f"  TE mean (pct):       {te_mean*100:+.4f}% per period")
                print(f"  TE std (pct):        {te_std*100:.4f}% per period")
                print(f"  Annualization:       {ann_factor:.4f} (√52)")

                if te_std > 0:
                    ir_stitched = (te_mean / te_std) * ann_factor
                    reported_ir = agg.get('information_ratio', 0.0)
                    print(f"  Calculated IR:       {ir_stitched:.4f}")
                    print(f"  Reported IR:         {reported_ir:.4f}")
                    if abs(ir_stitched - reported_ir) < 0.01:
                        print("  ✅ Calculation verified")
                    else:
                        print(f"  ⚠️ Mismatch (diff: {abs(ir_stitched - reported_ir):.4f})")

                    print(f"\n📊 Kickoff headline uses IR_stitched: {ir_stitched:.2f}")
                    print(f"   (Computed on same {actual_unique}-buy deduped path as aggregate alpha)")

            print("\n✅ Walk-forward evaluation succeeded")
            sys.exit(0)
    else:
        # Single-split evaluation (original)
        results = run_trilemma_evaluation(
            ticker=args.ticker,
            lookback=args.lookback,
            horizon=args.horizon,
            base_dca_amount=args.amount,
            dca_frequency=args.frequency,
            sensitivity=args.sensitivity,
            min_mult=args.min_mult,
            max_mult=args.max_mult,
            smoothing_alpha=args.alpha,
            num_epochs=args.epochs,
            device=args.device
        )

        if results is None:
            print("\n❌ Evaluation failed")
            sys.exit(1)
        else:
            print("\n✅ Evaluation succeeded")
            sys.exit(0)


if __name__ == '__main__':
    main()
