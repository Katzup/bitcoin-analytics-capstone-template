#!/usr/bin/env python3
"""
Model Backtesting Script
Tests trading performance of validated CNN models on held-out test data
Simulates trading strategies and calculates performance metrics
"""

import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from model import TradingCNN
from load_etf_data import ETFDataLoader
from dataset import StockImageDataset
import config


def calculate_returns(predictions: np.ndarray, actuals: np.ndarray,
                     prices: np.ndarray) -> Dict[str, float]:
    """
    Calculate trading returns based on model predictions

    Args:
        predictions: Model predicted returns
        actuals: Actual returns
        prices: Actual prices for the period

    Returns:
        dict: Performance metrics
    """
    # Simple strategy: Go long if prediction > 0, otherwise cash
    positions = (predictions > 0).astype(float)

    # Calculate strategy returns
    strategy_returns = positions * actuals

    # Cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0

    # Annualized metrics (assuming daily data)
    trading_days = len(strategy_returns)
    years = trading_days / 252

    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0

    # Sharpe ratio (risk-free rate = 0 for simplicity)
    if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    cumulative_wealth = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

    # Win rate
    winning_trades = np.sum(strategy_returns > 0)
    total_trades = np.sum(positions > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Prediction accuracy
    correct_direction = np.sum((predictions > 0) == (actuals > 0))
    accuracy = correct_direction / len(predictions) if len(predictions) > 0 else 0

    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'accuracy': float(accuracy),
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'num_predictions': len(predictions)
    }


def backtest_single_model(checkpoint_path: Path, ticker: str,
                         etf_data: Dict) -> Dict:
    """
    Backtest a single trained model

    Args:
        checkpoint_path: Path to model checkpoint
        ticker: ETF ticker symbol
        etf_data: Dictionary of ETF data

    Returns:
        dict: Backtesting results
    """
    print(f"\n{'='*60}")
    print(f"Backtesting: {ticker}")
    print(f"{'='*60}")

    try:
        # Load checkpoint
        print(f"📂 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

        # Initialize model
        model = TradingCNN(num_channels=checkpoint['config']['num_channels'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.DEVICE)
        model.eval()
        print(f"   ✅ Model loaded successfully")

        # Get data
        if ticker not in etf_data:
            return {
                'ticker': ticker,
                'status': 'error',
                'error': f'No data available for {ticker}'
            }

        df = etf_data[ticker]

        # Create dataset
        lookback = checkpoint['config']['lookback']
        horizon = checkpoint['config']['horizon']

        dataset = StockImageDataset(df, lookback=lookback, horizon=horizon)

        if len(dataset) == 0:
            return {
                'ticker': ticker,
                'status': 'error',
                'error': 'No valid samples in dataset'
            }

        print(f"   ✅ Dataset created: {len(dataset)} samples")

        # Split into train/val/test (same as training)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Get test indices
        test_start = train_size + val_size

        # Run predictions on test set
        print(f"🔄 Running predictions on test set ({test_size} samples)...")

        predictions = []
        actuals = []

        with torch.no_grad():
            for idx in range(test_start, len(dataset)):
                image, target = dataset[idx]
                image_batch = image.unsqueeze(0).to(config.DEVICE)

                output = model(image_batch)

                predictions.append(output.item())
                actuals.append(target.item())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        print(f"   ✅ Generated {len(predictions)} predictions")

        # Get corresponding prices for context
        test_dates_start = test_start + lookback + horizon
        test_df = df.iloc[test_dates_start:test_dates_start + len(predictions)]
        prices = test_df['close'].values if 'close' in test_df.columns else np.ones(len(predictions))

        # Calculate performance metrics
        print(f"📊 Calculating performance metrics...")
        metrics = calculate_returns(predictions, actuals, prices)

        # Add model info
        result = {
            'ticker': ticker,
            'status': 'success',
            'checkpoint_path': str(checkpoint_path),
            'config': checkpoint['config'],
            'training_metrics': {
                'best_val_loss': checkpoint['history']['best_val_loss'],
                'best_epoch': checkpoint['history']['best_epoch']
            },
            'backtest_metrics': metrics,
            'test_samples': test_size
        }

        # Print summary
        print(f"\n📈 Backtest Results:")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Prediction Accuracy: {metrics['accuracy']:.2%}")
        print(f"   Total Trades: {metrics['total_trades']}")

        print(f"\n✅ Backtest complete for {ticker}")
        return result

    except Exception as e:
        print(f"❌ Error backtesting {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }


def main():
    """Main backtesting workflow"""

    print("=" * 80)
    print("MODEL BACKTESTING: TOP 5 PERFORMERS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Top 5 models from performance analysis
    top_models = [
        'OXLCG',  # 0.000048 val loss, 35 epochs
        'HCXY',   # 0.000066 val loss, 7 epochs
        'VGI',    # 0.000093 val loss, 24 epochs
        'HYI',    # 0.000098 val loss, 81 epochs
        'IGI'     # 0.000115 val loss, 3 epochs
    ]

    print(f"📦 Testing top {len(top_models)} models:")
    for i, ticker in enumerate(top_models, 1):
        print(f"   {i}. {ticker}")
    print()

    # Load ETF data
    print("🔄 Loading ETF data for backtesting...")
    loader = ETFDataLoader()
    etf_data = loader.get_all_etf_data(days=400)
    print(f"   ✅ Loaded {len(etf_data)} ETFs")
    print()

    # Backtest each model
    all_results = []
    checkpoint_dir = Path(config.MODEL_DIR) / 'checkpoints'

    for idx, ticker in enumerate(top_models, 1):
        print(f"\n[{idx}/{len(top_models)}] Processing {ticker}...")

        checkpoint_path = checkpoint_dir / f"{ticker}_model.pt"

        if not checkpoint_path.exists():
            print(f"   ⚠️  Checkpoint not found: {checkpoint_path}")
            all_results.append({
                'ticker': ticker,
                'status': 'error',
                'error': 'Checkpoint file not found'
            })
            continue

        result = backtest_single_model(checkpoint_path, ticker, etf_data)
        all_results.append(result)

    # Generate summary report
    print("\n" + "=" * 80)
    print("BACKTESTING SUMMARY")
    print("=" * 80)

    successful = [r for r in all_results if r['status'] == 'success']
    errors = [r for r in all_results if r['status'] == 'error']

    print(f"\n✅ Successful: {len(successful)}/{len(all_results)} models")
    print(f"❌ Errors: {len(errors)} models")

    if successful:
        print(f"\n📊 Performance Rankings:")
        print()

        # Sort by Sharpe ratio
        sorted_by_sharpe = sorted(successful,
                                 key=lambda x: x['backtest_metrics']['sharpe_ratio'],
                                 reverse=True)

        print("By Sharpe Ratio:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Ticker':<10} {'Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Accuracy':<10}")
        print("-" * 80)

        for rank, result in enumerate(sorted_by_sharpe, 1):
            metrics = result['backtest_metrics']
            print(f"{rank:<6} {result['ticker']:<10} "
                  f"{metrics['annualized_return']:>10.2%}  "
                  f"{metrics['sharpe_ratio']:>8.2f}  "
                  f"{metrics['max_drawdown']:>10.2%}  "
                  f"{metrics['accuracy']:>8.2%}")

        print()

        # Sort by total return
        sorted_by_return = sorted(successful,
                                 key=lambda x: x['backtest_metrics']['total_return'],
                                 reverse=True)

        print("By Total Return:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Ticker':<10} {'Total Return':<15} {'Win Rate':<12} {'Trades':<10}")
        print("-" * 80)

        for rank, result in enumerate(sorted_by_return, 1):
            metrics = result['backtest_metrics']
            print(f"{rank:<6} {result['ticker']:<10} "
                  f"{metrics['total_return']:>13.2%}  "
                  f"{metrics['win_rate']:>10.2%}  "
                  f"{metrics['total_trades']:>8}")

        # Average metrics
        print("\n" + "=" * 80)
        print("AVERAGE PERFORMANCE METRICS")
        print("=" * 80)

        avg_return = np.mean([r['backtest_metrics']['total_return'] for r in successful])
        avg_ann_return = np.mean([r['backtest_metrics']['annualized_return'] for r in successful])
        avg_sharpe = np.mean([r['backtest_metrics']['sharpe_ratio'] for r in successful])
        avg_drawdown = np.mean([r['backtest_metrics']['max_drawdown'] for r in successful])
        avg_winrate = np.mean([r['backtest_metrics']['win_rate'] for r in successful])
        avg_accuracy = np.mean([r['backtest_metrics']['accuracy'] for r in successful])

        print(f"Average Total Return: {avg_return:.2%}")
        print(f"Average Annualized Return: {avg_ann_return:.2%}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Average Max Drawdown: {avg_drawdown:.2%}")
        print(f"Average Win Rate: {avg_winrate:.2%}")
        print(f"Average Prediction Accuracy: {avg_accuracy:.2%}")

    # Save detailed results
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_json_path = results_dir / f"backtest_results_{timestamp}.json"

    with open(results_json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models': len(all_results),
                'successful': len(successful),
                'errors': len(errors),
                'models_tested': top_models
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
