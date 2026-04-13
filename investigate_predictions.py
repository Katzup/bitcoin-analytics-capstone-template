#!/usr/bin/env python3
"""
Prediction Investigation Script
Analyzes prediction distributions and individual trades to understand why models fail
Investigates the disconnect between validation loss and trading profitability
"""

import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import Counter

from model import TradingCNN
from load_etf_data import ETFDataLoader
from dataset import StockImageDataset
import config


def analyze_predictions(predictions: np.ndarray, actuals: np.ndarray,
                       ticker: str) -> Dict:
    """
    Analyze prediction vs actual distributions

    Args:
        predictions: Model predicted returns
        actuals: Actual returns
        ticker: ETF ticker symbol

    Returns:
        dict: Analysis results
    """
    # Distribution statistics
    pred_stats = {
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'median': float(np.median(predictions)),
        'positive_pct': float(np.sum(predictions > 0) / len(predictions)),
        'q25': float(np.percentile(predictions, 25)),
        'q75': float(np.percentile(predictions, 75))
    }

    actual_stats = {
        'mean': float(np.mean(actuals)),
        'std': float(np.std(actuals)),
        'min': float(np.min(actuals)),
        'max': float(np.max(actuals)),
        'median': float(np.median(actuals)),
        'positive_pct': float(np.sum(actuals > 0) / len(actuals)),
        'q25': float(np.percentile(actuals, 25)),
        'q75': float(np.percentile(actuals, 75))
    }

    # Correlation analysis
    correlation = float(np.corrcoef(predictions, actuals)[0, 1])

    # Directional accuracy breakdown
    correct_direction = (predictions > 0) == (actuals > 0)
    directional_accuracy = float(np.mean(correct_direction))

    # Accuracy by prediction magnitude
    pred_abs = np.abs(predictions)
    quartiles = np.percentile(pred_abs, [25, 50, 75])

    q1_mask = pred_abs <= quartiles[0]
    q2_mask = (pred_abs > quartiles[0]) & (pred_abs <= quartiles[1])
    q3_mask = (pred_abs > quartiles[1]) & (pred_abs <= quartiles[2])
    q4_mask = pred_abs > quartiles[2]

    accuracy_by_magnitude = {
        'q1_low_confidence': float(np.mean(correct_direction[q1_mask])) if np.sum(q1_mask) > 0 else 0,
        'q2': float(np.mean(correct_direction[q2_mask])) if np.sum(q2_mask) > 0 else 0,
        'q3': float(np.mean(correct_direction[q3_mask])) if np.sum(q3_mask) > 0 else 0,
        'q4_high_confidence': float(np.mean(correct_direction[q4_mask])) if np.sum(q4_mask) > 0 else 0,
    }

    # Sign agreement
    both_positive = np.sum((predictions > 0) & (actuals > 0))
    both_negative = np.sum((predictions <= 0) & (actuals <= 0))
    pred_pos_actual_neg = np.sum((predictions > 0) & (actuals <= 0))
    pred_neg_actual_pos = np.sum((predictions <= 0) & (actuals > 0))

    sign_breakdown = {
        'both_positive': int(both_positive),
        'both_negative': int(both_negative),
        'pred_pos_actual_neg': int(pred_pos_actual_neg),
        'pred_neg_actual_pos': int(pred_neg_actual_pos),
        'agreement_rate': float((both_positive + both_negative) / len(predictions))
    }

    # Magnitude analysis for trades taken
    trade_mask = predictions > 0
    if np.sum(trade_mask) > 0:
        winning_trades = (predictions > 0) & (actuals > 0)
        losing_trades = (predictions > 0) & (actuals <= 0)

        avg_winning_return = float(np.mean(actuals[winning_trades])) if np.sum(winning_trades) > 0 else 0
        avg_losing_return = float(np.mean(actuals[losing_trades])) if np.sum(losing_trades) > 0 else 0

        trade_analysis = {
            'total_trades': int(np.sum(trade_mask)),
            'winning_trades': int(np.sum(winning_trades)),
            'losing_trades': int(np.sum(losing_trades)),
            'avg_winning_return': avg_winning_return,
            'avg_losing_return': avg_losing_return,
            'win_loss_ratio': abs(avg_winning_return / avg_losing_return) if avg_losing_return != 0 else 0,
            'avg_predicted_return': float(np.mean(predictions[trade_mask])),
            'avg_actual_return': float(np.mean(actuals[trade_mask]))
        }
    else:
        trade_analysis = {
            'total_trades': 0,
            'note': 'Model never predicted positive returns'
        }

    return {
        'ticker': ticker,
        'sample_size': len(predictions),
        'prediction_stats': pred_stats,
        'actual_stats': actual_stats,
        'correlation': correlation,
        'directional_accuracy': directional_accuracy,
        'accuracy_by_magnitude': accuracy_by_magnitude,
        'sign_breakdown': sign_breakdown,
        'trade_analysis': trade_analysis
    }


def get_worst_trades(predictions: np.ndarray, actuals: np.ndarray,
                     n_worst: int = 10) -> List[Dict]:
    """
    Identify worst individual trades

    Args:
        predictions: Model predictions
        actuals: Actual returns
        n_worst: Number of worst trades to return

    Returns:
        list: Worst trades with details
    """
    # Only analyze trades that were taken
    trade_mask = predictions > 0
    trade_indices = np.where(trade_mask)[0]

    if len(trade_indices) == 0:
        return []

    trade_predictions = predictions[trade_mask]
    trade_actuals = actuals[trade_mask]

    # Calculate P&L for each trade
    trade_pnl = trade_actuals

    # Sort by worst P&L
    worst_indices = np.argsort(trade_pnl)[:n_worst]

    worst_trades = []
    for idx in worst_indices:
        original_idx = trade_indices[idx]
        worst_trades.append({
            'sample_index': int(original_idx),
            'predicted_return': float(trade_predictions[idx]),
            'actual_return': float(trade_actuals[idx]),
            'pnl': float(trade_pnl[idx]),
            'error': float(abs(trade_predictions[idx] - trade_actuals[idx]))
        })

    return worst_trades


def get_best_trades(predictions: np.ndarray, actuals: np.ndarray,
                    n_best: int = 10) -> List[Dict]:
    """
    Identify best individual trades

    Args:
        predictions: Model predictions
        actuals: Actual returns
        n_best: Number of best trades to return

    Returns:
        list: Best trades with details
    """
    # Only analyze trades that were taken
    trade_mask = predictions > 0
    trade_indices = np.where(trade_mask)[0]

    if len(trade_indices) == 0:
        return []

    trade_predictions = predictions[trade_mask]
    trade_actuals = actuals[trade_mask]

    # Calculate P&L for each trade
    trade_pnl = trade_actuals

    # Sort by best P&L
    best_indices = np.argsort(trade_pnl)[-n_best:][::-1]

    best_trades = []
    for idx in best_indices:
        original_idx = trade_indices[idx]
        best_trades.append({
            'sample_index': int(original_idx),
            'predicted_return': float(trade_predictions[idx]),
            'actual_return': float(trade_actuals[idx]),
            'pnl': float(trade_pnl[idx]),
            'error': float(abs(trade_predictions[idx] - trade_actuals[idx]))
        })

    return best_trades


def investigate_model(checkpoint_path: Path, ticker: str,
                     etf_data: Dict) -> Dict:
    """
    Comprehensive investigation of single model predictions

    Args:
        checkpoint_path: Path to model checkpoint
        ticker: ETF ticker symbol
        etf_data: Dictionary of ETF data

    Returns:
        dict: Investigation results
    """
    print(f"\n{'='*60}")
    print(f"Investigating: {ticker}")
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

        # Get data
        if ticker not in etf_data:
            return {'ticker': ticker, 'status': 'error', 'error': 'No data'}

        df = etf_data[ticker]

        # Create dataset
        lookback = checkpoint['config']['lookback']
        horizon = checkpoint['config']['horizon']
        dataset = StockImageDataset(df, lookback=lookback, horizon=horizon)

        # Split (same as training)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_start = train_size + val_size

        # Run predictions
        print(f"🔄 Generating predictions...")
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

        # Analyze predictions
        print(f"📊 Analyzing prediction patterns...")
        analysis = analyze_predictions(predictions, actuals, ticker)

        # Get worst and best trades
        print(f"🔍 Identifying extreme trades...")
        worst_trades = get_worst_trades(predictions, actuals, n_worst=10)
        best_trades = get_best_trades(predictions, actuals, n_best=10)

        result = {
            'ticker': ticker,
            'status': 'success',
            'analysis': analysis,
            'worst_trades': worst_trades,
            'best_trades': best_trades,
            'training_metrics': {
                'best_val_loss': checkpoint['history']['best_val_loss'],
                'best_epoch': checkpoint['history']['best_epoch']
            }
        }

        # Print summary
        print(f"\n📈 Investigation Summary:")
        print(f"   Correlation: {analysis['correlation']:.3f}")
        print(f"   Directional Accuracy: {analysis['directional_accuracy']:.2%}")
        print(f"   Prediction Mean: {analysis['prediction_stats']['mean']:.4f}")
        print(f"   Actual Mean: {analysis['actual_stats']['mean']:.4f}")
        print(f"   Trades Taken: {analysis['trade_analysis'].get('total_trades', 0)}")

        if 'avg_winning_return' in analysis['trade_analysis']:
            print(f"   Avg Winning Return: {analysis['trade_analysis']['avg_winning_return']:.4f}")
            print(f"   Avg Losing Return: {analysis['trade_analysis']['avg_losing_return']:.4f}")
            print(f"   Win/Loss Ratio: {analysis['trade_analysis']['win_loss_ratio']:.2f}")

        print(f"\n✅ Investigation complete for {ticker}")
        return result

    except Exception as e:
        print(f"❌ Error investigating {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'ticker': ticker, 'status': 'error', 'error': str(e)}


def main():
    """Main investigation workflow"""

    print("=" * 80)
    print("PREDICTION DISTRIBUTION INVESTIGATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Top 5 models from backtesting
    top_models = ['OXLCG', 'HCXY', 'VGI', 'HYI', 'IGI']

    print(f"📦 Investigating {len(top_models)} models:")
    for i, ticker in enumerate(top_models, 1):
        print(f"   {i}. {ticker}")
    print()

    # Load ETF data
    print("🔄 Loading ETF data...")
    loader = ETFDataLoader()
    etf_data = loader.get_all_etf_data(days=400)
    print(f"   ✅ Loaded {len(etf_data)} ETFs")
    print()

    # Investigate each model
    all_results = []
    checkpoint_dir = Path(config.MODEL_DIR) / 'checkpoints'

    for idx, ticker in enumerate(top_models, 1):
        print(f"\n[{idx}/{len(top_models)}] Processing {ticker}...")

        checkpoint_path = checkpoint_dir / f"{ticker}_model.pt"

        if not checkpoint_path.exists():
            all_results.append({
                'ticker': ticker,
                'status': 'error',
                'error': 'Checkpoint not found'
            })
            continue

        result = investigate_model(checkpoint_path, ticker, etf_data)
        all_results.append(result)

    # Generate cross-model insights
    print("\n" + "=" * 80)
    print("CROSS-MODEL PATTERN ANALYSIS")
    print("=" * 80)

    successful = [r for r in all_results if r['status'] == 'success']

    if successful:
        # Correlation analysis
        correlations = [r['analysis']['correlation'] for r in successful]
        print(f"\n📊 Correlation Statistics:")
        print(f"   Mean: {np.mean(correlations):.3f}")
        print(f"   Std: {np.std(correlations):.3f}")
        print(f"   Min: {np.min(correlations):.3f}")
        print(f"   Max: {np.max(correlations):.3f}")

        # Directional accuracy
        accuracies = [r['analysis']['directional_accuracy'] for r in successful]
        print(f"\n🎯 Directional Accuracy Statistics:")
        print(f"   Mean: {np.mean(accuracies):.2%}")
        print(f"   Std: {np.std(accuracies):.2%}")
        print(f"   Min: {np.min(accuracies):.2%}")
        print(f"   Max: {np.max(accuracies):.2%}")

        # Prediction bias
        print(f"\n⚖️  Prediction Bias Analysis:")
        for r in successful:
            pred_mean = r['analysis']['prediction_stats']['mean']
            actual_mean = r['analysis']['actual_stats']['mean']
            bias = pred_mean - actual_mean
            print(f"   {r['ticker']}: Pred={pred_mean:.4f}, Actual={actual_mean:.4f}, Bias={bias:+.4f}")

        # Trade frequency
        print(f"\n📈 Trade Frequency Analysis:")
        for r in successful:
            total_trades = r['analysis']['trade_analysis'].get('total_trades', 0)
            sample_size = r['analysis']['sample_size']
            trade_freq = total_trades / sample_size if sample_size > 0 else 0
            print(f"   {r['ticker']}: {total_trades}/{sample_size} ({trade_freq:.1%})")

        # Accuracy by confidence
        print(f"\n🎲 Accuracy by Prediction Magnitude:")
        print(f"   {'Ticker':<10} {'Q1 Low':<10} {'Q2':<10} {'Q3':<10} {'Q4 High':<10}")
        print(f"   {'-'*50}")
        for r in successful:
            acc = r['analysis']['accuracy_by_magnitude']
            print(f"   {r['ticker']:<10} {acc['q1_low_confidence']:<10.2%} "
                  f"{acc['q2']:<10.2%} {acc['q3']:<10.2%} "
                  f"{acc['q4_high_confidence']:<10.2%}")

    # Save results
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f"prediction_investigation_{timestamp}.json"

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models_investigated': top_models,
            'results': all_results
        }, f, indent=2)

    print(f"\n💾 Saved investigation results to {results_path}")

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
