#!/usr/bin/env python3
"""
Ensemble Testing Script
Tests if correlation-weighted ensemble of models improves trading performance
Implements Option A.1 from synthesis report recommendations
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


class CorrelationWeightedEnsemble:
    """
    Ensemble that weights models by their correlation performance
    Excludes models with negative or near-zero correlation
    """

    def __init__(self, model_correlations: Dict[str, float], threshold: float = 0.1):
        """
        Args:
            model_correlations: {ticker: correlation} from investigation
            threshold: Minimum correlation to include model
        """
        self.correlations = {
            ticker: corr
            for ticker, corr in model_correlations.items()
            if corr > threshold
        }

        if len(self.correlations) == 0:
            raise ValueError(f"No models above correlation threshold {threshold}")

        # Normalize weights
        total_corr = sum(self.correlations.values())
        self.weights = {
            ticker: corr / total_corr
            for ticker, corr in self.correlations.items()
        }

        print(f"\n📊 Ensemble Configuration:")
        print(f"   Models included: {len(self.weights)}")
        print(f"   Correlation threshold: {threshold}")
        print(f"\n   Weights:")
        for ticker, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            corr = self.correlations[ticker]
            print(f"      {ticker}: {weight:.3f} (correlation: {corr:.3f})")

    def predict(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate ensemble predictions weighted by correlation

        Args:
            model_predictions: {ticker: predictions_array}

        Returns:
            ensemble_predictions: Weighted average predictions
        """
        # Get first array to determine shape
        ensemble = np.zeros_like(next(iter(model_predictions.values())))

        for ticker, weight in self.weights.items():
            if ticker in model_predictions:
                ensemble += weight * model_predictions[ticker]

        return ensemble


def load_model_predictions(tickers: List[str], etf_data: Dict) -> Dict[str, Dict]:
    """
    Load all models and generate predictions on test sets

    Args:
        tickers: List of model tickers to load
        etf_data: ETF price data

    Returns:
        dict: {ticker: {'predictions': array, 'actuals': array}}
    """
    checkpoint_dir = Path(config.MODEL_DIR) / 'checkpoints'
    all_predictions = {}

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Loading: {ticker}")
        print(f"{'='*60}")

        checkpoint_path = checkpoint_dir / f"{ticker}_model.pt"

        if not checkpoint_path.exists():
            print(f"   ⚠️  Checkpoint not found, skipping")
            continue

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

            # Initialize model
            model = TradingCNN(num_channels=checkpoint['config']['num_channels'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(config.DEVICE)
            model.eval()

            # Get data
            if ticker not in etf_data:
                print(f"   ⚠️  No data available, skipping")
                continue

            df = etf_data[ticker]

            # Create dataset
            lookback = checkpoint['config']['lookback']
            horizon = checkpoint['config']['horizon']
            dataset = StockImageDataset(df, lookback=lookback, horizon=horizon)

            if len(dataset) == 0:
                print(f"   ⚠️  No valid samples, skipping")
                continue

            # Get test split
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_start = train_size + val_size

            # Run predictions
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

            all_predictions[ticker] = {
                'predictions': predictions,
                'actuals': actuals
            }

            print(f"   ✅ Loaded: {len(predictions)} predictions")

        except Exception as e:
            print(f"   ❌ Error loading {ticker}: {str(e)}")
            continue

    return all_predictions


def calculate_backtest_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict:
    """
    Calculate trading performance metrics

    Args:
        predictions: Model predictions
        actuals: Actual returns

    Returns:
        dict: Performance metrics
    """
    # Simple long-only strategy
    positions = (predictions > 0).astype(float)

    # Strategy returns
    strategy_returns = positions * actuals

    # Cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0

    # Annualized metrics
    trading_days = len(strategy_returns)
    years = trading_days / 252

    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0

    # Sharpe ratio
    if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    cumulative_wealth = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

    # Win metrics
    winning_trades = np.sum(strategy_returns > 0)
    total_trades = np.sum(positions > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Directional accuracy
    correct_direction = np.sum((predictions > 0) == (actuals > 0))
    accuracy = correct_direction / len(predictions) if len(predictions) > 0 else 0

    # Win/Loss ratio
    winning_mask = (positions > 0) & (actuals > 0)
    losing_mask = (positions > 0) & (actuals <= 0)

    avg_win = np.mean(actuals[winning_mask]) if np.sum(winning_mask) > 0 else 0
    avg_loss = np.mean(actuals[losing_mask]) if np.sum(losing_mask) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'accuracy': float(accuracy),
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'win_loss_ratio': float(win_loss_ratio)
    }


def test_ensemble_vs_individuals(model_data: Dict[str, Dict],
                                correlations: Dict[str, float],
                                threshold: float = 0.1) -> Dict:
    """
    Test ensemble performance vs individual models

    Args:
        model_data: {ticker: {'predictions': array, 'actuals': array}}
        correlations: {ticker: correlation_value}
        threshold: Minimum correlation for inclusion

    Returns:
        dict: Comparison results
    """
    print("\n" + "="*80)
    print("ENSEMBLE VS INDIVIDUAL MODELS COMPARISON")
    print("="*80)

    # Get common test samples (use shortest sequence)
    min_length = min(len(data['predictions']) for data in model_data.values())
    print(f"\n📊 Using {min_length} common test samples")

    # Truncate all to same length
    for ticker in model_data:
        model_data[ticker]['predictions'] = model_data[ticker]['predictions'][:min_length]
        model_data[ticker]['actuals'] = model_data[ticker]['actuals'][:min_length]

    # Verify all actuals are identical (should be testing on same period)
    actuals_list = [data['actuals'] for data in model_data.values()]
    actuals = actuals_list[0]

    # Create ensemble
    ensemble = CorrelationWeightedEnsemble(correlations, threshold=threshold)

    # Get predictions from all models
    all_predictions = {
        ticker: data['predictions']
        for ticker, data in model_data.items()
    }

    # Generate ensemble predictions
    ensemble_predictions = ensemble.predict(all_predictions)

    print(f"\n🔄 Generating ensemble predictions...")
    print(f"   ✅ Complete: {len(ensemble_predictions)} predictions")

    # Calculate metrics for ensemble
    print(f"\n📈 Calculating ensemble metrics...")
    ensemble_metrics = calculate_backtest_metrics(ensemble_predictions, actuals)

    # Calculate metrics for each individual model
    individual_metrics = {}
    for ticker in model_data:
        predictions = model_data[ticker]['predictions']
        individual_metrics[ticker] = calculate_backtest_metrics(predictions, actuals)

    # Print comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)

    print(f"\n{'Model':<15} {'Return':<12} {'Sharpe':<10} {'Win Rate':<10} {'W/L Ratio':<12} {'Trades':<8}")
    print("-" * 80)

    # Individual models
    for ticker in sorted(individual_metrics.keys()):
        metrics = individual_metrics[ticker]
        print(f"{ticker:<15} {metrics['annualized_return']:>10.2%}  "
              f"{metrics['sharpe_ratio']:>8.2f}  "
              f"{metrics['win_rate']:>8.2%}  "
              f"{metrics['win_loss_ratio']:>10.2f}  "
              f"{metrics['total_trades']:>6}")

    # Ensemble
    print("-" * 80)
    print(f"{'ENSEMBLE':<15} {ensemble_metrics['annualized_return']:>10.2%}  "
          f"{ensemble_metrics['sharpe_ratio']:>8.2f}  "
          f"{ensemble_metrics['win_rate']:>8.2%}  "
          f"{ensemble_metrics['win_loss_ratio']:>10.2f}  "
          f"{ensemble_metrics['total_trades']:>6}")

    # Calculate improvement
    avg_individual_return = np.mean([m['annualized_return'] for m in individual_metrics.values()])
    avg_individual_sharpe = np.mean([m['sharpe_ratio'] for m in individual_metrics.values()])

    improvement_return = ensemble_metrics['annualized_return'] - avg_individual_return
    improvement_sharpe = ensemble_metrics['sharpe_ratio'] - avg_individual_sharpe

    print("\n" + "="*80)
    print("ENSEMBLE IMPROVEMENT")
    print("="*80)
    print(f"Return vs Average:  {improvement_return:+.2%}")
    print(f"Sharpe vs Average:  {improvement_sharpe:+.2f}")
    print(f"Win/Loss Ratio:     {ensemble_metrics['win_loss_ratio']:.2f}")

    # Determine if ensemble is better
    better_than_avg = ensemble_metrics['annualized_return'] > avg_individual_return
    positive_return = ensemble_metrics['annualized_return'] > 0

    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)

    if positive_return:
        print("✅ ENSEMBLE ACHIEVES POSITIVE RETURNS")
        print(f"   Annualized Return: {ensemble_metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {ensemble_metrics['sharpe_ratio']:.2f}")
    elif better_than_avg:
        print("⚠️  ENSEMBLE BETTER THAN AVERAGE BUT STILL NEGATIVE")
        print(f"   Improvement: {improvement_return:+.2%}")
        print(f"   But still negative: {ensemble_metrics['annualized_return']:.2%}")
    else:
        print("❌ ENSEMBLE WORSE THAN AVERAGE")
        print(f"   Decline: {improvement_return:.2%}")

    if ensemble_metrics['win_loss_ratio'] >= 1.0:
        print(f"\n✅ ENSEMBLE FIXES WIN/LOSS RATIO PROBLEM")
        print(f"   Ratio: {ensemble_metrics['win_loss_ratio']:.2f} (>= 1.0)")
    else:
        print(f"\n❌ ENSEMBLE STILL HAS WIN/LOSS RATIO PROBLEM")
        print(f"   Ratio: {ensemble_metrics['win_loss_ratio']:.2f} (< 1.0)")

    return {
        'ensemble_metrics': ensemble_metrics,
        'individual_metrics': individual_metrics,
        'ensemble_predictions': ensemble_predictions.tolist(),
        'actuals': actuals.tolist(),
        'improvement_return': float(improvement_return),
        'improvement_sharpe': float(improvement_sharpe),
        'positive_return': bool(positive_return),
        'better_than_average': bool(better_than_avg),
        'ensemble_weights': ensemble.weights,
        'models_included': list(ensemble.weights.keys())
    }


def main():
    """Main ensemble testing workflow"""

    print("=" * 80)
    print("ENSEMBLE TESTING: OPTION A.1")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Model correlations from investigation results
    correlations = {
        'OXLCG': 0.246,
        'HCXY': -0.035,  # Near-zero, should be excluded
        'VGI': -0.111,   # Negative, should be excluded
        'HYI': 0.692,    # Strong but ultra-conservative
        'IGI': 0.140     # Weak but positive
    }

    print("📊 Model Correlations from Investigation:")
    for ticker, corr in sorted(correlations.items(), key=lambda x: x[1], reverse=True):
        status = "✅" if corr > 0.1 else "❌"
        print(f"   {status} {ticker}: {corr:+.3f}")

    # Load ETF data
    print("\n🔄 Loading ETF data...")
    loader = ETFDataLoader()
    etf_data = loader.get_all_etf_data(days=400)
    print(f"   ✅ Loaded {len(etf_data)} ETFs")

    # Load all model predictions
    print("\n🔄 Loading models and generating predictions...")
    tickers = list(correlations.keys())
    model_data = load_model_predictions(tickers, etf_data)

    print(f"\n   ✅ Successfully loaded {len(model_data)} models")

    if len(model_data) < 2:
        print("\n❌ Not enough models loaded for ensemble testing")
        return

    # Test ensemble with threshold = 0.1 (exclude negative/near-zero correlations)
    print("\n🔄 Testing ensemble...")
    results = test_ensemble_vs_individuals(model_data, correlations, threshold=0.1)

    # Save results
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f"ensemble_test_{timestamp}.json"

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_type': 'correlation_weighted_ensemble',
            'correlation_threshold': 0.1,
            'results': results
        }, f, indent=2)

    print(f"\n💾 Saved results to {results_path}")

    # Generate summary report
    print("\n" + "="*80)
    print("NEXT STEPS RECOMMENDATION")
    print("="*80)

    if results['positive_return']:
        print("\n✅ ENSEMBLE SUCCESSFUL - Models have salvageable value!")
        print("\nRecommended next steps:")
        print("   1. Test confidence-based position sizing (Option A.2)")
        print("   2. Implement portfolio of best-performing approach")
        print("   3. Consider loss function improvements (Option B) for further gains")
    elif results['better_than_average']:
        print("\n⚠️  ENSEMBLE IMPROVES BUT STILL NEGATIVE")
        print("\nRecommended next steps:")
        print("   1. Try confidence-based filtering (Option A.2)")
        print("   2. If still negative, proceed to loss function experiments (Option B)")
        print("   3. Consider traditional ML baseline (Option A.3) in parallel")
    else:
        print("\n❌ ENSEMBLE DOES NOT IMPROVE PERFORMANCE")
        print("\nRecommended next steps:")
        print("   1. Test traditional ML baseline (Option A.3) immediately")
        print("   2. If baseline better, abandon CNN approach")
        print("   3. If baseline also fails, consider architecture redesign (Option C)")

    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
