#!/usr/bin/env python3
"""
Regime Classification Test - Phase 1 (Path B.1)
Tests whether predicting market regime is more tractable than predicting returns
Implements Option B.1 from baseline comparison synthesis report
"""

import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

from load_etf_data import ETFDataLoader
import config


def define_market_regimes(returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    Define market regimes based on price action and volatility

    Regimes:
    0 = Strong Downtrend (returns < -0.5 std)
    1 = Weak Downtrend (returns between -0.5 and 0 std)
    2 = Sideways/Neutral (returns between -0.2 and +0.2 std, low vol)
    3 = Weak Uptrend (returns between 0 and +0.5 std)
    4 = Strong Uptrend (returns > +0.5 std)
    5 = High Volatility (volatility > 1.5 std, any direction)

    Args:
        returns: Forward N-day returns
        volatility: Rolling volatility measure

    Returns:
        regime_labels: Integer regime classifications
    """
    regimes = np.zeros(len(returns), dtype=int)

    # Calculate thresholds
    return_mean = np.mean(returns)
    return_std = np.std(returns)
    vol_mean = np.mean(volatility)
    vol_std = np.std(volatility)

    # High volatility regime (takes precedence)
    high_vol_mask = volatility > (vol_mean + 1.5 * vol_std)
    regimes[high_vol_mask] = 5

    # For non-high-vol periods, classify by returns
    normal_vol_mask = ~high_vol_mask

    # Strong downtrend
    strong_down_mask = normal_vol_mask & (returns < (return_mean - 0.5 * return_std))
    regimes[strong_down_mask] = 0

    # Weak downtrend
    weak_down_mask = normal_vol_mask & (returns >= (return_mean - 0.5 * return_std)) & (returns < return_mean)
    regimes[weak_down_mask] = 1

    # Sideways (within 0.2 std of mean)
    sideways_mask = normal_vol_mask & (returns >= (return_mean - 0.2 * return_std)) & (returns <= (return_mean + 0.2 * return_std))
    regimes[sideways_mask] = 2

    # Weak uptrend
    weak_up_mask = normal_vol_mask & (returns > return_mean) & (returns <= (return_mean + 0.5 * return_std))
    regimes[weak_up_mask] = 3

    # Strong uptrend
    strong_up_mask = normal_vol_mask & (returns > (return_mean + 0.5 * return_std))
    regimes[strong_up_mask] = 4

    return regimes


def create_features(df: pd.DataFrame, lookback: int = 60, horizon: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create features and regime labels from OHLCV data

    Features: Same 306 features as XGBoost baseline
    Labels: Regime classifications instead of continuous returns

    Returns:
        features: (N, 306) feature array
        regime_labels: (N,) regime classifications
        raw_returns: (N,) continuous returns for regime definition
    """
    features = []
    raw_returns = []
    volatilities = []

    for i in range(lookback, len(df) - horizon):
        # Get lookback window
        window = df.iloc[i-lookback:i]

        # Flatten OHLCV data (300 features)
        ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values.flatten()

        # Calculate technical features (6 features)
        close_prices = window['close'].values
        returns_5d = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) >= 5 else 0
        returns_10d = (close_prices[-1] - close_prices[-10]) / close_prices[-10] if len(close_prices) >= 10 else 0
        returns_20d = (close_prices[-1] - close_prices[-20]) / close_prices[-20] if len(close_prices) >= 20 else 0
        volatility = np.std(close_prices)
        volumes = window['volume'].values
        volume_avg = np.mean(volumes)
        volume_std = np.std(volumes)

        # Combine all features
        feature_vector = np.concatenate([
            ohlcv,
            [returns_5d, returns_10d, returns_20d],
            [volatility, volume_avg, volume_std]
        ])

        # Target: N-day forward return (for regime definition)
        future_price = df.iloc[i + horizon]['close']
        current_price = df.iloc[i]['close']
        target_return = float((future_price - current_price) / current_price)

        features.append(feature_vector)
        raw_returns.append(target_return)
        volatilities.append(volatility)

    features = np.array(features)
    raw_returns = np.array(raw_returns)
    volatilities = np.array(volatilities)

    # Define regimes based on returns and volatility
    regime_labels = define_market_regimes(raw_returns, volatilities)

    return features, regime_labels, raw_returns


def test_regime_classification(ticker: str, etf_data: Dict) -> Dict:
    """
    Test regime classification vs return prediction for single ETF

    Args:
        ticker: ETF ticker symbol
        etf_data: Dictionary of ETF data

    Returns:
        dict: Test results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing: {ticker}")
    print(f"{'='*60}")

    try:
        # Get data
        if ticker not in etf_data:
            return {'ticker': ticker, 'status': 'error', 'error': 'No data available'}

        df = etf_data[ticker]

        # Create features and regime labels
        print(f"📊 Creating features and regime labels...")
        features, regime_labels, raw_returns = create_features(df, lookback=60, horizon=5)

        # Check regime distribution
        unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
        regime_dist = dict(zip(unique_regimes, regime_counts))

        print(f"   ✅ Created {len(features)} samples")
        print(f"   📈 Regime distribution:")
        regime_names = {
            0: "Strong Down",
            1: "Weak Down",
            2: "Sideways",
            3: "Weak Up",
            4: "Strong Up",
            5: "High Vol"
        }
        for regime_id, count in regime_dist.items():
            pct = count / len(regime_labels) * 100
            print(f"      {regime_id} ({regime_names.get(regime_id, 'Unknown')}): {count} ({pct:.1f}%)")

        # Split data (70/15/15)
        train_size = int(0.7 * len(features))
        val_size = int(0.15 * len(features))

        X_train = features[:train_size]
        y_train = regime_labels[:train_size]

        X_val = features[train_size:train_size+val_size]
        y_val = regime_labels[train_size:train_size+val_size]

        X_test = features[train_size+val_size:]
        y_test = regime_labels[train_size+val_size:]

        print(f"\n📦 Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")

        # Train XGBoost classifier
        print(f"\n🔄 Training XGBoost classifier...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Validation performance
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)

        print(f"   ✅ Training complete")
        print(f"   📊 Validation accuracy: {val_accuracy:.2%}")

        # Test performance
        print(f"\n🧪 Testing on hold-out set...")
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"   ✅ Test accuracy: {test_accuracy:.2%}")

        # Detailed classification report
        class_report = classification_report(
            y_test, test_predictions,
            target_names=[regime_names.get(i, f"Regime {i}") for i in unique_regimes],
            output_dict=True,
            zero_division=0
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, test_predictions)

        # Simulated trading performance based on regime predictions
        print(f"\n💰 Simulating regime-based trading strategy...")

        # Strategy: Go long in uptrend regimes (3, 4), stay out otherwise
        test_returns = raw_returns[train_size+val_size:]

        # Regime-based positions
        regime_positions = np.isin(test_predictions, [3, 4]).astype(float)

        # Calculate returns
        regime_strategy_returns = regime_positions * test_returns
        regime_cumulative = np.cumprod(1 + regime_strategy_returns) - 1
        regime_total_return = regime_cumulative[-1] if len(regime_cumulative) > 0 else 0

        # Annualized metrics
        trading_days = len(regime_strategy_returns)
        years = trading_days / 252

        if years > 0:
            regime_annualized = (1 + regime_total_return) ** (1 / years) - 1
        else:
            regime_annualized = 0

        # Sharpe ratio
        if len(regime_strategy_returns) > 0 and np.std(regime_strategy_returns) > 0:
            regime_sharpe = np.mean(regime_strategy_returns) / np.std(regime_strategy_returns) * np.sqrt(252)
        else:
            regime_sharpe = 0

        # Max drawdown
        cumulative_wealth = np.cumprod(1 + regime_strategy_returns)
        running_max = np.maximum.accumulate(cumulative_wealth)
        drawdown = (cumulative_wealth - running_max) / running_max
        regime_max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

        # Trade statistics
        regime_total_trades = int(np.sum(regime_positions))
        regime_winning_trades = int(np.sum((regime_positions > 0) & (test_returns > 0)))
        regime_win_rate = regime_winning_trades / regime_total_trades if regime_total_trades > 0 else 0

        print(f"   📈 Regime strategy performance:")
        print(f"      Total return: {regime_total_return:.2%}")
        print(f"      Annualized return: {regime_annualized:.2%}")
        print(f"      Sharpe ratio: {regime_sharpe:.2f}")
        print(f"      Max drawdown: {regime_max_dd:.2%}")
        print(f"      Win rate: {regime_win_rate:.2%}")
        print(f"      Total trades: {regime_total_trades}")

        result = {
            'ticker': ticker,
            'status': 'success',
            'regime_distribution': {int(k): int(v) for k, v in regime_dist.items()},
            'classification_metrics': {
                'validation_accuracy': float(val_accuracy),
                'test_accuracy': float(test_accuracy),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist()
            },
            'trading_performance': {
                'total_return': float(regime_total_return),
                'annualized_return': float(regime_annualized),
                'sharpe_ratio': float(regime_sharpe),
                'max_drawdown': float(regime_max_dd),
                'win_rate': float(regime_win_rate),
                'total_trades': int(regime_total_trades),
                'winning_trades': int(regime_winning_trades)
            }
        }

        print(f"\n✅ Testing complete for {ticker}")
        return result

    except Exception as e:
        print(f"❌ Error testing {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'ticker': ticker, 'status': 'error', 'error': str(e)}


def main():
    """Main regime classification testing workflow"""

    print("=" * 80)
    print("REGIME CLASSIFICATION TEST - PHASE 1 (PATH B.1)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Testing hypothesis: Regime prediction is more tractable than return prediction")
    print()

    # Same 5 ETFs as baseline tests
    test_tickers = ['OXLCG', 'HCXY', 'VGI', 'HYI', 'IGI']

    print(f"📦 Testing {len(test_tickers)} ETFs:")
    for i, ticker in enumerate(test_tickers, 1):
        print(f"   {i}. {ticker}")
    print()

    # Load ETF data
    print("🔄 Loading ETF data...")
    loader = ETFDataLoader()
    etf_data = loader.get_all_etf_data(days=400)
    print(f"   ✅ Loaded {len(etf_data)} ETFs")

    # Test each ETF
    all_results = []

    for idx, ticker in enumerate(test_tickers, 1):
        print(f"\n[{idx}/{len(test_tickers)}] Processing {ticker}...")
        result = test_regime_classification(ticker, etf_data)
        all_results.append(result)

    # Calculate aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    successful = [r for r in all_results if r['status'] == 'success']

    if successful:
        # Classification accuracy
        accuracies = [r['classification_metrics']['test_accuracy'] for r in successful]
        avg_accuracy = np.mean(accuracies)

        print(f"\n📊 Classification Performance:")
        print(f"   Average test accuracy: {avg_accuracy:.2%}")
        print(f"   Best accuracy: {np.max(accuracies):.2%}")
        print(f"   Worst accuracy: {np.min(accuracies):.2%}")

        # Trading performance
        returns = [r['trading_performance']['annualized_return'] for r in successful]
        sharpes = [r['trading_performance']['sharpe_ratio'] for r in successful]

        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpes)

        print(f"\n💰 Trading Performance:")
        print(f"   Average annualized return: {avg_return:.2%}")
        print(f"   Average Sharpe ratio: {avg_sharpe:.2f}")

        # Comparison to baseline
        print(f"\n📈 Comparison to XGBoost Baseline:")
        baseline_return = -0.0441  # From baseline test
        baseline_sharpe = -0.18

        improvement_return = avg_return - baseline_return
        improvement_sharpe = avg_sharpe - baseline_sharpe

        print(f"   Return improvement: {improvement_return:+.2%}")
        print(f"   Sharpe improvement: {improvement_sharpe:+.2f}")

        # Success assessment
        print("\n" + "=" * 80)
        print("PHASE 1 ASSESSMENT")
        print("=" * 80)

        success_criteria = {
            'classification_accuracy': avg_accuracy > 0.6,
            'positive_return': avg_return > 0,
            'positive_sharpe': avg_sharpe > 0,
            'better_than_baseline': avg_return > baseline_return
        }

        all_passed = all(success_criteria.values())

        if all_passed:
            print("\n✅ PHASE 1 SUCCESS - Regime classification is more tractable!")
            print(f"   ✅ Classification accuracy: {avg_accuracy:.2%} (target: >60%)")
            print(f"   ✅ Positive returns: {avg_return:.2%}")
            print(f"   ✅ Positive Sharpe: {avg_sharpe:.2f}")
            print(f"   ✅ Better than baseline: {improvement_return:+.2%}")
            print("\n📋 RECOMMENDATION: Proceed to Phase 2 (Enhanced Features)")
        elif success_criteria['better_than_baseline']:
            print("\n⚠️  PHASE 1 PARTIAL SUCCESS - Improvement but not profitable")
            print(f"   ✅ Better than baseline: {improvement_return:+.2%}")
            print(f"   ❌ Not yet profitable: {avg_return:.2%}")
            print("\n📋 RECOMMENDATION: Consider Phase 2 with enhanced features")
        else:
            print("\n❌ PHASE 1 FAILURE - Regime classification also struggles")
            print(f"   ❌ Worse than baseline: {improvement_return:.2%}")
            print("\n📋 RECOMMENDATION: Consider Path C (Loss Function) or Path D (Abandon)")

    # Save results
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f"regime_classification_{timestamp}.json"

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_type': 'regime_classification_phase1',
            'tickers_tested': test_tickers,
            'results': all_results,
            'aggregate_metrics': {
                'average_accuracy': float(avg_accuracy) if successful else 0,
                'average_return': float(avg_return) if successful else 0,
                'average_sharpe': float(avg_sharpe) if successful else 0,
                'improvement_vs_baseline': {
                    'return': float(improvement_return) if successful else 0,
                    'sharpe': float(improvement_sharpe) if successful else 0
                }
            },
            'success_criteria': success_criteria if successful else {}
        }, f, indent=2)

    print(f"\n💾 Saved results to {results_path}")

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
