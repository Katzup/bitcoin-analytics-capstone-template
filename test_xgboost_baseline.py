#!/usr/bin/env python3
"""
XGBoost Baseline Testing Script
Tests if traditional ML can achieve positive returns with same features as CNNs
Determines if problem is CNN-specific or universal (data/features insufficient)
"""

import xgboost as xgb
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from sklearn.model_selection import TimeSeriesSplit

from load_etf_data import ETFDataLoader
import config


def create_features(df: pd.DataFrame, lookback: int = 60,
                    horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create features from OHLCV data
    Same lookback as CNN but flattened to feature vector

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of days to look back
        horizon: Number of days forward for target

    Returns:
        features: (n_samples, n_features) array
        targets: (n_samples,) array of returns
    """
    features = []
    targets = []

    # Need at least lookback + horizon samples
    if len(df) < lookback + horizon:
        return np.array([]), np.array([])

    for i in range(lookback, len(df) - horizon):
        # Get lookback window
        window = df.iloc[i-lookback:i]

        # Flatten OHLCV data
        ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values.flatten()

        # Calculate simple technical features
        close_prices = window['close'].values

        # Returns over various periods
        returns_5d = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) >= 5 else 0
        returns_10d = (close_prices[-1] - close_prices[-10]) / close_prices[-10] if len(close_prices) >= 10 else 0
        returns_20d = (close_prices[-1] - close_prices[-20]) / close_prices[-20] if len(close_prices) >= 20 else 0

        # Volatility
        volatility = np.std(close_prices)

        # Volume features
        volumes = window['volume'].values
        volume_avg = np.mean(volumes)
        volume_std = np.std(volumes)

        # Combine all features
        feature_vector = np.concatenate([
            ohlcv,
            [returns_5d, returns_10d, returns_20d],
            [volatility, volume_avg, volume_std]
        ])

        # Target: N-day forward return
        future_price = df.iloc[i + horizon]['close']
        current_price = df.iloc[i]['close']
        target_return = (future_price - current_price) / current_price

        features.append(feature_vector)
        targets.append(float(target_return))

    return np.array(features), np.array(targets)


def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBRegressor:
    """
    Train XGBoost model with hyperparameter tuning

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Trained XGBoost model
    """
    print("🔧 Training XGBoost model...")

    # Configure model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10,
        eval_metric='rmse'
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Validation performance
    val_predictions = model.predict(X_val)
    val_mse = np.mean((val_predictions - y_val) ** 2)

    print(f"   ✅ Training complete")
    print(f"   Validation MSE: {val_mse:.6f}")

    return model


def calculate_backtest_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict:
    """
    Calculate trading performance metrics
    Same metrics as CNN backtesting for direct comparison
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
        'win_loss_ratio': float(win_loss_ratio),
        'num_predictions': len(predictions)
    }


def test_xgboost_baseline(ticker: str, etf_data: Dict,
                          lookback: int = 60, horizon: int = 5) -> Dict:
    """
    Test XGBoost baseline on single ETF

    Args:
        ticker: ETF ticker symbol
        etf_data: Dictionary of ETF data
        lookback: Days to look back
        horizon: Days forward for target

    Returns:
        dict: Test results
    """
    print(f"\n{'='*60}")
    print(f"Testing XGBoost Baseline: {ticker}")
    print(f"{'='*60}")

    try:
        # Get data
        if ticker not in etf_data:
            return {
                'ticker': ticker,
                'status': 'error',
                'error': f'No data available for {ticker}'
            }

        df = etf_data[ticker]

        # Create features
        print(f"🔄 Creating features (lookback={lookback}, horizon={horizon})...")
        X, y = create_features(df, lookback=lookback, horizon=horizon)

        if len(X) == 0:
            return {
                'ticker': ticker,
                'status': 'error',
                'error': 'Insufficient data for feature creation'
            }

        print(f"   ✅ Created {len(X)} samples with {X.shape[1]} features")

        # Same split as CNN: 70/15/15
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Train model
        model = train_xgboost_model(X_train, y_train, X_val, y_val)

        # Test predictions
        print(f"🔄 Generating test predictions...")
        test_predictions = model.predict(X_test)

        print(f"   ✅ Generated {len(test_predictions)} predictions")

        # Calculate metrics
        print(f"📊 Calculating performance metrics...")
        metrics = calculate_backtest_metrics(test_predictions, y_test)

        result = {
            'ticker': ticker,
            'status': 'success',
            'config': {
                'lookback': lookback,
                'horizon': horizon,
                'model_type': 'xgboost',
                'n_features': X.shape[1]
            },
            'backtest_metrics': metrics,
            'test_samples': len(X_test),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }

        # Print summary
        print(f"\n📈 XGBoost Backtest Results:")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")
        print(f"   Total Trades: {metrics['total_trades']}")

        print(f"\n✅ XGBoost test complete for {ticker}")
        return result

    except Exception as e:
        print(f"❌ Error testing {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }


def compare_to_cnn_ensemble(xgboost_results: Dict,
                            cnn_ensemble_path: Path) -> Dict:
    """
    Compare XGBoost results to CNN ensemble results

    Args:
        xgboost_results: XGBoost test results
        cnn_ensemble_path: Path to CNN ensemble results

    Returns:
        dict: Comparison analysis
    """
    print("\n" + "="*80)
    print("XGBOOST vs CNN ENSEMBLE COMPARISON")
    print("="*80)

    # Load CNN ensemble results
    try:
        with open(cnn_ensemble_path, 'r') as f:
            cnn_data = json.load(f)
            cnn_metrics = cnn_data['results']['ensemble_metrics']
    except Exception as e:
        print(f"⚠️  Could not load CNN ensemble results: {e}")
        cnn_metrics = None

    # Aggregate XGBoost results
    successful = [r for r in xgboost_results if r['status'] == 'success']

    if not successful:
        print("❌ No successful XGBoost tests to compare")
        return {'status': 'no_data'}

    avg_return = np.mean([r['backtest_metrics']['annualized_return'] for r in successful])
    avg_sharpe = np.mean([r['backtest_metrics']['sharpe_ratio'] for r in successful])
    avg_wl_ratio = np.mean([r['backtest_metrics']['win_loss_ratio'] for r in successful])

    print(f"\n📊 XGBoost Average Performance:")
    print(f"   Annualized Return: {avg_return:.2%}")
    print(f"   Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"   Win/Loss Ratio: {avg_wl_ratio:.2f}")

    if cnn_metrics:
        print(f"\n📊 CNN Ensemble Performance:")
        print(f"   Annualized Return: {cnn_metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {cnn_metrics['sharpe_ratio']:.2f}")
        print(f"   Win/Loss Ratio: {cnn_metrics['win_loss_ratio']:.2f}")

        # Comparison
        return_diff = avg_return - cnn_metrics['annualized_return']
        sharpe_diff = avg_sharpe - cnn_metrics['sharpe_ratio']

        print(f"\n🔍 XGBoost vs CNN Difference:")
        print(f"   Return Difference: {return_diff:+.2%}")
        print(f"   Sharpe Difference: {sharpe_diff:+.2f}")

        comparison = {
            'xgboost_avg_return': float(avg_return),
            'xgboost_avg_sharpe': float(avg_sharpe),
            'xgboost_avg_wl_ratio': float(avg_wl_ratio),
            'cnn_ensemble_return': float(cnn_metrics['annualized_return']),
            'cnn_ensemble_sharpe': float(cnn_metrics['sharpe_ratio']),
            'cnn_ensemble_wl_ratio': float(cnn_metrics['win_loss_ratio']),
            'return_difference': float(return_diff),
            'sharpe_difference': float(sharpe_diff),
            'xgboost_better': bool(avg_return > cnn_metrics['annualized_return'])
        }
    else:
        comparison = {
            'xgboost_avg_return': float(avg_return),
            'xgboost_avg_sharpe': float(avg_sharpe),
            'xgboost_avg_wl_ratio': float(avg_wl_ratio),
            'cnn_comparison': 'unavailable'
        }

    return comparison


def main():
    """Main XGBoost baseline testing workflow"""

    print("=" * 80)
    print("XGBOOST BASELINE TESTING: OPTION A.3")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test same 5 ETFs as CNN models
    test_tickers = ['OXLCG', 'HCXY', 'VGI', 'HYI', 'IGI']

    print(f"📦 Testing {len(test_tickers)} ETFs with XGBoost:")
    for i, ticker in enumerate(test_tickers, 1):
        print(f"   {i}. {ticker}")
    print()

    # Load ETF data
    print("🔄 Loading ETF data...")
    loader = ETFDataLoader()
    etf_data = loader.get_all_etf_data(days=400)
    print(f"   ✅ Loaded {len(etf_data)} ETFs")
    print()

    # Test each ETF
    all_results = []

    for idx, ticker in enumerate(test_tickers, 1):
        print(f"\n[{idx}/{len(test_tickers)}] Processing {ticker}...")
        result = test_xgboost_baseline(ticker, etf_data, lookback=60, horizon=5)
        all_results.append(result)

    # Compare to CNN ensemble
    cnn_ensemble_path = Path(config.RESULTS_DIR) / 'ensemble_test_20260103_225157.json'
    comparison = compare_to_cnn_ensemble(all_results, cnn_ensemble_path)

    # Save results
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f"xgboost_baseline_{timestamp}.json"

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_type': 'xgboost_baseline',
            'tickers_tested': test_tickers,
            'results': all_results,
            'comparison': comparison
        }, f, indent=2)

    print(f"\n💾 Saved results to {results_path}")

    # Generate recommendation
    print("\n" + "="*80)
    print("DECISION FRAMEWORK")
    print("="*80)

    successful = [r for r in all_results if r['status'] == 'success']

    if successful:
        avg_return = np.mean([r['backtest_metrics']['annualized_return'] for r in successful])
        avg_wl = np.mean([r['backtest_metrics']['win_loss_ratio'] for r in successful])

        positive_returns = avg_return > 0
        good_wl_ratio = avg_wl >= 1.0

        if positive_returns and good_wl_ratio:
            print("\n✅ XGBOOST SUCCESSFUL - Problem is CNN architecture!")
            print("\nRecommendations:")
            print("   1. ABANDON CNN approach - architecture fundamentally unsuitable")
            print("   2. Use XGBoost for production trading system")
            print("   3. Focus on feature engineering to improve XGBoost further")
            print("   4. Consider ensemble of XGBoost models (different hyperparameters)")
        elif positive_returns and not good_wl_ratio:
            print("\n⚠️  XGBOOST ACHIEVES POSITIVE RETURNS BUT W/L RATIO POOR")
            print("\nRecommendations:")
            print("   1. XGBoost better than CNN but needs improvement")
            print("   2. Implement position sizing based on confidence")
            print("   3. Add risk management (stop losses, position limits)")
            print("   4. Consider ensemble approach with multiple models")
        else:
            print("\n❌ XGBOOST ALSO FAILS - Problem is data/features, not architecture")
            print("\nRecommendations:")
            print("   1. Problem is fundamental - insufficient signal in OHLCV data")
            print("   2. Need better features: sentiment, fundamentals, order flow")
            print("   3. Consider multi-source data integration")
            print("   4. Re-evaluate whether 5-day returns are predictable at all")

    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
