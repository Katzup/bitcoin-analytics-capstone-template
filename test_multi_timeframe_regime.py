#!/usr/bin/env python3
"""
Multi-Timeframe Regime Classification Testing
Tests regime prediction across multiple time horizons: 20, 50, 100, 150, 200 days
Implements hierarchical trading strategy integrating signals across timeframes
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json
import sys

from validation import PreTrainingValidator, validate_before_training
from data_requirements import DataRequirementsChecker, DataRequirement, validate_data_requirements
from load_etf_data import ETFDataLoader


def define_market_regimes(returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    Define market regimes based on price action and volatility

    Regimes:
    0 = Strong Downtrend (returns < -0.5 std)
    1 = Weak Downtrend (returns between -0.5 and 0 std)
    2 = Sideways/Neutral (returns between -0.2 and +0.2 std)
    3 = Weak Uptrend (returns between 0 and +0.5 std)
    4 = Strong Uptrend (returns > +0.5 std)
    5 = High Volatility (volatility > 1.5 std, any direction)
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


def create_features_for_horizon(df: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create features and regime labels for specific prediction horizon

    Uses adaptive lookback: max(60, 3 * horizon) to ensure sufficient context

    Args:
        df: DataFrame with OHLCV data
        horizon: Prediction horizon in days (20, 50, 100, 150, 200)

    Returns:
        features: (N, 306) array of feature vectors
        regime_labels: (N,) array of regime classifications
        raw_returns: (N,) array of actual returns for validation
    """
    # Adaptive lookback: use 3x horizon, minimum 60 days
    lookback = max(60, 3 * horizon)

    features = []
    raw_returns = []
    volatilities = []

    for i in range(lookback, len(df) - horizon):
        # Get lookback window
        window = df.iloc[i-lookback:i]

        # For longer horizons, use only recent 60 days for features
        # (keeps feature dimensionality consistent)
        feature_window = window.tail(60)

        # Flatten OHLCV data (300 features from 60 days)
        ohlcv = feature_window[['open', 'high', 'low', 'close', 'volume']].values.astype(float).flatten()

        # Calculate technical features (6 features)
        close_prices = feature_window['close'].values.astype(float)
        returns_5d = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) >= 5 else 0
        returns_10d = (close_prices[-1] - close_prices[-10]) / close_prices[-10] if len(close_prices) >= 10 else 0
        returns_20d = (close_prices[-1] - close_prices[-20]) / close_prices[-20] if len(close_prices) >= 20 else 0
        volatility = float(np.std(close_prices))
        volumes = feature_window['volume'].values.astype(float)
        volume_avg = np.mean(volumes)
        volume_std = np.std(volumes)

        # Combine all features
        feature_vector = np.concatenate([
            ohlcv,
            [returns_5d, returns_10d, returns_20d],
            [volatility, volume_avg, volume_std]
        ])

        # Target: N-day forward return (for regime definition)
        future_price = float(df.iloc[i + horizon]['close'])
        current_price = float(df.iloc[i]['close'])
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


def hierarchical_trading_strategy(predictions_dict: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Hierarchical multi-timeframe trading strategy

    Logic:
    - Long-term (100/150/200 day average) determines overall trend direction
    - Medium-term (50 day) confirms trend
    - Short-term (20 day) times entry/exit

    Position sizing:
    - Strong signal (all timeframes agree): Full position
    - Medium signal (long + medium agree): Half position
    - Weak signal (conflicting): No position

    Args:
        predictions_dict: {horizon: regime_predictions_array}
                         Keys: 20, 50, 100, 150, 200

    Returns:
        positions: (N,) array of position sizes (0.0, 0.5, 1.0)
    """
    # Get predictions for each timeframe
    pred_20 = predictions_dict[20]
    pred_50 = predictions_dict[50]
    pred_100 = predictions_dict[100]
    pred_150 = predictions_dict[150]
    pred_200 = predictions_dict[200]

    # Determine sample size (use shortest sequence)
    n_samples = min(len(pred_20), len(pred_50), len(pred_100), len(pred_150), len(pred_200))

    # Truncate all to same length
    pred_20 = pred_20[:n_samples]
    pred_50 = pred_50[:n_samples]
    pred_100 = pred_100[:n_samples]
    pred_150 = pred_150[:n_samples]
    pred_200 = pred_200[:n_samples]

    positions = np.zeros(n_samples)

    for i in range(n_samples):
        # Long-term consensus: average of 100/150/200 day regimes
        long_term_avg = (pred_100[i] + pred_150[i] + pred_200[i]) / 3.0

        # Classify timeframe signals
        # Uptrend regimes: 3 (weak up), 4 (strong up)
        long_term_bullish = long_term_avg >= 3.0
        medium_term_bullish = pred_50[i] >= 3
        short_term_bullish = pred_20[i] >= 3

        # Position sizing based on timeframe agreement
        if long_term_bullish and medium_term_bullish and short_term_bullish:
            # All timeframes bullish: Full position
            positions[i] = 1.0
        elif long_term_bullish and medium_term_bullish:
            # Long + medium bullish, short-term uncertain: Half position
            positions[i] = 0.5
        elif long_term_bullish and short_term_bullish:
            # Long + short bullish, medium-term uncertain: Half position
            positions[i] = 0.5
        else:
            # Conflicting signals or bearish: No position
            positions[i] = 0.0

    return positions


def calculate_performance_metrics(positions: np.ndarray, returns: np.ndarray,
                                 timeframe_label: str) -> Dict:
    """Calculate trading performance metrics"""

    # Strategy returns
    strategy_returns = positions * returns

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

    # Trade statistics
    trades_taken = np.sum(positions > 0)
    winning_trades = np.sum((positions > 0) & (returns > 0))
    win_rate = winning_trades / trades_taken if trades_taken > 0 else 0

    # Win/Loss ratio
    winning_mask = (positions > 0) & (returns > 0)
    losing_mask = (positions > 0) & (returns <= 0)

    avg_win = np.mean(returns[winning_mask]) if np.sum(winning_mask) > 0 else 0
    avg_loss = np.mean(returns[losing_mask]) if np.sum(losing_mask) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'timeframe': timeframe_label,
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'total_trades': int(trades_taken),
        'winning_trades': int(winning_trades),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'win_loss_ratio': float(win_loss_ratio)
    }


def test_multi_timeframe_regime(ticker: str, etf_data: Dict) -> Dict:
    """
    Test multi-timeframe regime classification for single ETF

    Tests regimes for 20, 50, 100, 150, 200-day horizons
    Implements hierarchical trading strategy
    """
    print(f"\n{'='*70}")
    print(f"Testing Multi-Timeframe Regime Classification: {ticker}")
    print(f"{'='*70}")

    try:
        # Load data
        if ticker not in etf_data:
            return {'ticker': ticker, 'status': 'error', 'error': 'No data available'}

        df = etf_data[ticker]
        print(f"   Loaded {len(df)} days of data")

        # Test horizons: short (20), medium (50), long (100, 150, 200)
        horizons = [20, 50, 100, 150, 200]

        models = {}
        test_predictions = {}
        test_returns_dict = {}
        regime_accuracies = {}

        print(f"\n   Training models for {len(horizons)} timeframes...")

        for horizon in horizons:
            print(f"\n   Horizon: {horizon} days")
            print(f"   {'─'*50}")


        # Check data sufficiency for this horizon
        requirement = DataRequirement(
            lookback=max(60, 3 * horizon),
            horizon=horizon,
            min_samples_per_class=5
        )
        checker = DataRequirementsChecker(requirement)
        availability = checker.check_dataframe_sufficiency(df, ticker)

        if not availability.sufficient:
            print(f"      ⚠️  Insufficient data for horizon {horizon}")
            print(f"         Required: {availability.required_samples}, Available: {availability.available_samples}")
            print(f"         Shortage: {availability.shortage} samples")
            continue

        # Create features for this horizon
        features, regime_labels, raw_returns = create_features_for_horizon(df, horizon)
        print(f"      Generated {len(features)} samples")

        # Train/val/test split (70/15/15)
        train_size = int(0.7 * len(features))
        val_size = int(0.15 * len(features))

        X_train = features[:train_size]
        y_train = regime_labels[:train_size]

        X_val = features[train_size:train_size + val_size]
        y_val = regime_labels[train_size:train_size + val_size]

        X_test = features[train_size + val_size:]
        y_test = regime_labels[train_size + val_size:]
        test_returns = raw_returns[train_size + val_size:]

        # Validate splits before training
        validator = PreTrainingValidator(min_samples_per_class=5)
        validation_result = validator.validate_splits(
            X_train, y_train, X_val, y_val, X_test, y_test, horizon
        )

        if not validation_result.passed:
            print(f"      ❌ Validation failed for horizon {horizon}")
            for error in validation_result.errors:
                print(f"         - {error}")
            continue

        if validation_result.warnings:
            print(f"      ⚠️  Warnings:")
            for warning in validation_result.warnings:
                print(f"         - {warning}")

        print(f"      Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Train XGBoost classifier
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

        # Predict on test set
        predictions = model.predict(X_test)

        # Calculate regime classification accuracy
        accuracy = np.mean(predictions == y_test)
        regime_accuracies[horizon] = float(accuracy)

        print(f"      Regime accuracy: {accuracy:.2%}")

        # Store for hierarchical strategy
        models[horizon] = model
        test_predictions[horizon] = predictions
        test_returns_dict[horizon] = test_returns

        print(f"\n   {'='*70}")
        print(f"   Implementing Hierarchical Trading Strategy")
        print(f"   {'='*70}")

        # Get hierarchical positions
        positions = hierarchical_trading_strategy(test_predictions)

        # Align returns (use shortest sequence)
        n_samples = len(positions)
        aligned_returns = test_returns_dict[20][:n_samples]  # Use 20-day returns for position evaluation

        # Calculate hierarchical strategy metrics
        hierarchical_metrics = calculate_performance_metrics(
            positions, aligned_returns, "Hierarchical Multi-Timeframe"
        )

        # Also test individual timeframe strategies for comparison
        individual_metrics = []

        for horizon in horizons:
            # Simple strategy: long when predicted regime is uptrend (3 or 4)
            individual_positions = np.isin(test_predictions[horizon], [3, 4]).astype(float)
            individual_positions = individual_positions[:n_samples]  # Align

            metrics = calculate_performance_metrics(
                individual_positions, aligned_returns, f"{horizon}-day Only"
            )
            individual_metrics.append(metrics)

        # Print results
        print(f"\n   {'='*70}")
        print(f"   REGIME CLASSIFICATION ACCURACY")
        print(f"   {'='*70}")
        for horizon in horizons:
            print(f"   {horizon:3d}-day: {regime_accuracies[horizon]:>6.2%}")

        print(f"\n   {'='*70}")
        print(f"   TRADING PERFORMANCE COMPARISON")
        print(f"   {'='*70}")

        print(f"\n   {'Strategy':<30} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10} {'W/L Ratio':>10} {'Trades':>8}")
        print(f"   {'-'*80}")

        # Hierarchical
        m = hierarchical_metrics
        print(f"   {m['timeframe']:<30} {m['annualized_return']:>9.2%} {m['sharpe_ratio']:>8.2f} "
              f"{m['win_rate']:>9.2%} {m['win_loss_ratio']:>10.2f} {m['total_trades']:>8}")

        print(f"   {'-'*80}")

        # Individual timeframes
        for m in individual_metrics:
            print(f"   {m['timeframe']:<30} {m['annualized_return']:>9.2%} {m['sharpe_ratio']:>8.2f} "
                  f"{m['win_rate']:>9.2%} {m['win_loss_ratio']:>10.2f} {m['total_trades']:>8}")

        # Assessment
        print(f"\n   {'='*70}")
        print(f"   ASSESSMENT")
        print(f"   {'='*70}")

        hierarchical_positive = hierarchical_metrics['annualized_return'] > 0
        hierarchical_sharpe_positive = hierarchical_metrics['sharpe_ratio'] > 0

        if hierarchical_positive and hierarchical_sharpe_positive:
            print(f"   ✅ HIERARCHICAL STRATEGY ACHIEVES POSITIVE RETURNS")
            print(f"      Annualized Return: {hierarchical_metrics['annualized_return']:.2%}")
            print(f"      Sharpe Ratio: {hierarchical_metrics['sharpe_ratio']:.2f}")
            print(f"      Win/Loss Ratio: {hierarchical_metrics['win_loss_ratio']:.2f}")
        else:
            print(f"   ❌ HIERARCHICAL STRATEGY STILL NEGATIVE")
            print(f"      Annualized Return: {hierarchical_metrics['annualized_return']:.2%}")
            print(f"      Sharpe Ratio: {hierarchical_metrics['sharpe_ratio']:.2f}")

        return {
            'ticker': ticker,
            'status': 'success',
            'regime_accuracies': regime_accuracies,
            'hierarchical_metrics': hierarchical_metrics,
            'individual_metrics': individual_metrics,
            'n_samples': int(n_samples),
            'position_breakdown': {
                'full_position': int(np.sum(positions == 1.0)),
                'half_position': int(np.sum(positions == 0.5)),
                'no_position': int(np.sum(positions == 0.0))
            }
        }

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'ticker': ticker, 'status': 'error', 'error': str(e)}


def main():
    """Main testing workflow"""

    print("="*80)
    print("MULTI-TIMEFRAME REGIME CLASSIFICATION TEST")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Testing Horizons: 20, 50, 100, 150, 200 days")
    print("Strategy: Hierarchical (long-term determines trend, short-term times entry)")
    print()

    # Test on same 5 ETFs as baseline
    test_tickers = ['OXLCG', 'HCXY', 'VGI', 'HYI', 'IGI']

    print(f"Testing {len(test_tickers)} ETFs:")
    for i, ticker in enumerate(test_tickers, 1):
        print(f"   {i}. {ticker}")
    print()

    # Load data
    print("Loading ETF data...")
    loader = ETFDataLoader()
    etf_data = loader.get_all_etf_data(days=600)  # Need more data for 200-day predictions
    print(f"   ✅ Loaded {len(etf_data)} ETFs")

    # Test each ticker
    all_results = []

    for idx, ticker in enumerate(test_tickers, 1):
        print(f"\n[{idx}/{len(test_tickers)}]")
        result = test_multi_timeframe_regime(ticker, etf_data)
        all_results.append(result)

    # Calculate summary statistics
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL ETFS")
    print("="*80)

    successful = [r for r in all_results if r['status'] == 'success']

    if successful:
        # Average metrics
        avg_hierarchical_return = np.mean([r['hierarchical_metrics']['annualized_return']
                                          for r in successful])
        avg_hierarchical_sharpe = np.mean([r['hierarchical_metrics']['sharpe_ratio']
                                          for r in successful])

        print(f"\nAverage Hierarchical Performance:")
        print(f"   Annualized Return: {avg_hierarchical_return:>8.2%}")
        print(f"   Sharpe Ratio: {avg_hierarchical_sharpe:>13.2f}")

        # Compare to baseline
        baseline_return = -0.0441  # From XGBoost baseline test
        baseline_sharpe = -0.18

        improvement_return = avg_hierarchical_return - baseline_return
        improvement_sharpe = avg_hierarchical_sharpe - baseline_sharpe

        print(f"\nImprovement vs XGBoost Baseline:")
        print(f"   Return Improvement: {improvement_return:>8.2%}")
        print(f"   Sharpe Improvement: {improvement_sharpe:>13.2f}")

        # Success assessment
        print("\n" + "="*80)
        print("PHASE 1 ASSESSMENT: Multi-Timeframe Regime Classification")
        print("="*80)

        if avg_hierarchical_return > 0 and avg_hierarchical_sharpe > 0:
            print("\n✅ PHASE 1 SUCCESS - Regime classification with multi-timeframe works!")
            print("\nRecommended Next Steps:")
            print("   1. Proceed to Phase 2: Add enhanced features (sentiment, fundamentals)")
            print("   2. Test if additional data sources improve regime prediction accuracy")
            print("   3. Validate on out-of-sample period before production deployment")
        elif avg_hierarchical_return > baseline_return:
            print("\n⚠️  PHASE 1 PARTIAL SUCCESS - Improvement but still negative")
            print("\nRecommended Next Steps:")
            print("   1. Try enhanced features (Path A) to improve regime accuracy")
            print("   2. Experiment with different position sizing rules")
            print("   3. Consider loss function redesign (Option C)")
        else:
            print("\n❌ PHASE 1 FAILURE - Multi-timeframe regime classification doesn't help")
            print("\nRecommended Next Steps:")
            print("   1. Enhanced features may still be necessary (Path A)")
            print("   2. Consider reformulating problem differently (Path B.2 or B.3)")
            print("   3. May need to abandon 5-day ETF prediction approach (Path D)")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f"multi_timeframe_regime_{timestamp}.json"

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_type': 'multi_timeframe_regime_classification',
            'horizons_tested': [20, 50, 100, 150, 200],
            'strategy': 'hierarchical',
            'tickers_tested': test_tickers,
            'results': all_results,
            'summary': {
                'avg_hierarchical_return': float(avg_hierarchical_return) if successful else None,
                'avg_hierarchical_sharpe': float(avg_hierarchical_sharpe) if successful else None,
                'improvement_vs_baseline': {
                    'return': float(improvement_return) if successful else None,
                    'sharpe': float(improvement_sharpe) if successful else None
                }
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to {results_path}")

    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
