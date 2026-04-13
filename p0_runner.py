"""
P0 Runner - End-to-End Pipeline for Visual Trading System

Complete pipeline:
1. Fetch SPY data from Polygon API
2. Store in SQLite
3. Create binary GAF dataset
4. Run walk-forward backtest
5. Display results

Usage:
    export POLYGON_API_KEY='your_api_key'
    python p0_runner.py
"""

import os
import sys
from datetime import datetime, timedelta
import torch
import config
from polygon_data import PolygonDataFetcher
from data_store import DataStore
from walkforward_backtest import WalkForwardBacktester


def run_p0_pipeline(
    ticker='SPY',
    days_history=730,
    train_window=252,
    test_window=21,
    num_epochs=30,
    force_refresh=False
):
    """
    Run complete P0 pipeline.

    Args:
        ticker: Stock ticker to trade
        days_history: Days of historical data to fetch
        train_window: Training window size in days
        test_window: Test window size in days
        num_epochs: Training epochs per window
        force_refresh: If True, re-fetch data even if exists

    Returns:
        dict: Backtest results
    """
    print("=" * 80)
    print("Visual Trading System - P0 Pipeline")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"History: {days_history} days")
    print(f"Train Window: {train_window} days")
    print(f"Test Window: {test_window} days")
    print(f"Epochs per Window: {num_epochs}")
    print(f"Device: {config.DEVICE}")
    print("=" * 80)

    # Step 1: Check for existing data
    print("\n[Step 1/4] Checking for existing data...")
    store = DataStore()
    existing_data = store.get_bars(ticker)

    if existing_data.empty or force_refresh:
        print(f"  No existing data found. Fetching from Polygon...")

        # Check for API key
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            print("\n❌ ERROR: POLYGON_API_KEY environment variable not set")
            print("Set it with: export POLYGON_API_KEY='your_api_key'")
            print("\nAlternatively, you can:")
            print("  1. Get a free API key from https://polygon.io/")
            print("  2. Set the environment variable")
            print("  3. Re-run this script")
            return None

        # Fetch data
        fetcher = PolygonDataFetcher(api_key)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')

        df = fetcher.fetch_daily_bars(ticker, start_date, end_date)

        if df.empty:
            print(f"❌ Failed to fetch data for {ticker}")
            return None

        # Store in database
        print(f"  Saving to database...")
        store.save_bars(df, replace=True)
        print(f"  ✅ Data saved to database")

        price_data = df
    else:
        print(f"  ✅ Found {len(existing_data)} bars in database")
        price_data = existing_data

    # Ensure we have required columns
    if 'close' not in price_data.columns or 'volume' not in price_data.columns:
        print("❌ Data missing required columns: 'close', 'volume'")
        return None

    print(f"  Data range: {price_data['date'].min()} to {price_data['date'].max()}")
    print(f"  Total bars: {len(price_data)}")

    # Step 2: Data validation
    print("\n[Step 2/4] Validating data...")
    min_required = config.LOOKBACK_WINDOW + config.PREDICTION_HORIZON + train_window + test_window
    if len(price_data) < min_required:
        print(f"❌ Insufficient data. Need {min_required} bars, have {len(price_data)}")
        print(f"  Try increasing days_history parameter or using a different ticker")
        return None

    print(f"  ✅ Sufficient data for backtesting")

    # Step 3: Run walk-forward backtest
    print("\n[Step 3/4] Running walk-forward backtest...")
    print(f"  This may take several minutes depending on data size...")

    backtester = WalkForwardBacktester(
        price_data=price_data,
        lookback=config.LOOKBACK_WINDOW,
        horizon=config.PREDICTION_HORIZON,
        train_window=train_window,
        test_window=test_window,
        initial_capital=100000,
        position_size=1.0,  # Use full capital per trade
        transaction_cost=config.TRANSACTION_COST,
        prediction_threshold=config.PREDICTION_THRESHOLD,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )

    results = backtester.run(num_epochs=num_epochs, device=config.DEVICE)

    if not results or results.get('num_trades', 0) == 0:
        print("⚠️ Backtest completed but no trades were generated")
        return results

    # Step 4: Save results
    print("\n[Step 4/4] Saving results...")

    # Save trade log
    if 'trades' in results and not results['trades'].empty:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        trade_file = f"{config.RESULTS_DIR}/{ticker}_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['trades'].to_csv(trade_file, index=False)
        print(f"  ✅ Trades saved to {trade_file}")

    # Save equity curve
    if 'equity_curve' in results and not results['equity_curve'].empty:
        equity_file = f"{config.RESULTS_DIR}/{ticker}_equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['equity_curve'].to_csv(equity_file, index=False)
        print(f"  ✅ Equity curve saved to {equity_file}")

    print("\n" + "=" * 80)
    print("P0 PIPELINE COMPLETE")
    print("=" * 80)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Visual Trading System P0 Pipeline')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker (default: SPY)')
    parser.add_argument('--days', type=int, default=730, help='Days of history (default: 730)')
    parser.add_argument('--train-window', type=int, default=252, help='Training window (default: 252)')
    parser.add_argument('--test-window', type=int, default=21, help='Test window (default: 21)')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per window (default: 30)')
    parser.add_argument('--refresh', action='store_true', help='Force refresh data from API')

    args = parser.parse_args()

    results = run_p0_pipeline(
        ticker=args.ticker,
        days_history=args.days,
        train_window=args.train_window,
        test_window=args.test_window,
        num_epochs=args.epochs,
        force_refresh=args.refresh
    )

    if results is None:
        print("\n❌ Pipeline failed")
        sys.exit(1)
    else:
        print("\n✅ Pipeline succeeded")
        sys.exit(0)


if __name__ == '__main__':
    main()
