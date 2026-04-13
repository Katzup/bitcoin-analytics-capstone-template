"""
P0 Runner - PostgreSQL Version

Same as p0_runner.py but uses PostgreSQL instead of SQLite.

Usage:
    export POLYGON_API_KEY='your_api_key'
    export DB_PASSWORD='your_db_password'
    python p0_runner_postgres.py
"""

import os
import sys
from datetime import datetime, timedelta
import torch
import config
from polygon_data import PolygonDataFetcher
from postgres_store import PostgresStore  # Use PostgreSQL instead of SQLite
from walkforward_backtest import WalkForwardBacktester


def run_p0_pipeline_postgres(
    ticker='SPY',
    days_history=730,
    train_window=252,
    test_window=21,
    num_epochs=30,
    force_refresh=False
):
    """
    Run P0 pipeline with PostgreSQL storage.

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
    print("Visual Trading System - P0 Pipeline (PostgreSQL)")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"History: {days_history} days")
    print(f"Storage: PostgreSQL")
    print("=" * 80)

    # Step 1: Initialize PostgreSQL store
    print("\n[Step 1/4] Connecting to PostgreSQL...")
    try:
        store = PostgresStore()
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("\nMake sure:")
        print("  - PostgreSQL is running")
        print("  - Database 'sentientedge' exists")
        print("  - DB_PASSWORD environment variable is set")
        return None

    # Check for existing data
    existing_data = store.get_bars(ticker)

    if existing_data.empty or force_refresh:
        print(f"  Fetching new data from Polygon...")

        # Check for API key
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            print("\n❌ ERROR: POLYGON_API_KEY environment variable not set")
            return None

        # Fetch data
        fetcher = PolygonDataFetcher(api_key)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')

        df = fetcher.fetch_daily_bars(ticker, start_date, end_date)

        if df.empty:
            print(f"❌ Failed to fetch data for {ticker}")
            return None

        # Store in PostgreSQL
        print(f"  Saving to PostgreSQL...")
        store.save_bars(df, replace=True)
        price_data = df
    else:
        print(f"  ✅ Found {len(existing_data)} bars in PostgreSQL")
        price_data = existing_data

    print(f"  Data range: {price_data['date'].min()} to {price_data['date'].max()}")

    # Step 2-4: Same as SQLite version
    print("\n[Step 2/4] Validating data...")
    min_required = config.LOOKBACK_WINDOW + config.PREDICTION_HORIZON + train_window + test_window
    if len(price_data) < min_required:
        print(f"❌ Insufficient data. Need {min_required} bars, have {len(price_data)}")
        return None

    print(f"  ✅ Sufficient data for backtesting")

    print("\n[Step 3/4] Running walk-forward backtest...")
    backtester = WalkForwardBacktester(
        price_data=price_data,
        lookback=config.LOOKBACK_WINDOW,
        horizon=config.PREDICTION_HORIZON,
        train_window=train_window,
        test_window=test_window,
        initial_capital=100000,
        position_size=1.0,
        transaction_cost=config.TRANSACTION_COST,
        prediction_threshold=config.PREDICTION_THRESHOLD,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )

    results = backtester.run(num_epochs=num_epochs, device=config.DEVICE)

    if not results or results.get('num_trades', 0) == 0:
        print("⚠️ Backtest completed but no trades were generated")
        return results

    print("\n[Step 4/4] Saving results...")
    if 'trades' in results and not results['trades'].empty:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        trade_file = f"{config.RESULTS_DIR}/{ticker}_trades_pg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['trades'].to_csv(trade_file, index=False)
        print(f"  ✅ Trades saved to {trade_file}")

    if 'equity_curve' in results and not results['equity_curve'].empty:
        equity_file = f"{config.RESULTS_DIR}/{ticker}_equity_pg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['equity_curve'].to_csv(equity_file, index=False)
        print(f"  ✅ Equity curve saved to {equity_file}")

    print("\n" + "=" * 80)
    print("P0 PIPELINE COMPLETE (PostgreSQL)")
    print("=" * 80)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Visual Trading System P0 Pipeline (PostgreSQL)')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker')
    parser.add_argument('--days', type=int, default=730, help='Days of history')
    parser.add_argument('--train-window', type=int, default=252, help='Training window')
    parser.add_argument('--test-window', type=int, default=21, help='Test window')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per window')
    parser.add_argument('--refresh', action='store_true', help='Force refresh data')

    args = parser.parse_args()

    results = run_p0_pipeline_postgres(
        ticker=args.ticker,
        days_history=args.days,
        train_window=args.train_window,
        test_window=args.test_window,
        num_epochs=args.epochs,
        force_refresh=args.refresh
    )

    if results is None:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
