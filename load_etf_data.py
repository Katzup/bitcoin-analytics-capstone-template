#!/usr/bin/env python3
"""
ETF Data Loader for Visual Trading System
Loads top-scoring ETFs from stock recommender cache and retrieves OHLC data from PostgreSQL
"""

import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add stock recommender to path
STOCK_RECOMMENDER_PATH = Path(__file__).parent.parent / 'Stock_Recommender_Projects' / 'ChatGPTAI_Stock_Recommender'
sys.path.insert(0, str(STOCK_RECOMMENDER_PATH))

from historical_data_manager import HistoricalDataManager


class ETFDataLoader:
    """Load ETF OHLC data for Visual Trading System"""

    def __init__(self, cache_path=None):
        """
        Initialize ETF data loader

        Args:
            cache_path: Path to top_25_etfs.json (defaults to stock recommender cache)
        """
        if cache_path is None:
            cache_path = STOCK_RECOMMENDER_PATH / 'cache_data' / 'top_25_etfs.json'

        self.cache_path = Path(cache_path)
        self.etf_list = self._load_etf_list()

    def _load_etf_list(self):
        """Load top ETFs from cache"""
        with open(self.cache_path) as f:
            etfs = json.load(f)

        # Extract tickers
        tickers = [etf['ticker'] for etf in etfs]
        print(f"📊 Loaded {len(tickers)} top-scoring ETFs from cache")
        return tickers

    def get_etf_data(self, ticker, days=365):
        """
        Get OHLC data for a specific ETF

        Args:
            ticker: ETF ticker symbol
            days: Number of days of historical data

        Returns:
            pandas.DataFrame with columns: date, open, high, low, close, volume
        """
        with HistoricalDataManager() as db:
            # Get historical data from PostgreSQL
            tuples = db.get_historical_data(ticker, days=days)

            if not tuples:
                print(f"⚠️  No data found for {ticker}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                tuples,
                columns=['date', 'open', 'high', 'low', 'close', 'volume', 'vwap']
            )

            # Sort by date ascending (required for Visual Trading System)
            df = df.sort_values('date')

            # Drop VWAP column (not needed for basic OHLC)
            df = df.drop('vwap', axis=1)

        # Convert PostgreSQL Decimal types to float64 for NumPy/XGBoost compatibility
        df = df.astype({
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        })

            # Set date as index
            df.set_index('date', inplace=True)

            return df

    def get_all_etf_data(self, days=365, limit=None):
        """
        Get OHLC data for all top ETFs

        Args:
            days: Number of days of historical data
            limit: Limit number of ETFs (for testing)

        Returns:
            dict: {ticker: DataFrame}
        """
        etf_data = {}
        etfs_to_load = self.etf_list[:limit] if limit else self.etf_list

        print(f"\n🔄 Loading OHLC data for {len(etfs_to_load)} ETFs...")

        for idx, ticker in enumerate(etfs_to_load, 1):
            df = self.get_etf_data(ticker, days=days)

            if not df.empty:
                etf_data[ticker] = df
                print(f"   [{idx:2d}/{len(etfs_to_load)}] {ticker}: {len(df)} days of data")
            else:
                print(f"   [{idx:2d}/{len(etfs_to_load)}] {ticker}: ❌ No data")

        print(f"\n✅ Loaded {len(etf_data)} ETFs successfully")
        return etf_data

    def save_etf_data(self, ticker, df, output_dir='data'):
        """
        Save ETF data to CSV for Visual Trading System

        Args:
            ticker: ETF ticker
            df: DataFrame with OHLC data
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        csv_path = output_path / f"{ticker}_ohlc.csv"
        df.to_csv(csv_path)
        print(f"💾 Saved {ticker} data to {csv_path}")

        return csv_path


def main():
    """Example usage"""
    print("=" * 60)
    print("ETF DATA LOADER FOR VISUAL TRADING SYSTEM")
    print("=" * 60)

    # Initialize loader
    loader = ETFDataLoader()

    # Example 1: Load single ETF
    print("\n📊 Example 1: Load single ETF (HSBH)")
    df = loader.get_etf_data('HSBH', days=365)

    if not df.empty:
        print(f"\nData shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())

    # Example 2: Load all top ETFs (limited to 5 for demo)
    print("\n\n📊 Example 2: Load multiple ETFs (first 5)")
    etf_data = loader.get_all_etf_data(days=365, limit=5)

    print("\n📋 Summary:")
    for ticker, df in etf_data.items():
        print(f"   {ticker}: {len(df)} days, price range ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Example 3: Save to CSV
    print("\n\n📊 Example 3: Save ETF data to CSV")
    if etf_data:
        first_ticker = list(etf_data.keys())[0]
        csv_path = loader.save_etf_data(first_ticker, etf_data[first_ticker])

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Use this loader in your Visual Trading System:
   from load_etf_data import ETFDataLoader
   loader = ETFDataLoader()
   df = loader.get_etf_data('HSBH', days=400)

2. Integrate with dataset.py:
   dataset = StockImageDataset(df, lookback=60, horizon=5)

3. Train models on top-scoring ETFs:
   for ticker, df in loader.get_all_etf_data().items():
       train_model(ticker, df)

4. Backtest trading strategies:
   from backtest import Backtester
   backtester = Backtester(model, df)
   results = backtester.run()
""")


if __name__ == '__main__':
    main()
