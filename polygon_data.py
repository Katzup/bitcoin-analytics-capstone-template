"""
Polygon.io data ingestion for Visual Trading System.

Fetches daily OHLCV data from Polygon API and stores in SQLite.
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional, List
import time


class PolygonDataFetcher:
    """Fetch daily OHLCV data from Polygon.io API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon data fetcher.

        Args:
            api_key: Polygon API key (defaults to POLYGON_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found. Set environment variable or pass to constructor.")

        self.base_url = "https://api.polygon.io/v2"

    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'SPY')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of results per request

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        url = f"{self.base_url}/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"

        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': limit,
            'apiKey': self.api_key
        }

        print(f"Fetching {ticker} data from {start_date} to {end_date}...")

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('resultsCount', 0) == 0:
                print(f"⚠️ No data returned for {ticker}")
                return pd.DataFrame()

            results = data.get('results', [])

            # Convert to DataFrame
            df = pd.DataFrame(results)

            # Rename columns to standard OHLCV format
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'n': 'transactions'
            })

            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['timestamp'].dt.date

            # Add ticker column
            df['ticker'] = ticker

            # Select and order columns
            columns = ['ticker', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[columns]

            print(f"✅ Fetched {len(df)} bars for {ticker}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        delay: float = 0.1
    ) -> pd.DataFrame:
        """
        Fetch data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            delay: Delay between API calls in seconds (rate limiting)

        Returns:
            Combined DataFrame for all tickers
        """
        all_data = []

        for ticker in tickers:
            df = self.fetch_daily_bars(ticker, start_date, end_date)
            if not df.empty:
                all_data.append(df)

            # Rate limiting
            time.sleep(delay)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['ticker', 'date'])

        print(f"\n✅ Total bars fetched: {len(combined)} across {len(tickers)} tickers")

        return combined

    def get_recent_data(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch recent data for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history to fetch

        Returns:
            DataFrame with OHLCV data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        return self.fetch_daily_bars(ticker, start_date, end_date)


def main():
    """Test data fetching."""
    print("=" * 80)
    print("Polygon Data Fetcher Test")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("\n⚠️ POLYGON_API_KEY environment variable not set.")
        print("Set it with: export POLYGON_API_KEY='your_api_key'")
        return

    # Initialize fetcher
    fetcher = PolygonDataFetcher()

    # Fetch SPY data for last 2 years
    print("\nFetching SPY data...")
    df = fetcher.get_recent_data('SPY', days=730)

    if not df.empty:
        print(f"\nData shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        print("\nData info:")
        print(df.info())

    return df


if __name__ == '__main__':
    main()
