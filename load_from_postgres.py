"""
Load OHLC data from PostgreSQL database for visual trading system.

This script connects to your existing PostgreSQL database and loads
historical OHLC data for use with the visual trading system.

Requires: psycopg2 or psycopg2-binary
Install with: pip install psycopg2-binary
"""

import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import config


def load_ohlc_from_postgres(
    ticker,
    start_date=None,
    end_date=None,
    db_config=None
):
    """
    Load OHLC data from PostgreSQL database.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date (YYYY-MM-DD) or datetime object
        end_date: End date (YYYY-MM-DD) or datetime object
        db_config: Database configuration dict with keys:
                   host, database, user, password
                   If None, uses config.DB_CONFIG

    Returns:
        DataFrame with columns: close, volume, indexed by date
    """

    # Use default database config if not provided
    if db_config is None:
        db_config = config.DB_CONFIG

    # Default date range if not specified
    if end_date is None:
        end_date = datetime.now().date()
    if start_date is None:
        start_date = end_date - timedelta(days=365)  # 1 year default

    # Convert to string format if datetime objects
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')

    print(f"Loading {ticker} data from {start_date} to {end_date}")

    # Connect to database
    try:
        conn = psycopg2.connect(
            host=db_config['host'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config.get('password')
        )

        # Query OHLC data
        # Adjust table/column names based on your schema
        query = """
            SELECT
                date,
                close,
                volume
            FROM sentientedge.ohlcv
            WHERE ticker = %s
              AND date >= %s
              AND date <= %s
            ORDER BY date ASC
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(ticker, start_date, end_date)
        )

        conn.close()

        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        print(f"✅ Loaded {len(df)} days of data")

        if len(df) == 0:
            print("⚠️  Warning: No data found for specified parameters")

        return df

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def load_multiple_tickers(
    tickers,
    start_date=None,
    end_date=None,
    db_config=None
):
    """
    Load OHLC data for multiple tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        db_config: Database configuration

    Returns:
        Dict mapping ticker -> DataFrame
    """
    data = {}

    for ticker in tickers:
        try:
            df = load_ohlc_from_postgres(ticker, start_date, end_date, db_config)
            data[ticker] = df
        except Exception as e:
            print(f"⚠️  Failed to load {ticker}: {e}")
            continue

    return data


def example_usage():
    """Example usage of data loading functions."""

    print("=" * 80)
    print("LOAD DATA FROM POSTGRESQL")
    print("=" * 80)

    # Example 1: Load single ticker
    print("\nExample 1: Load single ticker")
    print("-" * 80)

    try:
        df = load_ohlc_from_postgres(
            ticker='AAPL',
            start_date='2023-01-01',
            end_date='2024-12-31'
        )

        print(f"\nData summary:")
        print(df.describe())

    except Exception as e:
        print(f"Could not load data: {e}")
        print("\nMake sure to:")
        print("1. Install psycopg2: pip install psycopg2-binary")
        print("2. Configure database credentials in config.py")
        print("3. Ensure database is accessible")

    # Example 2: Load multiple tickers
    print("\n\nExample 2: Load multiple tickers")
    print("-" * 80)

    tickers = ['AAPL', 'MSFT', 'GOOGL']

    try:
        data_dict = load_multiple_tickers(
            tickers,
            start_date='2023-01-01',
            end_date='2024-12-31'
        )

        print(f"\nLoaded {len(data_dict)} tickers:")
        for ticker, df in data_dict.items():
            print(f"  {ticker}: {len(df)} days")

    except Exception as e:
        print(f"Could not load data: {e}")

    # Example 3: Integration with visual trading system
    print("\n\nExample 3: Use with visual trading system")
    print("-" * 80)
    print("""
from dataset import StockImageDataset
from load_from_postgres import load_ohlc_from_postgres

# Load data
price_data = load_ohlc_from_postgres('AAPL', start_date='2023-01-01')

# Create image dataset
dataset = StockImageDataset(price_data, lookback=60, horizon=5)

# Train model...
""")


if __name__ == '__main__':
    example_usage()
