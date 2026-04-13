"""
SQLite data storage layer for Visual Trading System.

Handles storing and retrieving OHLCV data from local SQLite database.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List
import config


class DataStore:
    """SQLite storage for OHLCV data."""

    def __init__(self, db_path: str = './data/trading_data.db'):
        """
        Initialize data store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Create data directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # OHLCV data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_bars (
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    timestamp DATETIME,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    PRIMARY KEY (ticker, date)
                )
            """)

            # Index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_date
                ON daily_bars(ticker, date)
            """)

            conn.commit()

    def save_bars(self, df: pd.DataFrame, replace: bool = False):
        """
        Save OHLCV bars to database.

        Args:
            df: DataFrame with columns: ticker, date, timestamp, open, high, low, close, volume
            replace: If True, replace existing data; if False, append
        """
        if df.empty:
            print("⚠️ Empty DataFrame, nothing to save")
            return

        # Ensure required columns exist
        required = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Select only required columns (timestamp is optional)
        columns = ['ticker', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        save_cols = [col for col in columns if col in df.columns]
        df_save = df[save_cols].copy()

        # Convert date to string format
        df_save['date'] = pd.to_datetime(df_save['date']).dt.strftime('%Y-%m-%d')

        with sqlite3.connect(self.db_path) as conn:
            if replace:
                # Delete existing data for these tickers
                tickers = df_save['ticker'].unique()
                placeholders = ','.join(['?' for _ in tickers])
                conn.execute(
                    f"DELETE FROM daily_bars WHERE ticker IN ({placeholders})",
                    list(tickers)
                )

            # Insert data (ignore duplicates)
            df_save.to_sql('daily_bars', conn, if_exists='append', index=False)

        print(f"✅ Saved {len(df_save)} bars to database")

    def get_bars(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV bars from database.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)

        Returns:
            DataFrame with OHLCV data
        """
        query = "SELECT * FROM daily_bars WHERE ticker = ?"
        params = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)

        Returns:
            Combined DataFrame for all tickers
        """
        placeholders = ','.join(['?' for _ in tickers])
        query = f"SELECT * FROM daily_bars WHERE ticker IN ({placeholders})"
        params = list(tickers)

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY ticker, date ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_available_tickers(self) -> List[str]:
        """Get list of tickers in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM daily_bars ORDER BY ticker")
            tickers = [row[0] for row in cursor.fetchall()]

        return tickers

    def get_date_range(self, ticker: str) -> tuple:
        """
        Get available date range for a ticker.

        Returns:
            Tuple of (start_date, end_date)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MIN(date), MAX(date) FROM daily_bars WHERE ticker = ?",
                (ticker,)
            )
            result = cursor.fetchone()

        if result[0] is None:
            return None, None

        return result[0], result[1]

    def get_stats(self) -> dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total bars
            cursor.execute("SELECT COUNT(*) FROM daily_bars")
            total_bars = cursor.fetchone()[0]

            # Number of tickers
            cursor.execute("SELECT COUNT(DISTINCT ticker) FROM daily_bars")
            num_tickers = cursor.fetchone()[0]

            # Date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM daily_bars")
            date_range = cursor.fetchone()

        return {
            'total_bars': total_bars,
            'num_tickers': num_tickers,
            'start_date': date_range[0],
            'end_date': date_range[1],
            'db_path': self.db_path
        }


def main():
    """Test data store."""
    print("=" * 80)
    print("Data Store Test")
    print("=" * 80)

    # Initialize store
    store = DataStore()

    # Get stats
    stats = store.get_stats()
    print("\nDatabase Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get available tickers
    tickers = store.get_available_tickers()
    print(f"\nAvailable tickers: {tickers}")

    if tickers:
        # Test retrieving data
        ticker = tickers[0]
        print(f"\nRetrieving data for {ticker}...")
        df = store.get_bars(ticker)
        print(f"Shape: {df.shape}")
        print(df.head())


if __name__ == '__main__':
    main()
