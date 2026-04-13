"""
PostgreSQL data storage layer for Visual Trading System.

Alternative to SQLite for production use. Connects to existing sentientedge database.
"""

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from typing import Optional, List
import os
import config


class PostgresStore:
    """PostgreSQL storage for OHLCV data."""

    def __init__(
        self,
        host: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize PostgreSQL store.

        Args:
            host: Database host (defaults to config.DB_CONFIG or env var)
            database: Database name (defaults to config.DB_CONFIG or env var)
            user: Database user (defaults to config.DB_CONFIG or env var)
            password: Database password (defaults to env var DB_PASSWORD)
        """
        self.host = host or config.DB_CONFIG.get('host') or os.getenv('DB_HOST', 'localhost')
        self.database = database or config.DB_CONFIG.get('database') or os.getenv('DB_NAME', 'sentientedge')
        self.user = user or config.DB_CONFIG.get('user') or os.getenv('DB_USER', 'postgres')
        self.password = password or os.getenv('DB_PASSWORD')

        if not self.password:
            raise ValueError("Database password required. Set DB_PASSWORD environment variable.")

        # Test connection
        self._test_connection()

        # Initialize schema
        self._init_schema()

    def _test_connection(self):
        """Test database connection."""
        try:
            conn = self._connect()
            conn.close()
            print(f"✅ Connected to PostgreSQL: {self.database}@{self.host}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def _connect(self):
        """Create database connection."""
        return psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )

    def _init_schema(self):
        """Create tables and indexes if they don't exist."""
        conn = self._connect()
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'daily_bars_vts'
            );
        """)
        exists = cursor.fetchone()[0]

        if not exists:
            # Create table
            cursor.execute("""
                CREATE TABLE daily_bars_vts (
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    timestamp TIMESTAMPTZ,
                    open NUMERIC NOT NULL,
                    high NUMERIC NOT NULL,
                    low NUMERIC NOT NULL,
                    close NUMERIC NOT NULL,
                    volume BIGINT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (ticker, date)
                );
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX idx_vts_ticker_date ON daily_bars_vts(ticker, date);
            """)
            cursor.execute("""
                CREATE INDEX idx_vts_date ON daily_bars_vts(date);
            """)

            conn.commit()
            print("✅ Created daily_bars_vts table")
        else:
            print("✅ Using existing daily_bars_vts table")

        cursor.close()
        conn.close()

    def save_bars(self, df: pd.DataFrame, replace: bool = False):
        """
        Save OHLCV bars to database.

        Args:
            df: DataFrame with columns: ticker, date, timestamp, open, high, low, close, volume
            replace: If True, delete existing data for tickers first
        """
        if df.empty:
            print("⚠️ Empty DataFrame, nothing to save")
            return

        # Ensure required columns
        required = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Prepare data
        df_save = df.copy()
        df_save['date'] = pd.to_datetime(df_save['date']).dt.date

        conn = self._connect()
        cursor = conn.cursor()

        try:
            if replace:
                # Delete existing data for these tickers
                tickers = tuple(df_save['ticker'].unique())
                if len(tickers) == 1:
                    cursor.execute(
                        "DELETE FROM daily_bars_vts WHERE ticker = %s",
                        (tickers[0],)
                    )
                else:
                    cursor.execute(
                        "DELETE FROM daily_bars_vts WHERE ticker IN %s",
                        (tickers,)
                    )

            # Prepare values for insert
            columns = ['ticker', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            insert_cols = [col for col in columns if col in df_save.columns]

            values = []
            for _, row in df_save.iterrows():
                value_tuple = tuple(row[col] if col in df_save.columns else None for col in insert_cols)
                values.append(value_tuple)

            # Insert with ON CONFLICT DO NOTHING (ignore duplicates)
            cols_str = ', '.join(insert_cols)
            placeholders = ', '.join(['%s'] * len(insert_cols))

            execute_values(
                cursor,
                f"""
                INSERT INTO daily_bars_vts ({cols_str})
                VALUES %s
                ON CONFLICT (ticker, date) DO NOTHING
                """,
                values,
                template=f"({placeholders})"
            )

            conn.commit()
            print(f"✅ Saved {len(df_save)} bars to PostgreSQL")

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

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
        query = "SELECT * FROM daily_bars_vts WHERE ticker = %s"
        params = [ticker]

        if start_date:
            query += " AND date >= %s"
            params.append(start_date)

        if end_date:
            query += " AND date <= %s"
            params.append(end_date)

        query += " ORDER BY date ASC"

        conn = self._connect()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

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
        query = "SELECT * FROM daily_bars_vts WHERE ticker IN %s"
        params = [tuple(tickers)]

        if start_date:
            query += " AND date >= %s"
            params.append(start_date)

        if end_date:
            query += " AND date <= %s"
            params.append(end_date)

        query += " ORDER BY ticker, date ASC"

        conn = self._connect()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_available_tickers(self) -> List[str]:
        """Get list of tickers in database."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT ticker FROM daily_bars_vts ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return tickers

    def get_date_range(self, ticker: str) -> tuple:
        """
        Get available date range for a ticker.

        Returns:
            Tuple of (start_date, end_date)
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT MIN(date), MAX(date) FROM daily_bars_vts WHERE ticker = %s",
            (ticker,)
        )
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result[0] is None:
            return None, None

        return result[0], result[1]

    def get_stats(self) -> dict:
        """Get database statistics."""
        conn = self._connect()
        cursor = conn.cursor()

        # Total bars
        cursor.execute("SELECT COUNT(*) FROM daily_bars_vts")
        total_bars = cursor.fetchone()[0]

        # Number of tickers
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM daily_bars_vts")
        num_tickers = cursor.fetchone()[0]

        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM daily_bars_vts")
        date_range = cursor.fetchone()

        cursor.close()
        conn.close()

        return {
            'total_bars': total_bars,
            'num_tickers': num_tickers,
            'start_date': date_range[0],
            'end_date': date_range[1],
            'database': f"{self.database}@{self.host}",
            'table': 'daily_bars_vts'
        }


def main():
    """Test PostgreSQL store."""
    print("=" * 80)
    print("PostgreSQL Store Test")
    print("=" * 80)

    try:
        # Initialize store
        store = PostgresStore()

        # Get stats
        stats = store.get_stats()
        print("\nDatabase Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Get available tickers
        tickers = store.get_available_tickers()
        print(f"\nAvailable tickers: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  - PostgreSQL is running")
        print("  - Database 'sentientedge' exists")
        print("  - DB_PASSWORD environment variable is set")


if __name__ == '__main__':
    main()
