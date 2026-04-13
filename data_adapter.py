"""
Data Adapter for Multi-Timeframe BTC Data
Handles fetching and resampling with leakage safety
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Literal
from config_multi_timeframe import BarConfig
import yfinance as yf


class DataAdapter:
    """
    Fetch and normalize OHLCV data across multiple timeframes.

    Returns canonical schema: [date, open, high, low, close, volume]
    Ensures proper 24/7 BTC handling and resampling safety.
    """

    def __init__(self, ticker: str = "BTC-USD", source: str = "yfinance"):
        """
        Initialize data adapter.

        Args:
            ticker: Asset ticker (default: BTC-USD for yfinance)
            source: Data source ("yfinance", "binance", "polygon")
                   Note: yfinance only supports daily/hourly, not minute bars
        """
        self.ticker = ticker
        self.source = source

    def fetch_bars(
        self,
        bar_config: BarConfig,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for specified timeframe.

        Args:
            bar_config: BarConfig specifying timeframe
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None
            days_back: If start_date None, fetch this many days back

        Returns:
            DataFrame with columns: [date, open, high, low, close, volume]
            Index: DatetimeIndex (UTC-aligned for BTC 24/7)

        Raises:
            ValueError: If timeframe not supported by source
        """
        # Determine date range
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None and days_back is not None:
            start = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days_back)
            start_date = start.strftime("%Y-%m-%d")

        # Fetch based on source
        if self.source == "yfinance":
            return self._fetch_yfinance(bar_config, start_date, end_date)
        elif self.source == "binance":
            return self._fetch_binance(bar_config, start_date, end_date)
        elif self.source == "polygon":
            return self._fetch_polygon(bar_config, start_date, end_date)
        else:
            raise ValueError(f"Unsupported source: {self.source}")

    def _fetch_yfinance(
        self,
        bar_config: BarConfig,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch from yfinance.

        Limitations:
        - Only supports 1h, 1d (no 4h, no 1m/5m)
        - Intraday limited to ~730 days history
        """
        # Map bar_config to yfinance interval
        if bar_config.timespan == "day" and bar_config.multiplier == 1:
            interval = "1d"
        elif bar_config.timespan == "hour" and bar_config.multiplier == 1:
            interval = "1h"
        elif bar_config.timespan == "hour" and bar_config.multiplier == 4:
            # yfinance doesn't have native 4h - we'll resample from 1h
            print("⚠️  yfinance doesn't support 4h bars - fetching 1h and resampling")
            interval = "1h"
        else:
            raise ValueError(
                f"yfinance doesn't support {bar_config.multiplier}{bar_config.timespan} bars. "
                "Use 1h or 1d, or switch to binance/polygon source."
            )

        # Fetch data
        ticker = yf.Ticker(self.ticker)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {self.ticker} from {start_date} to {end_date}")

        # Normalize to canonical schema
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Datetime': 'date',  # For intraday
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Keep only canonical columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        # Ensure datetime index (UTC for BTC 24/7)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.set_index('date').sort_index()

        # If we fetched 1h for 4h request, resample now
        if bar_config.timespan == "hour" and bar_config.multiplier == 4:
            df = self._resample_ohlcv(df, from_bars=1, to_bars=4, timespan="hour")

        return df

    def _fetch_binance(
        self,
        bar_config: BarConfig,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch from Binance klines API.

        Binance supports: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        Deep history available (years of 1m data).
        Free public API (no key needed).
        """
        import requests
        from datetime import datetime

        # Map bar_config to Binance interval
        interval_map = {
            ("hour", 1): "1h",
            ("hour", 4): "4h",
            ("day", 1): "1d",
        }

        # Also support minute bars (for future use)
        if bar_config.timespan == "minute":
            if bar_config.multiplier == 1:
                interval = "1m"
            elif bar_config.multiplier == 5:
                interval = "5m"
            elif bar_config.multiplier == 15:
                interval = "15m"
            else:
                raise ValueError(f"Unsupported minute interval: {bar_config.multiplier}m")
        else:
            key = (bar_config.timespan, bar_config.multiplier)
            if key not in interval_map:
                raise ValueError(
                    f"Unsupported Binance interval: {bar_config.multiplier}{bar_config.timespan}. "
                    f"Supported: 1m, 5m, 15m, 1h, 4h, 1d"
                )
            interval = interval_map[key]

        # Convert dates to milliseconds
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        # Binance API endpoint
        url = "https://api.binance.com/api/v3/klines"

        # Fetch data in chunks (Binance limit: 1000 bars per request)
        all_bars = []
        current_start = start_ms
        limit = 1000

        print(f"Fetching {interval} bars from Binance (BTCUSDT)...")

        while current_start < end_ms:
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': limit
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            bars = response.json()

            if not bars:
                break

            all_bars.extend(bars)

            # Update start time for next batch (last bar's close time + 1ms)
            current_start = bars[-1][6] + 1

            # Progress indicator
            if len(all_bars) % 5000 == 0:
                print(f"  Fetched {len(all_bars)} bars...")

        print(f"  Total: {len(all_bars)} bars")

        # Convert to DataFrame
        # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(all_bars, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Rename and select canonical columns
        df = df.rename(columns={'open_time': 'date'})
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        # Set index
        df = df.set_index('date').sort_index()

        return df

    def _fetch_polygon(
        self,
        bar_config: BarConfig,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch from Polygon (Massive) aggregates.

        Placeholder for future implementation.
        Polygon supports minute aggregates with configurable multiplier.
        """
        raise NotImplementedError(
            "Polygon adapter not yet implemented. "
            "Will support flexible minute/hour/day aggregates."
        )

    def _resample_ohlcv(
        self,
        df: pd.DataFrame,
        from_bars: int,
        to_bars: int,
        timespan: Literal["hour", "day"]
    ) -> pd.DataFrame:
        """
        Resample OHLCV upward (e.g., 1h → 4h, 4h → 1d).

        CRITICAL: Only uses closed bars to prevent data leakage.

        Args:
            df: Input DataFrame with OHLCV (datetime index)
            from_bars: Source bar size (e.g., 1 for 1h)
            to_bars: Target bar size (e.g., 4 for 4h)
            timespan: "hour" or "day"

        Returns:
            Resampled DataFrame with OHLCV
        """
        if to_bars <= from_bars:
            raise ValueError(f"Can only resample upward: {to_bars} must be > {from_bars}")

        # Determine resampling rule
        if timespan == "hour":
            rule = f"{to_bars}H"
        elif timespan == "day":
            rule = f"{to_bars}D"
        else:
            raise ValueError(f"Unknown timespan: {timespan}")

        # Resample with proper OHLCV aggregation
        # IMPORTANT: label='left', closed='left' ensures we only use past data
        resampled = df.resample(rule, label='left', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Drop any incomplete bars (NaN from resampling)
        resampled = resampled.dropna()

        return resampled


class Resampler:
    """
    Utility for safe upward resampling of OHLCV data.

    Ensures no data leakage by only using closed bars.
    """

    @staticmethod
    def resample(
        df: pd.DataFrame,
        target_config: BarConfig,
        source_config: Optional[BarConfig] = None
    ) -> pd.DataFrame:
        """
        Resample OHLCV to target timeframe.

        Args:
            df: Input OHLCV DataFrame (datetime index)
            target_config: Target bar configuration
            source_config: Source bar configuration (inferred if None)

        Returns:
            Resampled DataFrame

        Example:
            >>> # Resample 1h → 4h
            >>> from config_multi_timeframe import BTC_1H, BTC_4H
            >>> df_1h = adapter.fetch_bars(BTC_1H.bars, days_back=90)
            >>> df_4h = Resampler.resample(df_1h, BTC_4H.bars, BTC_1H.bars)
        """
        # If source not specified, assume finest granularity (1h)
        if source_config is None:
            source_config = BarConfig(timespan="hour", multiplier=1, bars_per_day=24)

        # Validate upward resampling
        if target_config.bars_per_day > source_config.bars_per_day:
            raise ValueError(
                f"Can only resample upward: target ({target_config.bars_per_day} bars/day) "
                f"must be <= source ({source_config.bars_per_day} bars/day)"
            )

        # If already at target granularity, return as-is
        if target_config.bars_per_day == source_config.bars_per_day:
            return df.copy()

        # Calculate resampling rule
        if target_config.timespan == "day":
            rule = f"{target_config.multiplier}D"
        else:  # hour
            rule = f"{target_config.multiplier}H"

        # Resample with leakage safety
        resampled = df.resample(rule, label='left', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Drop incomplete bars
        resampled = resampled.dropna()

        return resampled


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    from config_multi_timeframe import BTC_1D, BTC_4H, BTC_1H

    # Example 1: Fetch daily bars (yfinance)
    print("=" * 60)
    print("Example 1: Fetch BTC daily bars (yfinance)")
    print("=" * 60)

    adapter = DataAdapter(ticker="BTC-USD", source="yfinance")

    try:
        df_1d = adapter.fetch_bars(BTC_1D.bars, days_back=365)
        print(f"Fetched {len(df_1d)} daily bars")
        print(f"Date range: {df_1d.index.min()} to {df_1d.index.max()}")
        print(f"\nFirst 3 bars:")
        print(df_1d.head(3))
        print(f"\nLast 3 bars:")
        print(df_1d.tail(3))
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Example 2: Fetch BTC hourly bars (yfinance)")
    print("=" * 60)

    try:
        df_1h = adapter.fetch_bars(BTC_1H.bars, days_back=90)
        print(f"Fetched {len(df_1h)} hourly bars")
        print(f"Date range: {df_1h.index.min()} to {df_1h.index.max()}")
        print(f"Expected bars (90d × 24h): {90 * 24} = 2160")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Example 3: Resample 1h → 4h")
    print("=" * 60)

    try:
        if 'df_1h' in locals() and not df_1h.empty:
            df_4h = Resampler.resample(df_1h, BTC_4H.bars, BTC_1H.bars)
            print(f"Resampled {len(df_1h)} 1h bars → {len(df_4h)} 4h bars")
            print(f"Ratio: {len(df_1h) / len(df_4h):.2f} (should be ~4.0)")
            print(f"\nFirst 3 4h bars:")
            print(df_4h.head(3))
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Example 4: Cost model application")
    print("=" * 60)

    btc_price = 90000.0
    print(f"BTC Spot Price: ${btc_price:,.2f}\n")

    for name, preset in [("1d", BTC_1D), ("4h", BTC_4H), ("1h", BTC_1H)]:
        buy_price = preset.costs.apply_to_price(btc_price, side='buy')
        cost_dollars = buy_price - btc_price
        print(f"{name:3s}: ${buy_price:,.2f} (+${cost_dollars:6.2f}, {preset.costs.total_bps:5.1f} bps)")
