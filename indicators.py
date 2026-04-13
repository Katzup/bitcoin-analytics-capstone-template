"""
Technical indicator calculations for Visual Trading System.

Adds RSI, MACD, Bollinger Bands, and other indicators to price data.
"""

import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices
        period: RSI period (default: 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of closing prices
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: ATR period (default: 14)

    Returns:
        Series of ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    Args:
        close: Series of closing prices
        volume: Series of volume

    Returns:
        Series of OBV values
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame with OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with additional indicator columns
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)

    # MACD
    macd_line, signal_line, histogram = calculate_macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = histogram

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower

    # Bollinger Band Width (normalized)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Price position in Bollinger Bands
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR (if high/low available)
    if 'high' in df.columns and 'low' in df.columns:
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as % of price

    # OBV
    df['obv'] = calculate_obv(df['close'], df['volume'])

    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Price momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)

    # Volume momentum
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Fill NaN values with forward fill, then backward fill
    df = df.ffill().bfill()

    return df


def normalize_indicators(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    """
    Normalize indicator values to [0, 1] range for GAF transformation.

    Args:
        df: DataFrame with indicator columns
        indicators: List of indicator column names to normalize

    Returns:
        DataFrame with normalized indicators
    """
    df = df.copy()

    for indicator in indicators:
        if indicator in df.columns:
            # Min-max normalization
            min_val = df[indicator].min()
            max_val = df[indicator].max()

            if max_val > min_val:
                df[f'{indicator}_norm'] = (df[indicator] - min_val) / (max_val - min_val)
            else:
                df[f'{indicator}_norm'] = 0.5  # If all values are same, set to middle

    return df


def main():
    """Test indicator calculations."""
    print("Testing technical indicators...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(200) * 2),
        'high': 100 + np.cumsum(np.random.randn(200) * 2) + 2,
        'low': 100 + np.cumsum(np.random.randn(200) * 2) - 2,
        'close': 100 + np.cumsum(np.random.randn(200) * 2),
        'volume': np.random.randint(1000000, 10000000, 200)
    })

    # Add indicators
    df_with_indicators = add_technical_indicators(df)

    print(f"\nOriginal columns: {list(df.columns)}")
    print(f"With indicators: {list(df_with_indicators.columns)}")
    print(f"\nAdded {len(df_with_indicators.columns) - len(df.columns)} indicators")

    print("\nSample data:")
    print(df_with_indicators[['close', 'rsi', 'macd', 'bb_upper', 'bb_lower']].tail())


if __name__ == '__main__':
    main()
