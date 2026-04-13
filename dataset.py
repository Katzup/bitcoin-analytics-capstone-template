"""
Stock Image Dataset for Visual Time Series Forecasting

Converts OHLC time series data into multi-channel images using Gramian Angular Fields (GAF).
Each sample is a multi-channel image where channels represent different market aspects.
"""

import numpy as np
from pyts.image import GramianAngularField
import torch
from torch.utils.data import Dataset


class StockImageDataset(Dataset):
    """
    Dataset for converting stock time series into images for CNN-based prediction.

    Uses Gramian Angular Fields to transform 1D time series into 2D images that
    preserve temporal correlation structure.
    """

    def __init__(self, price_data, lookback=60, horizon=5):
        """
        Initialize the dataset with price data and parameters.

        Args:
            price_data: DataFrame with 'close', 'volume' columns
            lookback: Number of time steps to convert to image (window size)
            horizon: Number of days ahead to predict (forward return period)
        """
        self.lookback = lookback
        self.horizon = horizon
        self.gaf = GramianAngularField(image_size=lookback, method='summation')
        self.images, self.labels = self._create_images(price_data)

    def _create_images(self, data):
        """
        Create multi-channel images and labels from time series data.

        Each window of lookback days is transformed into a 2-channel image:
        - Channel 1: Price GAF (temporal correlation of close prices)
        - Channel 2: Volume GAF (temporal correlation of volume)

        Returns:
            images: Array of shape (num_samples, num_channels, image_size, image_size)
            labels: Array of forward returns (percentage change)
        """
        images, labels = [], []

        for i in range(len(data) - self.lookback - self.horizon):
            # Extract window of data
            window = data[i:i+self.lookback]

            # Channel 1: Price GAF
            price_img = self.gaf.transform(window['close'].values.reshape(1, -1))

            # Channel 2: Volume GAF
            vol_img = self.gaf.transform(window['volume'].values.reshape(1, -1))

            # Stack channels (shape: num_channels x image_size x image_size)
            img = np.stack([price_img[0], vol_img[0]], axis=0)
            images.append(img)

            # Label: forward return (percentage)
            future_price = data.iloc[i+self.lookback+self.horizon]['close']
            current_price = data.iloc[i+self.lookback]['close']
            labels.append((future_price - current_price) / current_price)

        return np.array(images), np.array(labels)

    def __getitem__(self, idx):
        """
        Return single training sample as PyTorch tensors.

        Args:
            idx: Index of the sample

        Returns:
            tuple: (image_tensor, label_tensor)
        """
        return torch.FloatTensor(self.images[idx]), torch.FloatTensor([self.labels[idx]])

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.images)


class MultiChannelStockDataset(Dataset):
    """
    Extended dataset with additional technical indicator channels.

    Supports adding RSI, MACD, Bollinger Bands as additional image channels.
    """

    def __init__(self, price_data, lookback=60, horizon=5, indicators=None):
        """
        Initialize with price data and optional technical indicators.

        Args:
            price_data: DataFrame with OHLC and volume columns
            lookback: Number of time steps to convert to image
            horizon: Number of days ahead to predict
            indicators: List of indicator column names to include as channels
        """
        self.lookback = lookback
        self.horizon = horizon
        self.indicators = indicators or []
        self.gaf = GramianAngularField(image_size=lookback, method='summation')
        self.images, self.labels = self._create_images(price_data)

    def _create_images(self, data):
        """Create multi-channel images with technical indicators."""
        images, labels = [], []

        for i in range(len(data) - self.lookback - self.horizon):
            window = data[i:i+self.lookback]

            # Base channels
            channels = []
            channels.append(self.gaf.transform(window['close'].values.reshape(1, -1))[0])
            channels.append(self.gaf.transform(window['volume'].values.reshape(1, -1))[0])

            # Add indicator channels if available
            for indicator in self.indicators:
                if indicator in window.columns:
                    ind_img = self.gaf.transform(window[indicator].values.reshape(1, -1))
                    channels.append(ind_img[0])

            img = np.stack(channels, axis=0)
            images.append(img)

            # Label
            future_price = data.iloc[i+self.lookback+self.horizon]['close']
            current_price = data.iloc[i+self.lookback]['close']
            labels.append((future_price - current_price) / current_price)

        return np.array(images), np.array(labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.images[idx]), torch.FloatTensor([self.labels[idx]])

    def __len__(self):
        return len(self.images)


class BinaryStockImageDataset(StockImageDataset):
    """
    Binary classification dataset for directional prediction (up/down).

    Instead of predicting returns, predicts whether price will be higher
    or lower after the horizon period.
    """

    def __init__(self, price_data, lookback=60, horizon=5):
        """
        Initialize binary classification dataset.

        Args:
            price_data: DataFrame with 'close', 'volume' columns
            lookback: Number of time steps to convert to image (window size)
            horizon: Number of days ahead to predict direction
        """
        # Call parent constructor but override label creation
        self.lookback = lookback
        self.horizon = horizon
        self.gaf = GramianAngularField(image_size=lookback, method='summation')
        self.images, self.labels = self._create_binary_images(price_data)

    def _create_binary_images(self, data):
        """
        Create images with binary direction labels.

        Returns:
            images: Array of shape (num_samples, num_channels, image_size, image_size)
            labels: Array of binary labels (0=down, 1=up)
        """
        images, labels = [], []

        for i in range(len(data) - self.lookback - self.horizon):
            # Extract window of data
            window = data[i:i+self.lookback]

            # Channel 1: Price GAF
            price_img = self.gaf.transform(window['close'].values.reshape(1, -1))

            # Channel 2: Volume GAF
            vol_img = self.gaf.transform(window['volume'].values.reshape(1, -1))

            # Stack channels
            img = np.stack([price_img[0], vol_img[0]], axis=0)
            images.append(img)

            # Binary label: 1 if price goes up, 0 if down
            future_price = data.iloc[i+self.lookback+self.horizon]['close']
            current_price = data.iloc[i+self.lookback]['close']
            label = 1 if future_price > current_price else 0
            labels.append(label)

        return np.array(images), np.array(labels)

    def __getitem__(self, idx):
        """
        Return single training sample.

        Returns:
            tuple: (image_tensor, label_tensor)
                - image_tensor: shape (num_channels, image_size, image_size)
                - label_tensor: shape (1,) with binary class label
        """
        return torch.FloatTensor(self.images[idx]), torch.LongTensor([self.labels[idx]])


class EnhancedBinaryDataset(BinaryStockImageDataset):
    """
    Enhanced binary classification dataset with technical indicators.

    Creates multi-channel GAF images from:
    - Price (close)
    - Volume
    - RSI
    - MACD
    - Bollinger Band position
    """

    def __init__(self, price_data, lookback=60, horizon=5, use_indicators=True):
        """
        Initialize enhanced dataset.

        Args:
            price_data: DataFrame with OHLCV and indicator columns
            lookback: Window size for image generation
            horizon: Prediction horizon in days
            use_indicators: If True, add indicator channels (default: True)
        """
        self.lookback = lookback
        self.horizon = horizon
        self.use_indicators = use_indicators
        self.gaf = GramianAngularField(image_size=lookback, method='summation')

        # Determine number of channels
        self.channels = ['close', 'volume']
        if use_indicators:
            # Add indicator channels if available
            available_indicators = ['rsi', 'macd', 'bb_position']
            for ind in available_indicators:
                if ind in price_data.columns:
                    self.channels.append(ind)

        self.images, self.labels = self._create_enhanced_images(price_data)

    def _create_enhanced_images(self, data):
        """
        Create multi-channel images with indicators.

        Returns:
            images: Array of shape (num_samples, num_channels, image_size, image_size)
            labels: Array of binary labels
        """
        images, labels = [], []

        for i in range(len(data) - self.lookback - self.horizon):
            window = data[i:i+self.lookback]

            channels = []

            # Create GAF for each channel
            for channel_name in self.channels:
                if channel_name in window.columns:
                    channel_data = window[channel_name].values.reshape(1, -1)

                    # Normalize to [-1, 1] for GAF
                    channel_min = channel_data.min()
                    channel_max = channel_data.max()

                    if channel_max > channel_min:
                        normalized = 2 * (channel_data - channel_min) / (channel_max - channel_min) - 1
                    else:
                        normalized = np.zeros_like(channel_data)

                    # Transform to GAF
                    gaf_image = self.gaf.transform(normalized)
                    channels.append(gaf_image[0])

            # Stack channels
            img = np.stack(channels, axis=0)
            images.append(img)

            # Binary label
            future_price = data.iloc[i+self.lookback+self.horizon]['close']
            current_price = data.iloc[i+self.lookback]['close']
            label = 1 if future_price > current_price else 0
            labels.append(label)

        print(f"  Created dataset with {len(self.channels)} channels: {self.channels}")

        return np.array(images), np.array(labels)
