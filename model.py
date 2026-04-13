"""
CNN Models for Stock Image-based Prediction

Implements convolutional neural networks for regression and classification
of stock returns from image-transformed time series.
"""

import torch
import torch.nn as nn


class TradingCNN(nn.Module):
    """
    Simple CNN for stock return prediction from multi-channel images.

    Architecture:
    - 3 convolutional blocks with progressive channel expansion (32->64->128)
    - Each block: Conv2d -> ReLU -> MaxPool2d/AdaptiveAvgPool2d
    - Final linear regression head
    """

    def __init__(self, num_channels=2, image_size=60):
        """
        Initialize the CNN model.

        Args:
            num_channels: Number of input image channels (default: 2 for price + volume)
            image_size: Size of input images (default: 60x60)
        """
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 32 filters
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 60x60 -> 30x30

            # Block 2: 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 30x30 -> 15x15

            # Block 3: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Any size -> 1x1 (allows variable input sizes)
        )

        # Regression head
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, num_channels, image_size, image_size)

        Returns:
            Predicted returns of shape (batch_size, 1)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128)
        return self.regressor(x)


class TradingCNNClassifier(nn.Module):
    """
    CNN for classification-based trading (buy/sell/hold).

    Similar architecture to TradingCNN but with softmax output for
    multi-class classification.
    """

    def __init__(self, num_channels=2, image_size=60, num_classes=3):
        """
        Initialize the classifier.

        Args:
            num_channels: Number of input image channels
            image_size: Size of input images
            num_classes: Number of output classes (default: 3 for buy/sell/hold)
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Classification head
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_channels, image_size, image_size)

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DeepTradingCNN(nn.Module):
    """
    Deeper CNN with batch normalization and dropout for improved performance.

    More sophisticated architecture for better feature extraction from
    complex market patterns.
    """

    def __init__(self, num_channels=2, image_size=60, dropout=0.5):
        """
        Initialize the deep CNN.

        Args:
            num_channels: Number of input image channels
            image_size: Size of input images
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


class DeepTradingCNNClassifier(nn.Module):
    """
    Deep CNN for binary classification with technical indicators.

    Similar to DeepTradingCNN but with classification head instead of regression.
    """

    def __init__(self, num_channels=2, image_size=60, num_classes=2, dropout=0.5):
        """
        Initialize the deep classifier.

        Args:
            num_channels: Number of input image channels
            image_size: Size of input images
            num_classes: Number of output classes (default: 2 for binary)
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
