# Visual Trading System

Image-based stock trading system using Gramian Angular Fields (GAF) and Convolutional Neural Networks (CNNs) for time series forecasting.

## 📚 Overview

This project implements a novel approach to stock prediction by transforming 1D time series data into 2D images using Gramian Angular Fields, then applying computer vision techniques (CNNs) for pattern recognition and return prediction.

### Key Concepts

- **Gramian Angular Fields (GAF)**: Transform temporal correlation into polar coordinate space, creating images that preserve time-series structure
- **Multi-Channel Images**: Stack price, volume, and technical indicators as separate channels
- **CNN-based Prediction**: Leverage convolutional neural networks for feature extraction and return forecasting
- **Walk-Forward Backtesting**: Realistic trading simulation with proper train/test separation

## 🏗️ Project Structure

```
Visual_Trading_System/
├── dataset.py          # Data loading and image transformation
├── model.py            # CNN architectures (TradingCNN, DeepTradingCNN)
├── train.py            # Training pipeline with early stopping
├── backtest.py         # Walk-forward backtesting framework
├── config.py           # Configuration and hyperparameters
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```python
import pandas as pd
from dataset import StockImageDataset

# Load your OHLC data
price_data = pd.read_csv('your_data.csv')  # Must have 'close' and 'volume' columns

# Create dataset
dataset = StockImageDataset(
    price_data,
    lookback=60,      # 60-day windows
    horizon=5         # Predict 5 days ahead
)
```

### 3. Train Model

```python
from train import create_model, create_dataloaders, train
import config

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(dataset)

# Create model
model = create_model(model_type='simple', num_channels=2)

# Train
history = train(
    model,
    train_loader,
    val_loader,
    num_epochs=config.NUM_EPOCHS,
    device=config.DEVICE
)
```

### 4. Backtest

```python
from backtest import Backtester

# Load trained model
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Run backtest
backtester = Backtester(model, price_data, initial_capital=100000)
results = backtester.run()
backtester.print_results(results)
```

## 📊 Architecture

### StockImageDataset

Converts time series windows into multi-channel images:

- **Channel 1**: Price GAF (temporal correlation of close prices)
- **Channel 2**: Volume GAF (temporal correlation of volume)
- **Optional**: Additional channels for RSI, MACD, Bollinger Bands

### TradingCNN

Simple 3-layer CNN for regression:
- Conv2d (2→32) → ReLU → MaxPool2d
- Conv2d (32→64) → ReLU → MaxPool2d
- Conv2d (64→128) → ReLU → AdaptiveAvgPool2d
- Linear (128→1) regression head

### DeepTradingCNN

Advanced architecture with:
- Batch normalization for stable training
- Dropout for regularization
- Deeper layers for complex pattern extraction

## ⚙️ Configuration

Edit `config.py` to adjust:

```python
# Data parameters
LOOKBACK_WINDOW = 60        # Window size for image generation
PREDICTION_HORIZON = 5      # Days ahead to predict
IMAGE_SIZE = 60             # GAF image size

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Backtesting
POSITION_SIZE = 0.1         # 10% per position
STOP_LOSS = -0.05           # 5% stop loss
TAKE_PROFIT = 0.10          # 10% take profit
```

## 📈 Performance Metrics

The backtesting framework calculates:

- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Profit Factor**: Gross profits / gross losses
- **Total Return**: Overall portfolio performance

## 🔬 Research Foundation

Based on the paper: "Visual time series forecasting: an image-driven approach"

Key innovations:
- Encoding temporal dependencies in 2D space
- Leveraging pre-trained CNN architectures
- Multi-channel representations of market data

## 🛠️ Extension Ideas

1. **Multi-Timeframe**: Stack images from different timeframes (daily, hourly, 15-min)
2. **Attention Mechanisms**: Add self-attention for long-range dependencies
3. **Transfer Learning**: Use pre-trained ResNet/EfficientNet backbones
4. **Ensemble Methods**: Combine multiple model predictions
5. **Alternative Transformations**: Experiment with MTF, Recurrence Plots
6. **Multi-Asset**: Train on portfolio of stocks simultaneously

## 📝 Usage Examples

### Simple Single-Stock Training

```python
from dataset import StockImageDataset
from train import create_model, create_dataloaders, train
import pandas as pd
import config

# Load data
data = pd.read_csv('AAPL.csv')

# Create dataset
dataset = StockImageDataset(data, lookback=60, horizon=5)

# Train
train_loader, val_loader, test_loader = create_dataloaders(dataset)
model = create_model('simple')
history = train(model, train_loader, val_loader, device=config.DEVICE)
```

### Multi-Channel with Indicators

```python
from dataset import MultiChannelStockDataset

# Add technical indicators to your dataframe first
data['rsi'] = calculate_rsi(data['close'])
data['macd'] = calculate_macd(data['close'])

# Create multi-channel dataset
dataset = MultiChannelStockDataset(
    data,
    lookback=60,
    horizon=5,
    indicators=['rsi', 'macd']
)
```

## 📄 License

MIT License - Feel free to use and modify for your trading research.

## ⚠️ Execution Assumptions & Limitations

### What This System Models
- **Buy-only accumulation**: When to buy more/less aggressively
- **Signal quality**: Predictive power of image-based features
- **Allocation timing**: Optimal entry under constraints

### What It Does NOT Model
- Sell decisions or liquidation timing
- Real-time execution microstructure (bid/ask, slippage)
- Market impact of large orders

### Execution Model

| Parameter | Assumption |
|-----------|-----------|
| **Signal Timestamp** | End of day t (uses data through close t) |
| **Execution Timestamp** | Day t (same-day approximation) |
| **Execution Price** | Daily close (proxy for next-open/VWAP) |
| **Transaction Costs** | 0 bps base case; viable to 50+ bps |
| **Look-ahead Bias** | **NONE** (validated by 3 tests) |
| **Cash Constraints** | Budget-normalized (same total for all strategies) |
| **Sell Logic** | **NONE** (buy-and-hold only) |

### Validation
- **Last-row modification test**: First N-1 features unchanged (diff < 1e-6)
- **Purge test**: Truncated data produces identical features
- **Shift test**: Forward-shifted features collapse performance

See `EXECUTION_ASSUMPTIONS.md` for full sensitivity analysis.

---

## ⚠️ Disclaimer

This is a research tool for educational purposes. Past performance does not guarantee future results. Use at your own risk. Not financial advice.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional image transformation methods (MTF, Recurrence Plots)
- More sophisticated CNN architectures (Vision Transformers)
- Enhanced backtesting with slippage and market impact models
- Integration with live trading APIs
- Performance visualization tools
