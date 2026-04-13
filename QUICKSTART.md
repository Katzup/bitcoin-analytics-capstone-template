# Visual Trading System - Quick Start Guide

## P0 Pipeline - Proof of Concept

The P0 pipeline runs an end-to-end binary classification trading system on SPY.

### Prerequisites

1. **Python 3.8+** with virtual environment
2. **Polygon.io API key** (free tier works)
   - Sign up at https://polygon.io/
   - Copy your API key

### Installation

```bash
# Clone/navigate to project
cd Visual_Trading_System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Set Polygon API key
export POLYGON_API_KEY='your_api_key_here'

# On Windows:
# set POLYGON_API_KEY=your_api_key_here
```

### Run P0 Pipeline

```bash
# Default: SPY, 2 years history, CPU-only
python p0_runner.py

# Custom ticker and parameters
python p0_runner.py --ticker AAPL --days 1000 --epochs 50

# Force refresh data from API
python p0_runner.py --refresh
```

### What P0 Does

1. **Fetch Data**: Downloads SPY daily OHLCV from Polygon API
2. **Store**: Saves to local SQLite database (`./data/trading_data.db`)
3. **Transform**: Converts price/volume windows to GAF images
4. **Train**: Binary classifier predicts up/down for next 5 days
5. **Backtest**: Walk-forward backtest with realistic trading costs
6. **Results**: Saves trades and equity curve to `./results/`

### Pipeline Parameters

```python
--ticker SPY           # Stock ticker to trade
--days 730             # Days of historical data (default: 2 years)
--train-window 252     # Training window size (default: 1 year)
--test-window 21       # Test window size (default: 1 month)
--epochs 30            # Training epochs per window
--refresh              # Force re-download data
```

### Expected Output

```
Visual Trading System - P0 Pipeline
================================================================================
Ticker: SPY
History: 730 days
Train Window: 252 days
Test Window: 21 days
Epochs per Window: 30
Device: cpu
================================================================================

[Step 1/4] Checking for existing data...
  ✅ Found 730 bars in database

[Step 2/4] Validating data...
  ✅ Sufficient data for backtesting

[Step 3/4] Running walk-forward backtest...
  Training model...
  ✅ Training complete (Best F1: 0.5234)
  📈 Long 100 shares @ $450.23 (prob=0.652)
  📉 Exit @ $455.67 | PnL: +$544.00 (+1.21%) | horizon (5 days)
  ...

[Step 4/4] Saving results...
  ✅ Trades saved to ./results/SPY_trades_20260102_123045.csv
  ✅ Equity curve saved to ./results/SPY_equity_20260102_123045.csv

================================================================================
BACKTEST RESULTS
================================================================================
Final Equity:      $103,245.67
Total Return:      +3.25%
Number of Trades:  15
Win Rate:          53.33%
Avg PnL:           $216.38 (+0.48%)
Sharpe Ratio:      0.82
Max Drawdown:      -5.23%
================================================================================
```

### Configuration Files

- **`config.py`**: All hyperparameters and settings
- **`requirements.txt`**: Python dependencies
- **`.env.example`**: Environment variable template

### Key Parameters in config.py

```python
# Data
LOOKBACK_WINDOW = 60        # Days to convert to image
PREDICTION_HORIZON = 5      # Days ahead to predict
IMAGE_SIZE = 60             # GAF image size

# Model
NUM_CLASSES = 2             # Binary: up/down
NUM_CHANNELS = 2            # Price + volume

# Trading
PREDICTION_THRESHOLD = 0.5  # Probability for long signal
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to trade
TRANSACTION_COST = 0.001    # 0.1% per trade

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
```

### Troubleshooting

**"POLYGON_API_KEY not found"**
- Set the environment variable: `export POLYGON_API_KEY='your_key'`
- Or add to `.env` file (copy from `.env.example`)

**"Insufficient data"**
- Increase `--days` parameter
- Use a more liquid ticker (SPY, QQQ, AAPL)
- Check Polygon API returned data

**"Empty dataset"**
- Verify data has 'close' and 'volume' columns
- Check for missing values in data
- Ensure enough bars after lookback window

**Slow training**
- Reduce `--epochs` (try 20-30)
- Reduce `--train-window` (try 126 for 6 months)
- Use GPU if available (auto-detected)

### Next Steps (P1)

P0 validates the pipeline on single ticker. P1 expands to:
- Multiple tickers (~50 liquid stocks)
- PostgreSQL database
- More sophisticated backtesting
- Performance optimization
- Production deployment prep

### File Structure

```
Visual_Trading_System/
├── p0_runner.py              # P0 end-to-end pipeline ← START HERE
├── polygon_data.py           # Polygon API integration
├── data_store.py             # SQLite storage
├── dataset.py                # GAF image generation
├── model.py                  # CNN architectures
├── train_binary.py           # Binary classification training
├── walkforward_backtest.py   # Backtesting framework
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── QUICKSTART.md             # This file
├── README.md                 # Full documentation
├── data/                     # SQLite database (auto-created)
├── results/                  # Backtest results (auto-created)
└── checkpoints/              # Model checkpoints (auto-created)
```

### Support

- Check `README.md` for detailed documentation
- Review `config.py` for all parameters
- Inspect results CSVs for trade details
