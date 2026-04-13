"""
Configuration parameters for the visual trading system.

Contains all hyperparameters, paths, and settings for data processing,
model training, and backtesting.
"""

import torch

# ============================================================================
# Data Parameters
# ============================================================================

# Image generation
LOOKBACK_WINDOW = 60        # Number of days to convert to image
PREDICTION_HORIZON = 5      # Days ahead to predict
IMAGE_SIZE = 60             # Size of GAF images (60x60)
GAF_METHOD = 'summation'    # 'summation' or 'difference'

# Data split
TRAIN_SPLIT = 0.7           # 70% for training
VAL_SPLIT = 0.15            # 15% for validation
TEST_SPLIT = 0.15           # 15% for testing

# ============================================================================
# Model Parameters
# ============================================================================

# Architecture
NUM_CHANNELS = 2            # Number of image channels (price + volume)
MODEL_TYPE = 'simple'       # 'simple', 'deep', or 'classifier'
DROPOUT = 0.5               # Dropout probability for DeepTradingCNN

# Classification (if using TradingCNNClassifier)
NUM_CLASSES = 2             # Binary: Up/Down for next H days
PREDICTION_THRESHOLD = 0.5  # Probability threshold for long signal (default: 0.5)
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to take position (increased from 0.6)

# ============================================================================
# Training Parameters
# ============================================================================

# Optimization
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Loss function
LOSS_TYPE = 'mse'           # 'mse' for regression, 'crossentropy' for classification

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Backtesting Parameters
# ============================================================================

# Walk-forward analysis
WALK_FORWARD_WINDOW = 252   # Days in training window (1 year)
WALK_FORWARD_STEP = 21      # Days to step forward (1 month)

# Trading rules
POSITION_SIZE = 0.1         # 10% of portfolio per position
TRANSACTION_COST = 0.001    # 0.1% per trade
MAX_POSITIONS = 10          # Maximum concurrent positions

# Risk management
STOP_LOSS = -0.05           # 5% stop loss
TAKE_PROFIT = 0.10          # 10% take profit
MAX_HOLDING_PERIOD = 20     # Maximum days to hold position

# ============================================================================
# Paths
# ============================================================================

DATA_DIR = './data'
MODEL_DIR = './models'
RESULTS_DIR = './results'
CHECKPOINT_DIR = './checkpoints'

# ============================================================================
# Logging
# ============================================================================

LOG_INTERVAL = 10           # Log every N batches during training
SAVE_INTERVAL = 5           # Save checkpoint every N epochs

# ============================================================================
# Data Sources (for future integration)
# ============================================================================

# PostgreSQL connection (if using existing database)
DB_CONFIG = {
    'host': 'localhost',
    'database': 'sentientedge',
    'user': 'postgres',
    'password': None  # Set via environment variable
}

# API keys (set via environment variables)
POLYGON_API_KEY = None
MASSIVE_API_KEY = None

# ============================================================================
# Technical Indicators (for MultiChannelStockDataset)
# ============================================================================

INDICATORS = [
    'rsi',              # Relative Strength Index
    'macd',             # MACD
    'bb_upper',         # Bollinger Band Upper
    'bb_lower',         # Bollinger Band Lower
]

# Indicator parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
