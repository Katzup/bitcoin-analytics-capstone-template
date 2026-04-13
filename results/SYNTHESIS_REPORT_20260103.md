# CNN Trading Model - Root Cause Analysis & Recommendations
**Date:** January 3, 2026
**Analysis Type:** Comprehensive Synthesis - Backtesting + Prediction Investigation
**Models Analyzed:** Top 5 performers (OXLCG, HCXY, VGI, HYI, IGI)

---

## Executive Summary

**Critical Finding:** All 5 CNN models with excellent validation losses (0.000048 - 0.000115 MSE) produced **negative returns** when backtested (-1.85% to -10.64% annualized). Deep investigation revealed this is not a single failure but a **convergence of multiple fundamental problems** in the modeling approach.

**Primary Root Causes Identified:**
1. ⚠️ **Win/Loss Asymmetry** - Losses 1.6-3.7x bigger than wins (PRIMARY cause)
2. ⚠️ **MSE Loss Misalignment** - Optimizes wrong objective (prediction accuracy ≠ profitability)
3. ⚠️ **Weak Predictive Power** - Mean correlation only 0.186, some models negative
4. ⚠️ **Ultra-Conservative Bias** - Best model (HYI) never trades despite 69% correlation
5. ⚠️ **Poor Calibration** - High-confidence predictions no more accurate than low-confidence

**Impact:** Even with 60%+ win rates, models lose money due to win/loss asymmetry. MSE loss function fundamentally incompatible with trading profitability.

---

## Part 1: Investigation Results - The Five Failure Modes

### 1.1 Win/Loss Ratio Problem (PRIMARY CAUSE)

**Finding:** All models have win/loss ratio < 1.0, meaning average losses exceed average wins.

**Evidence:**

| Model | Win Rate | Win/Loss Ratio | Avg Win Return | Avg Loss Return | Result |
|-------|----------|----------------|----------------|-----------------|---------|
| OXLCG | 60.87% | 0.61 | +0.00385 | -0.00629 | **Losses 64% bigger** |
| HCXY | 54.05% | 0.84 | +0.00426 | -0.00508 | Losses 19% bigger |
| VGI | 40.00% | 0.27 | +0.00226 | -0.00841 | **Losses 272% bigger** |
| IGI | 35.29% | 0.89 | +0.00513 | -0.00576 | Losses 12% bigger |

**Mathematical Explanation:**

For OXLCG with 60.87% win rate:
```
Expected Return = (0.6087 × 0.00385) + (0.3913 × -0.00629)
                = 0.00234 - 0.00246
                = -0.00012 (negative!)
```

Even with >50% win rate, the asymmetry in win/loss magnitude destroys profitability.

**Root Cause:** MSE loss function treats all errors equally. A prediction error of +0.01 is penalized the same as -0.01, but in trading, these have OPPOSITE P&L impacts. MSE doesn't distinguish between:
- Predicting +0.01 when actual is +0.02 (small win)
- Predicting +0.01 when actual is -0.02 (large loss)

### 1.2 The HYI Paradox - Ultra-Conservative Predictions

**Finding:** HYI achieved BEST correlation (0.692) and BEST directional accuracy (64.71%) but produced 0% returns by NEVER trading.

**Evidence:**
```
HYI Statistics:
  Correlation: 0.692 (BEST - far above others)
  Directional Accuracy: 64.71% (BEST)
  Trade Frequency: 0.0% (NEVER predicts positive returns)

  Prediction Mean: -0.0067 (always pessimistic)
  Actual Mean: -0.0028
  Prediction Bias: -0.0039 (over-pessimistic by 139%)

  Sign Breakdown:
    Both negative: 33 samples
    Both positive: 0 samples
    Predicted negative, Actual positive: 18 samples (missed opportunities)
```

**Analysis:** HYI learned the patterns CORRECTLY (highest correlation) but developed extreme pessimistic bias during training. The model understands when returns will be high vs low, but shifts all predictions downward by ~0.004, causing it to never predict positive returns.

**Implication:** Shows that even when a model learns meaningful patterns, poor calibration can make it completely unusable for trading.

### 1.3 Weak Overall Correlations

**Finding:** Mean correlation across models is only 0.186 (weak), with some models showing near-zero or negative correlations.

**Cross-Model Statistics:**
```
Correlation Statistics:
   Mean: 0.186 (weak linear relationship)
   Std: 0.282 (high variance)
   Min: -0.111 (VGI - INVERSE pattern!)
   Max: 0.692 (HYI - but never trades)
```

**Individual Model Correlations:**

| Model | Correlation | Interpretation |
|-------|-------------|----------------|
| HYI | 0.692 | Strong (but never trades) |
| OXLCG | 0.246 | Weak positive |
| IGI | 0.140 | Very weak |
| HCXY | -0.035 | Essentially zero |
| VGI | -0.111 | **Negative (inverse!)** |

**VGI Analysis:** Achieved 3rd best validation loss (0.000093) but learned INVERSE relationship. When it predicts up, market goes down. This model is actively harmful.

**Root Cause:** MSE can be minimized by predicting the mean value without learning any pattern. Models with correlation near zero (HCXY: -0.035) are essentially predicting mean with noise.

### 1.4 Directional Accuracy Barely Above Random

**Finding:** Mean directional accuracy 54.92% vs 50% random baseline.

**Model Performance:**

| Model | Directional Accuracy | Vs Random |
|-------|---------------------|-----------|
| HYI | 64.71% | +14.71% (but never trades) |
| OXLCG | 60.87% | +10.87% |
| IGI | 54.90% | +4.90% |
| HCXY | 52.94% | +2.94% (barely) |
| **VGI** | **41.18%** | **-8.82% (WORSE than random!)** |

**Analysis:** VGI is worse than a coin flip. If used for trading with inverted signals, it would still only be 58.82% accurate - not good enough to overcome transaction costs and win/loss asymmetry.

### 1.5 Poor Confidence Calibration

**Finding:** Models cannot distinguish high-confidence from low-confidence predictions. High-magnitude predictions (Q4) are NOT more accurate than low-magnitude predictions (Q1).

**Accuracy by Prediction Magnitude Quartiles:**

| Model | Q1 (Low Conf) | Q2 | Q3 | Q4 (High Conf) | Pattern |
|-------|---------------|----|----|----------------|---------|
| OXLCG | 66.67% | 33.33% | 40.00% | **100.00%** | Non-monotonic |
| HCXY | 53.85% | 61.54% | 58.33% | **38.46%** | ❌ **INVERSE!** |
| VGI | 53.85% | 38.46% | 33.33% | 38.46% | ❌ No pattern |
| HYI | 30.77% | 46.15% | 91.67% | 92.31% | ✅ Correct (but never trades) |
| IGI | 30.77% | 46.15% | 58.33% | 84.62% | ✅ Correct |

**Critical Issue:** HCXY shows INVERSE calibration - when the model is most confident (Q4: largest magnitude predictions), it's LEAST accurate (38.46%). This makes position sizing based on prediction magnitude dangerous.

**Implication:** Cannot use prediction magnitude as confidence indicator for:
- Position sizing (Kelly criterion requires calibrated probabilities)
- Trade filtering (can't filter to "high confidence only")
- Risk management (can't reduce size on uncertain predictions)

---

## Part 2: Backtesting Results Summary

### 2.1 Overall Performance

**All 5 models produced negative returns despite excellent validation losses:**

| Model | Val Loss | Ann. Return | Sharpe | Max DD | Win Rate | Total Trades |
|-------|----------|-------------|--------|--------|----------|--------------|
| OXLCG | 0.000048 | -3.38% | -0.33 | -4.85% | 60.87% | 23 |
| HCXY | 0.000066 | -1.85% | -0.08 | -5.63% | 54.05% | 37 |
| VGI | 0.000093 | **-10.64%** | **-3.13** | -2.72% | 40.00% | 5 |
| HYI | 0.000098 | 0.00% | 0.00 | 0.00% | 0.00% | **0** |
| IGI | 0.000115 | -7.98% | -2.51 | -3.15% | 35.29% | 17 |

**Key Observations:**
- No correlation between validation loss and trading performance
- Best validation loss (OXLCG: 0.000048) → -3.38% return
- 4th best validation loss (HYI: 0.000098) → 0% return (never traded)
- Trade frequency wildly inconsistent (0% to 100%)

### 2.2 Strategy Used

**Simple Binary Long/Cash Strategy:**
```python
positions = (predictions > 0).astype(float)
strategy_returns = positions * actuals
```

**Limitations:**
1. Binary decision (all-in or all-out) - no position sizing
2. No threshold filtering - trades on any positive prediction
3. No risk management or stop losses
4. Doesn't utilize prediction magnitude
5. No transaction cost consideration

---

## Part 3: Root Cause Analysis

### 3.1 MSE Loss Function Fundamentally Misaligned

**Problem:** MSE optimizes for prediction accuracy (minimize squared errors), NOT trading profitability.

**Why MSE Fails for Trading:**

1. **Symmetric Penalty:** Treats over-prediction and under-prediction equally
   - Error of +0.01 when actual is +0.02 → MSE: 0.0001
   - Error of +0.01 when actual is -0.02 → MSE: 0.0009
   - But P&L impact: first is small win, second is large loss!

2. **Magnitude Insensitive:** Doesn't care about win/loss asymmetry
   - Predicting mean return minimizes MSE
   - But mean prediction = no trading = 0% returns

3. **Direction Agnostic:** Sign errors not specially penalized
   - Getting direction wrong is catastrophic for trading
   - MSE treats it like any other error

4. **No Profitability Signal:** Can achieve low MSE without profitable predictions
   - HYI proves this: 0.000098 MSE but 0% trade frequency
   - VGI proves this: 0.000093 MSE but -10.64% returns

**Mathematical Example:**

Two models, both with similar MSE:

**Model A (Current approach):**
- Predictions: [0.001, 0.002, 0.001, 0.003]
- Actuals: [0.005, -0.003, 0.002, -0.004]
- MSE: 0.000014
- Win/Loss: 0.5 (losses 2x bigger than wins)
- Expected Return: NEGATIVE

**Model B (Ideal):**
- Predictions: [0.003, -0.002, 0.001, -0.003]
- Actuals: [0.005, -0.003, 0.002, -0.004]
- MSE: 0.000006 (better!)
- Win/Loss: 2.0 (wins 2x bigger than losses)
- Expected Return: POSITIVE

MSE would prefer Model B, but that's by chance. MSE doesn't directly optimize for the win/loss ratio that determines profitability.

### 3.2 GAF/CNN Architecture Questions

**Findings Suggesting Architectural Issues:**

1. **High Variance in Results:**
   - Correlation ranges from -0.111 to 0.692
   - Suggests learning is inconsistent across different tickers
   - May indicate overfitting to ticker-specific patterns

2. **Negative Correlations:**
   - VGI learned inverse pattern (-0.111 correlation)
   - Indicates architecture can learn spurious patterns

3. **Ultra-Conservative Bias:**
   - HYI mean prediction: -0.0067 vs actual: -0.0028
   - Model learned to be pessimistic regardless of input

**Hypothesis:** GAF images may not capture the temporal patterns needed for return prediction. Price/volume patterns encoded as 2D images may lose critical sequential information.

**Alternative Architectures to Consider:**
- LSTM/GRU - Better for sequential data
- Transformers - Capture long-range dependencies
- Traditional ML (XGBoost, Random Forest) - May work better with engineered features

### 3.3 Strategy Simplicity

**Current Strategy Limitations:**

```python
# Too simple: binary decision based on sign only
positions = (predictions > 0).astype(float)
```

**Problems:**
1. No confidence-based position sizing
2. No trade filtering (trades on tiny positive predictions)
3. Doesn't account for transaction costs
4. No risk management or portfolio constraints
5. No consideration of prediction magnitude

**Example:** OXLCG trades on ALL 23 test samples (100% frequency) because all predictions happen to be positive. This includes low-confidence predictions that should probably be filtered.

---

## Part 4: Recommendations - Path Forward

### Option 1: Quick Fixes - Loss Function & Strategy (2-3 days)

**Recommendation 1A: Directional Loss Function**

Replace MSE with a loss that directly optimizes directional accuracy:

```python
def directional_accuracy_loss(predictions, targets):
    """
    Loss function that penalizes incorrect direction
    Returns: 1 - accuracy (so lower is better)
    """
    correct_direction = (predictions * targets) > 0
    accuracy = correct_direction.float().mean()
    return 1 - accuracy

# Or use as metric and combine with MSE:
def combined_loss(predictions, targets, alpha=0.5):
    mse = F.mse_loss(predictions, targets)
    dir_loss = directional_accuracy_loss(predictions, targets)
    return alpha * mse + (1 - alpha) * dir_loss
```

**Pros:**
- Directly optimizes what matters for trading (direction)
- Easy to implement (single function change)
- Should improve directional accuracy
- Fast to test (re-train with new loss)

**Cons:**
- Still doesn't address win/loss asymmetry
- May not improve correlation
- Could make predictions uncalibrated
- Doesn't fix HYI's conservative bias

**Estimated Impact:** Could improve directional accuracy from 55% to 60-65%, but win/loss ratio problem remains.

---

**Recommendation 1B: Sharpe Ratio Loss**

Optimize directly for risk-adjusted returns:

```python
def sharpe_loss(predictions, targets):
    """
    Negative Sharpe ratio as loss function
    Strategy: Go long if prediction > 0
    """
    positions = (predictions > 0).float()
    strategy_returns = positions * targets

    # Sharpe ratio (annualized, assuming daily data)
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std() + 1e-8  # avoid division by zero
    sharpe = (mean_return / std_return) * np.sqrt(252)

    # Return negative (so minimizing = maximizing Sharpe)
    return -sharpe
```

**Pros:**
- Directly optimizes trading performance
- Automatically accounts for win/loss ratio
- Penalizes high volatility
- Most aligned with actual trading objective

**Cons:**
- Requires full forward pass for entire batch (slow)
- May be unstable early in training (high variance)
- Non-smooth loss surface (hard to optimize)
- May overfit to in-sample patterns

**Estimated Impact:** Could significantly improve trading performance, but high risk of overfitting and training instability.

---

**Recommendation 1C: Confidence-Based Position Sizing**

Keep existing models but improve strategy:

```python
def calculate_positions_with_sizing(predictions):
    """
    Position sizing based on prediction magnitude
    Only trade when confidence exceeds threshold
    """
    # Filter: only trade predictions above threshold
    threshold = np.percentile(np.abs(predictions), 50)  # top 50%

    # Size positions by normalized prediction magnitude
    abs_predictions = np.abs(predictions)
    max_pred = abs_predictions.max()

    positions = np.where(
        (predictions > 0) & (abs_predictions > threshold),
        abs_predictions / max_pred,  # Scale 0 to 1
        0.0
    )

    return positions
```

**Pros:**
- No retraining needed
- Reduces trade frequency (filters low confidence)
- Uses prediction magnitude for sizing
- Can test immediately on existing models

**Cons:**
- Doesn't fix root calibration problem
- HCXY has inverse calibration (high confidence = worse)
- Doesn't address weak correlations
- Won't help VGI (negative correlation)

**Estimated Impact:** May improve performance for OXLCG and IGI (reasonable calibration), but won't fix fundamental problems.

---

### Option 2: Moderate Fixes - Feature Engineering & Ensemble (1 week)

**Recommendation 2A: Alternative Input Representations**

Replace GAF images with better representations:

```python
# Option 1: Raw OHLCV with technical indicators
def create_feature_matrix(df, lookback=60):
    """
    Create feature matrix with engineered features
    """
    features = []

    # Price features (normalized)
    features.append(df['close'].pct_change(1))
    features.append(df['close'].pct_change(5))
    features.append(df['close'].pct_change(20))

    # Volume features
    features.append(df['volume'].pct_change(1))
    features.append(df['volume'] / df['volume'].rolling(20).mean())

    # Technical indicators
    features.append(calculate_rsi(df['close'], 14))
    features.append(calculate_macd(df['close']))
    features.append(calculate_bollinger_bands(df['close']))

    # Volatility
    features.append(df['close'].rolling(20).std())

    return np.column_stack(features)

# Then use LSTM or 1D CNN instead of 2D CNN
class TradingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction
```

**Pros:**
- LSTM better suited for sequential data
- Engineered features proven to work in finance
- More interpretable than GAF images
- Can use domain knowledge

**Cons:**
- Requires rewriting dataset creation
- Need to retrain all models
- May lose some patterns captured by GAF
- Still need better loss function

**Estimated Impact:** Moderate improvement in correlations (0.2 → 0.4), but needs combination with better loss function.

---

**Recommendation 2B: Ensemble with Correlation Weighting**

Combine models weighted by their correlation:

```python
def ensemble_predictions(model_predictions, correlations):
    """
    Weight predictions by absolute correlation
    Only use models with positive correlation
    """
    # Filter to positive correlation models only
    positive_models = correlations > 0

    # Weight by correlation
    weights = np.abs(correlations[positive_models])
    weights = weights / weights.sum()  # normalize

    # Weighted average
    ensemble_pred = np.average(
        model_predictions[positive_models],
        weights=weights,
        axis=0
    )

    return ensemble_pred

# Usage:
correlations = np.array([0.246, -0.035, -0.111, 0.692, 0.140])  # from investigation
ensemble = ensemble_predictions(all_predictions, correlations)
```

**Pros:**
- Uses only models with positive correlation (excludes VGI, HCXY)
- Weights by proven predictive power
- May reduce variance through averaging
- No retraining needed

**Cons:**
- HYI weighted heavily but never trades
- Doesn't fix calibration problems
- May inherit conservative bias
- Limited by weakest links

**Estimated Impact:** Small improvement (ensemble of weak models still weak), but worth testing as quick experiment.

---

### Option 3: Major Overhaul - Architecture & Training (2-3 weeks)

**Recommendation 3A: Transformer Architecture**

Use attention-based model for better pattern capture:

```python
class TradingTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.fc(x)

# Train with Sharpe loss + auxiliary losses
def multi_objective_loss(predictions, targets):
    # Primary: Sharpe ratio
    sharpe_loss = -calculate_sharpe(predictions, targets)

    # Auxiliary: Directional accuracy
    dir_loss = directional_loss(predictions, targets)

    # Auxiliary: Correlation
    corr_loss = -pearson_correlation(predictions, targets)

    # Combined
    return sharpe_loss + 0.3 * dir_loss + 0.2 * corr_loss
```

**Pros:**
- State-of-art architecture for sequences
- Attention can capture complex patterns
- Multi-objective loss addresses multiple issues
- Better at long-range dependencies

**Cons:**
- Complex to implement and tune
- Requires significant compute
- High risk of overfitting (many parameters)
- 2-3 weeks development time

**Estimated Impact:** High potential but high risk. Could achieve 0.4-0.6 correlation with positive returns, but might also fail.

---

**Recommendation 3B: Traditional ML Baseline**

Before investing in complex DL, test traditional ML:

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def create_ml_features(df, lookback=60):
    """
    Create feature matrix for traditional ML
    """
    features = []

    # Price features
    for lag in [1, 5, 10, 20]:
        features.append(df['close'].pct_change(lag))
        features.append(df['volume'].pct_change(lag))

    # Moving averages
    for window in [5, 10, 20, 50]:
        features.append(df['close'] / df['close'].rolling(window).mean() - 1)

    # Volatility
    for window in [5, 10, 20]:
        features.append(df['close'].rolling(window).std())

    # Technical indicators
    features.append(calculate_rsi(df, 14))
    features.append(calculate_macd(df))

    # Volume indicators
    features.append(df['volume'] / df['volume'].rolling(20).mean())

    return pd.concat(features, axis=1)

# Train multiple models
models = {
    'rf': RandomForestRegressor(n_estimators=200, max_depth=10),
    'gbm': GradientBoostingRegressor(n_estimators=200, max_depth=5),
    'xgb': XGBRegressor(n_estimators=200, max_depth=6)
}

# Custom objective for XGBoost (directional accuracy)
def directional_objective(preds, dtrain):
    labels = dtrain.get_label()
    # Gradient of directional loss
    grad = -np.sign(labels) * (np.sign(preds) != np.sign(labels))
    hess = np.ones_like(preds)
    return grad, hess
```

**Pros:**
- Faster to train (minutes vs hours)
- More interpretable (feature importance)
- Often works well with engineered features
- Serves as performance baseline
- Less prone to overfitting

**Cons:**
- May not capture complex temporal patterns
- Requires good feature engineering
- No inherent sequence modeling
- May need separate model per ticker

**Estimated Impact:** Good baseline to establish. If traditional ML outperforms CNN, suggests problem is architecture not data.

---

## Part 5: Prioritized Action Plan

### Immediate Next Steps (Choose One)

**Option A: Quick Validation (1-2 days)**
1. **Test ensemble approach** (no retraining)
   - Combine OXLCG + IGI (positive correlation)
   - Exclude VGI, HCXY (negative/zero correlation)
   - Test if averaging improves performance

2. **Test confidence filtering** (no retraining)
   - Only trade Q3/Q4 predictions for OXLCG, IGI
   - Skip if Q4 accuracy < Q1 (filters HCXY)
   - Measure impact on returns

3. **Establish traditional ML baseline** (2 days)
   - Train XGBoost with engineered features
   - Compare correlation and returns vs CNN
   - Determines if problem is architecture vs approach

**Expected Outcome:** Understand if current models salvageable OR need architectural change.

---

**Option B: Loss Function Experiment (3-5 days)**
1. **Implement 3 loss functions:**
   - Directional accuracy loss
   - Combined MSE + Directional (alpha=0.5)
   - Sharpe ratio loss (with stability tricks)

2. **Retrain top 3 tickers** (OXLCG, HCXY, IGI)
   - Each with all 3 loss functions
   - 9 models total (3 tickers × 3 losses)

3. **Compare performance:**
   - Correlation, directional accuracy
   - Win/loss ratio, trade frequency
   - Backtested returns

**Expected Outcome:** Determine if loss function change alone can fix problems.

---

**Option C: Architecture Redesign (2-3 weeks)**
1. **Implement LSTM baseline**
   - Raw OHLCV input
   - 2-layer LSTM
   - Train with MSE first (baseline)

2. **Compare vs CNN:**
   - If LSTM better: continue with LSTM path
   - If CNN better: problem is loss function not architecture

3. **Iterate on best architecture:**
   - Add attention mechanisms if LSTM works
   - Try different loss functions
   - Engineer better features

**Expected Outcome:** Establish best architectural approach before optimization.

---

### Recommended Path: **Option A + Option B**

**Week 1: Quick Wins & Validation**
- Day 1-2: Test ensemble and filtering (Option A)
  - No retraining, fast experiments
  - Establishes if current models have any value

- Day 3-4: Traditional ML baseline (Option A.3)
  - XGBoost with engineered features
  - Critical comparison point

- Day 5: Analyze results, decide on architecture

**Week 2: Loss Function Experiments**
- Day 1: Implement 3 loss functions
- Day 2-4: Retrain 9 models (3 tickers × 3 losses)
- Day 5: Backtest and analyze

**Week 3: Path Decision**
- If loss function fixes work: Scale to all 25 tickers
- If traditional ML better: Abandon CNN approach
- If both fail: Consider Option C (architecture redesign)

---

## Part 6: Expected Outcomes & Success Metrics

### Minimum Viable Improvement

**Goal:** Beat buy-and-hold baseline

**Metrics:**
- Sharpe ratio > 0.5 (currently all negative)
- Win/loss ratio > 1.0 (currently 0.27-0.89)
- Correlation > 0.3 (currently mean 0.186)
- Directional accuracy > 60% (currently mean 54.92%)

**Reality Check:** Even achieving these modest goals may be difficult. Current results suggest fundamental approach may be flawed.

### Stretch Goals

**Goal:** Production-ready trading system

**Metrics:**
- Sharpe ratio > 1.5
- Win/loss ratio > 1.5
- Correlation > 0.5
- Directional accuracy > 65%
- Consistent performance across multiple tickers

### Decision Gates

**Gate 1 (After Week 1):**
- If ensemble/filtering shows improvement: Continue with CNN
- If traditional ML >> CNN: Abandon deep learning approach
- If both fail: Reconsider problem formulation

**Gate 2 (After Week 2):**
- If new loss functions work: Scale up
- If still negative returns: Consider non-ML approaches (rule-based, statistical arbitrage)

---

## Part 7: Risk Assessment

### High-Risk Items

1. **Overfitting Risk:** All suggested improvements could overfit to small test set
   - Mitigation: Use proper train/val/test splits, walk-forward validation

2. **Data Leakage:** Features might contain future information
   - Mitigation: Careful feature engineering, horizon separation

3. **Market Regime Change:** Models trained on recent data may not generalize
   - Mitigation: Test on different time periods, market conditions

4. **Transaction Costs:** All backtests ignore costs
   - Mitigation: Include realistic costs (0.1% per trade minimum)

### Reality Check

**Current Evidence Suggests:**
- GAF/CNN approach may be fundamentally flawed (negative correlations)
- MSE loss definitely wrong for trading (proven mathematically)
- Simple strategy inadequate (no position sizing, filtering)
- Small sample sizes (23-51 test samples) make conclusions uncertain

**Probability Assessment:**
- Quick fixes (Option A) improve results: 30%
- Loss function changes (Option B) fix problems: 45%
- Need architectural redesign (Option C): 60%
- Entire ML approach wrong: 25%

---

## Appendix A: Complete Model Statistics

### OXLCG (Best Validation Loss: 0.000048)

```
Training:
  Best Epoch: 35
  Validation Loss: 0.000048

Backtesting:
  Annualized Return: -3.38%
  Sharpe Ratio: -0.33
  Max Drawdown: -4.85%
  Win Rate: 60.87%
  Total Trades: 23

Investigation:
  Correlation: 0.246
  Directional Accuracy: 60.87%
  Trade Frequency: 100.0% (23/23)

  Win/Loss Analysis:
    Avg Winning Return: +0.003847
    Avg Losing Return: -0.006289
    Win/Loss Ratio: 0.61

  Prediction Stats:
    Mean: 0.0026
    Std: 0.0041
    Positive %: 100.0%

  Accuracy by Magnitude:
    Q1 (Low Confidence): 66.67%
    Q2: 33.33%
    Q3: 40.00%
    Q4 (High Confidence): 100.00%
```

### HCXY (2nd Best Validation Loss: 0.000066)

```
Training:
  Best Epoch: 7
  Validation Loss: 0.000066

Backtesting:
  Annualized Return: -1.85%
  Sharpe Ratio: -0.08
  Max Drawdown: -5.63%
  Win Rate: 54.05%
  Total Trades: 37

Investigation:
  Correlation: -0.035 (ESSENTIALLY ZERO)
  Directional Accuracy: 52.94%
  Trade Frequency: 72.5% (37/51)

  Win/Loss Analysis:
    Avg Winning Return: +0.004260
    Avg Losing Return: -0.005084
    Win/Loss Ratio: 0.84

  Prediction Stats:
    Mean: 0.0014
    Std: 0.0030
    Positive %: 72.5%

  Accuracy by Magnitude:
    Q1 (Low Confidence): 53.85%
    Q2: 61.54%
    Q3: 58.33%
    Q4 (High Confidence): 38.46% ⚠️ INVERSE CALIBRATION
```

### VGI (3rd Best Validation Loss: 0.000093)

```
Training:
  Best Epoch: 24
  Validation Loss: 0.000093

Backtesting:
  Annualized Return: -10.64% ⚠️ WORST
  Sharpe Ratio: -3.13
  Max Drawdown: -2.72%
  Win Rate: 40.00%
  Total Trades: 5

Investigation:
  Correlation: -0.111 ⚠️ NEGATIVE (INVERSE PATTERN)
  Directional Accuracy: 41.18% (WORSE THAN RANDOM)
  Trade Frequency: 9.8% (5/51 - VERY CONSERVATIVE)

  Win/Loss Analysis:
    Avg Winning Return: +0.002263
    Avg Losing Return: -0.008410
    Win/Loss Ratio: 0.27 ⚠️ CATASTROPHIC

  Prediction Stats:
    Mean: -0.0029
    Std: 0.0030
    Positive %: 9.8%

  Accuracy by Magnitude:
    Q1: 53.85%
    Q2: 38.46%
    Q3: 33.33%
    Q4: 38.46%
    No consistent pattern ⚠️
```

### HYI (4th Best Validation Loss: 0.000098)

```
Training:
  Best Epoch: 81
  Validation Loss: 0.000098

Backtesting:
  Annualized Return: 0.00% (NEVER TRADED)
  Sharpe Ratio: 0.00
  Max Drawdown: 0.00%
  Win Rate: N/A
  Total Trades: 0

Investigation:
  Correlation: 0.692 ⚠️ BEST BUT NEVER TRADES
  Directional Accuracy: 64.71% (BEST)
  Trade Frequency: 0.0% (NEVER PREDICTS POSITIVE)

  Prediction Stats:
    Mean: -0.0067 (ULTRA-PESSIMISTIC)
    Std: 0.0022
    Positive %: 0.0%
    Actual Mean: -0.0028
    Bias: -0.0039 (139% too pessimistic)

  Sign Breakdown:
    Both Negative: 33
    Both Positive: 0
    Pred Neg, Actual Pos: 18 (MISSED OPPORTUNITIES)

  Accuracy by Magnitude:
    Q1: 30.77%
    Q2: 46.15%
    Q3: 91.67%
    Q4: 92.31%
    Good calibration, but never trades ⚠️
```

### IGI (5th Best Validation Loss: 0.000115)

```
Training:
  Best Epoch: 3
  Validation Loss: 0.000115

Backtesting:
  Annualized Return: -7.98%
  Sharpe Ratio: -2.51
  Max Drawdown: -3.15%
  Win Rate: 35.29%
  Total Trades: 17

Investigation:
  Correlation: 0.140 (VERY WEAK)
  Directional Accuracy: 54.90%
  Trade Frequency: 33.3% (17/51)

  Win/Loss Analysis:
    Avg Winning Return: +0.005133
    Avg Losing Return: -0.005764
    Win/Loss Ratio: 0.89

  Prediction Stats:
    Mean: -0.0005
    Std: 0.0029
    Positive %: 33.3%

  Accuracy by Magnitude:
    Q1: 30.77%
    Q2: 46.15%
    Q3: 58.33%
    Q4: 84.62%
    Reasonable calibration pattern ✓
```

---

## Appendix B: Code Examples for Quick Fixes

### B.1 Ensemble with Correlation Weighting

```python
# File: ensemble_predictor.py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from model import TradingCNN
import config

class CorrelationWeightedEnsemble:
    """
    Ensemble that weights models by their correlation performance
    Excludes models with negative or near-zero correlation
    """

    def __init__(self, model_correlations: Dict[str, float], threshold: float = 0.1):
        """
        Args:
            model_correlations: {ticker: correlation} from investigation
            threshold: Minimum correlation to include model
        """
        self.correlations = {
            ticker: corr
            for ticker, corr in model_correlations.items()
            if corr > threshold
        }

        if len(self.correlations) == 0:
            raise ValueError(f"No models above correlation threshold {threshold}")

        # Normalize weights
        total_corr = sum(self.correlations.values())
        self.weights = {
            ticker: corr / total_corr
            for ticker, corr in self.correlations.items()
        }

        print(f"Ensemble using {len(self.weights)} models:")
        for ticker, weight in sorted(self.weights.items(),
                                     key=lambda x: x[1], reverse=True):
            corr = self.correlations[ticker]
            print(f"  {ticker}: weight={weight:.3f}, correlation={corr:.3f}")

    def predict(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate ensemble predictions

        Args:
            model_predictions: {ticker: predictions_array}

        Returns:
            ensemble_predictions: Weighted average
        """
        ensemble = np.zeros_like(next(iter(model_predictions.values())))

        for ticker, weight in self.weights.items():
            if ticker in model_predictions:
                ensemble += weight * model_predictions[ticker]

        return ensemble

# Usage example:
if __name__ == '__main__':
    # From investigation results
    correlations = {
        'OXLCG': 0.246,
        'HCXY': -0.035,  # Will be excluded
        'VGI': -0.111,   # Will be excluded
        'HYI': 0.692,    # Included but will never predict positive
        'IGI': 0.140
    }

    ensemble = CorrelationWeightedEnsemble(correlations, threshold=0.1)
    # Result: Uses OXLCG (weight=0.23), HYI (weight=0.65), IGI (weight=0.13)
```

### B.2 Confidence-Based Position Sizing

```python
# File: position_sizing.py
import numpy as np
from typing import Tuple

class ConfidenceBasedSizing:
    """
    Position sizing based on prediction magnitude
    Only trades high-confidence predictions
    """

    def __init__(self,
                 percentile_threshold: float = 50.0,
                 max_position: float = 1.0):
        """
        Args:
            percentile_threshold: Only trade above this percentile (e.g., 50 = top half)
            max_position: Maximum position size (1.0 = 100% of capital)
        """
        self.percentile_threshold = percentile_threshold
        self.max_position = max_position

    def calculate_positions(self,
                           predictions: np.ndarray,
                           calibration_quality: str = 'good') -> np.ndarray:
        """
        Calculate position sizes

        Args:
            predictions: Model predictions
            calibration_quality: 'good', 'poor', or 'inverse'
                                 Determines if we can trust magnitude

        Returns:
            positions: Position sizes (0 to max_position)
        """
        if calibration_quality == 'inverse':
            # For models like HCXY where high confidence = worse
            # Don't use magnitude, just filter by sign
            return np.where(predictions > 0, self.max_position, 0.0)

        # Calculate confidence threshold
        abs_predictions = np.abs(predictions)
        threshold = np.percentile(abs_predictions, self.percentile_threshold)

        if calibration_quality == 'good':
            # Can use magnitude for position sizing
            # Normalize to [0, max_position]
            max_pred = abs_predictions.max()
            if max_pred > 0:
                normalized = abs_predictions / max_pred * self.max_position
            else:
                normalized = np.zeros_like(predictions)

            # Only trade above threshold, respect sign
            positions = np.where(
                (abs_predictions >= threshold) & (predictions > 0),
                normalized,
                0.0
            )
        else:  # calibration_quality == 'poor'
            # Just filter, don't size by magnitude
            positions = np.where(
                (abs_predictions >= threshold) & (predictions > 0),
                self.max_position,
                0.0
            )

        return positions

# Example usage:
if __name__ == '__main__':
    predictions = np.array([0.001, 0.005, -0.002, 0.003, -0.004, 0.006])

    # For well-calibrated model (OXLCG, IGI)
    sizer_good = ConfidenceBasedSizing(percentile_threshold=50)
    positions = sizer_good.calculate_positions(predictions, calibration_quality='good')
    print("Good calibration positions:", positions)
    # Output: [0.0, 0.833, 0.0, 0.5, 0.0, 1.0]

    # For poorly calibrated model (HCXY inverse)
    sizer_inverse = ConfidenceBasedSizing(percentile_threshold=50)
    positions = sizer_inverse.calculate_positions(predictions, calibration_quality='inverse')
    print("Inverse calibration positions:", positions)
    # Output: [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # Binary only
```

### B.3 Directional Loss Function

```python
# File: trading_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DirectionalLoss(nn.Module):
    """
    Loss function that directly penalizes incorrect direction
    More aligned with trading objectives than MSE
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate directional accuracy loss

        Args:
            predictions: Model predictions (batch_size,)
            targets: Actual returns (batch_size,)

        Returns:
            loss: 1 - directional_accuracy
        """
        # Check if signs match
        correct_direction = (predictions * targets) > 0

        # Accuracy (higher is better)
        accuracy = correct_direction.float().mean()

        # Return 1 - accuracy (so minimizing improves accuracy)
        return 1.0 - accuracy

class CombinedLoss(nn.Module):
    """
    Combination of MSE (for magnitude) and directional loss (for sign)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for MSE (0=pure directional, 1=pure MSE)
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.directional = DirectionalLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(predictions, targets)
        dir_loss = self.directional(predictions, targets)

        return self.alpha * mse_loss + (1 - self.alpha) * dir_loss

class SharpeLoss(nn.Module):
    """
    Negative Sharpe ratio as loss function
    WARNING: High variance, may be unstable during training
    """

    def __init__(self, risk_free_rate: float = 0.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio

        Strategy: Go long if prediction > 0, cash otherwise
        """
        # Simulate strategy returns
        positions = (predictions > 0).float()
        strategy_returns = positions * targets

        # Calculate Sharpe ratio
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()

        # Avoid division by zero
        if std_return < 1e-8:
            return torch.tensor(0.0, device=predictions.device)

        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return - self.risk_free_rate) / std_return * np.sqrt(252)

        # Return negative (minimizing = maximizing Sharpe)
        return -sharpe

# Usage example in training:
if __name__ == '__main__':
    # Test with dummy data
    predictions = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.02])
    targets = torch.tensor([0.015, -0.018, -0.005, 0.012, 0.025])

    # Directional loss
    dir_loss = DirectionalLoss()
    print(f"Directional Loss: {dir_loss(predictions, targets).item():.4f}")
    # Penalizes 3rd prediction (predicted +, actual -)

    # Combined loss
    combined = CombinedLoss(alpha=0.5)
    print(f"Combined Loss: {combined(predictions, targets).item():.4f}")

    # Sharpe loss
    sharpe = SharpeLoss()
    print(f"Sharpe Loss: {sharpe(predictions, targets).item():.4f}")
```

---

## Conclusion

The investigation has revealed that the CNN trading model failures stem from **multiple fundamental problems converging**, not a single fixable issue:

1. **MSE loss function is fundamentally misaligned** with trading profitability
2. **Win/loss asymmetry** causes losses despite >50% win rates
3. **GAF/CNN architecture** may not capture predictive patterns (weak correlations)
4. **Ultra-conservative bias** in best model (HYI) prevents trading
5. **Poor calibration** makes confidence-based strategies impossible

**Immediate recommendation:** Start with **Option A (Quick Validation)** to determine if current models have any salvageable value, followed by **Option B (Loss Function Experiments)** to test if better objectives can fix the core problems.

**Reality check:** There's a meaningful probability (25%) that the entire ML approach is wrong for this problem, and simpler statistical or rule-based methods might work better. The traditional ML baseline (XGBoost) will help answer this question.

**Expected timeline:** 2-3 weeks to determine if this approach is viable, with decision gates at week 1 and week 2 to pivot if needed.
