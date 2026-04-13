# Visual Trading System - Backtesting Analysis Report
**Date:** 2026-01-03
**Models Tested:** Top 5 performers from batch training (OXLCG, HCXY, VGI, HYI, IGI)
**Test Period:** Held-out test sets (15% of data, ~23-51 samples per ETF)

## Executive Summary

**CRITICAL FINDING:** Despite excellent validation losses during training (as low as 0.000048 MSE), all five top-performing CNN models produced **negative returns** when backtested on held-out data using a simple long-only trading strategy.

**Key Results:**
- ❌ **Average Total Return:** -0.92% (all models lost money)
- ❌ **Average Annualized Return:** -4.77% (portfolio would lose ~5% per year)
- ❌ **Average Sharpe Ratio:** -1.21 (negative risk-adjusted returns)
- ⚠️ **Average Win Rate:** 38.04% (worse than random)
- ⚠️ **Average Prediction Accuracy:** 54.92% (barely better than 50% random)

**Fundamental Insight:** Low MSE validation loss does NOT predict profitable trading performance. This represents a critical disconnect between the training objective (minimize prediction error) and the trading objective (maximize risk-adjusted returns).

---

## Backtesting Methodology

### Strategy Implementation
```python
# Simple Long-Only Strategy
positions = (predictions > 0).astype(float)
strategy_returns = positions * actuals

# Position Logic:
# - If model predicts positive return → Go long (position = 1.0)
# - If model predicts negative return → Hold cash (position = 0.0)
# - No short positions, no leverage
```

### Data Split
- **Train:** 70% (temporal ordering preserved)
- **Validation:** 15% (for early stopping during training)
- **Test:** 15% (**HELD-OUT** - never seen during training)

**Critical:** Test data is truly unseen by the model, making these results representative of real-world performance.

### Performance Metrics

**Return Metrics:**
- **Total Return:** Cumulative wealth change over test period
- **Annualized Return:** Geometric mean return scaled to 252 trading days
- **Sharpe Ratio:** (mean_return / std_return) * sqrt(252), risk-free rate = 0

**Risk Metrics:**
- **Maximum Drawdown:** Worst peak-to-trough decline during test period
- **Win Rate:** Percentage of profitable trades among all trades taken

**Accuracy Metrics:**
- **Prediction Accuracy:** Directional correctness (predicted sign matches actual sign)
- **Total Trades:** Number of times model predicted positive return

### Assumptions & Limitations
- No transaction costs (real-world results would be worse)
- No slippage (assumes perfect execution at close prices)
- No position sizing based on confidence (binary long/cash)
- Daily rebalancing (may not be realistic)
- No consideration of market regime (bull/bear/sideways)

---

## Detailed Model Results

### Rank 1: OXLCG (Best Validation Loss)
**Training Performance:**
- Validation Loss: **0.000048** (best among all 25 models)
- Best Epoch: 35
- Dataset: 147 samples (smallest dataset)

**Backtesting Performance:**
- Total Return: **-0.31%** ❌
- Annualized Return: **-3.38%** ❌
- Sharpe Ratio: **-0.33** ❌
- Max Drawdown: **-4.85%**
- Win Rate: **60.87%** (best among 5 models)
- Prediction Accuracy: **60.87%**
- Total Trades: **23** (out of 23 test samples)
- Winning Trades: **14**

**Analysis:**
- Despite lowest validation loss, still lost money on every metric
- High trade frequency (100% of samples) shows model is confident
- Win rate of 60.87% suggests directional accuracy is decent
- **BUT:** Losing trades must be larger than winning trades (negative return despite 60% win rate)
- This indicates model may correctly predict direction but underestimate magnitude of losses

---

### Rank 2: HCXY (2nd Best Validation Loss)
**Training Performance:**
- Validation Loss: **0.000066**
- Best Epoch: 7 (very fast convergence)
- Dataset: 335 samples (full dataset)

**Backtesting Performance:**
- Total Return: **-0.38%** ❌
- Annualized Return: **-1.85%** ❌
- Sharpe Ratio: **-0.08** ❌
- Max Drawdown: **-5.63%** (worst drawdown)
- Win Rate: **54.05%**
- Prediction Accuracy: **52.94%**
- Total Trades: **37** (out of 51 test samples)
- Winning Trades: **20**

**Analysis:**
- Second-worst annualized return
- Win rate slightly above random (54% vs 50%)
- Prediction accuracy barely better than coin flip (52.94%)
- Made 37 trades (72% trade rate) but still lost money
- Fast convergence (epoch 7) may indicate model found simple but unprofitable pattern

---

### Rank 3: VGI (3rd Best Validation Loss)
**Training Performance:**
- Validation Loss: **0.000093**
- Best Epoch: 24
- Dataset: 335 samples

**Backtesting Performance:**
- Total Return: **-2.25%** ❌ (worst total return)
- Annualized Return: **-10.64%** ❌ (worst annualized return)
- Sharpe Ratio: **-3.13** ❌ (worst risk-adjusted return)
- Max Drawdown: **-2.72%**
- Win Rate: **40.00%** (below random)
- Prediction Accuracy: **41.18%** (worse than random!)
- Total Trades: **5** (out of 51 test samples, only 9.8% trade rate)
- Winning Trades: **2**

**Analysis:**
- **WORST PERFORMER** despite 3rd best validation loss
- Only traded 5 times out of 51 opportunities (very conservative)
- Win rate 40%, accuracy 41% - both worse than random (50%)
- Model is both infrequent AND inaccurate when it does trade
- Suggests model learned to avoid trading (conservative) but when it does trade, it's wrong
- This is the clearest example of validation loss NOT correlating with profitability

---

### Rank 4: HYI (4th Best Validation Loss)
**Training Performance:**
- Validation Loss: **0.000098**
- Best Epoch: 81 (long training time)
- Dataset: 335 samples

**Backtesting Performance:**
- Total Return: **0.00%** (no trades = no returns)
- Annualized Return: **0.00%**
- Sharpe Ratio: **0.00**
- Max Drawdown: **0.00%**
- Win Rate: **0.00%** (undefined, no trades)
- Prediction Accuracy: **64.71%** (highest accuracy!)
- Total Trades: **0** ❌ (NEVER predicted positive return)
- Winning Trades: **0**

**Analysis:**
- **MOST INTERESTING FAILURE MODE:** Never generated a single buy signal
- Model ALWAYS predicted negative returns throughout entire test period
- Yet has highest prediction accuracy (64.71%) for directional correctness
- This means: Model correctly predicted many down days, but also avoided many up days
- **Ultra-conservative:** Learned to avoid risk entirely
- Result: No losses, but also no gains (equivalent to holding cash)
- This suggests model may have learned market is risky and best strategy is to never trade

---

### Rank 5: IGI (5th Best Validation Loss)
**Training Performance:**
- Validation Loss: **0.000115**
- Best Epoch: 3 (very fast convergence)
- Dataset: 335 samples

**Backtesting Performance:**
- Total Return: **-1.67%** ❌
- Annualized Return: **-7.98%** ❌ (second worst)
- Sharpe Ratio: **-2.51** ❌ (second worst)
- Max Drawdown: **-3.15%**
- Win Rate: **35.29%** (worst win rate)
- Prediction Accuracy: **54.90%**
- Total Trades: **17** (out of 51 test samples, 33% trade rate)
- Winning Trades: **6**

**Analysis:**
- Second-worst performer overall
- Converged very fast (epoch 3) like HCXY, suggests simple pattern
- Made 17 trades but only won 6 (35.29% win rate)
- Prediction accuracy 54.90% is close to random despite trading
- Fast convergence + poor results suggests model found local minimum quickly
- Model is picking wrong trades (only 35% win rate)

---

## Performance Comparison Analysis

### By Sharpe Ratio (Risk-Adjusted Returns)
| Rank | Ticker | Sharpe | Ann. Return | Max DD | Accuracy |
|------|--------|--------|-------------|--------|----------|
| 1 | HYI | 0.00 | 0.00% | 0.00% | 64.71% |
| 2 | HCXY | -0.08 | -1.85% | -5.63% | 52.94% |
| 3 | OXLCG | -0.33 | -3.38% | -4.85% | 60.87% |
| 4 | IGI | -2.51 | -7.98% | -3.15% | 54.90% |
| 5 | VGI | -3.13 | -10.64% | -2.72% | 41.18% |

**Insight:** HYI "wins" by not playing (0% return, 0 Sharpe). All other models have deeply negative Sharpe ratios.

### By Total Return
| Rank | Ticker | Total Return | Win Rate | Trades | Val Loss |
|------|--------|-------------|----------|--------|----------|
| 1 | HYI | 0.00% | 0.00% | 0 | 0.000098 |
| 2 | OXLCG | -0.31% | 60.87% | 23 | 0.000048 |
| 3 | HCXY | -0.38% | 54.05% | 37 | 0.000066 |
| 4 | IGI | -1.67% | 35.29% | 17 | 0.000115 |
| 5 | VGI | -2.25% | 40.00% | 5 | 0.000093 |

**Insight:** NO correlation between validation loss and total return. Best validation loss (OXLCG: 0.000048) still lost -0.31%.

### By Win Rate
| Rank | Ticker | Win Rate | Accuracy | Trades | Comments |
|------|--------|----------|----------|--------|----------|
| 1 | OXLCG | 60.87% | 60.87% | 23 | Best win rate, still lost money |
| 2 | HCXY | 54.05% | 52.94% | 37 | Barely above random |
| 3 | VGI | 40.00% | 41.18% | 5 | Below random |
| 4 | IGI | 35.29% | 54.90% | 17 | Worst win rate |
| 5 | HYI | 0.00% | 64.71% | 0 | Never traded |

**Insight:** Even 60% win rate (OXLCG) resulted in losses. Losing trades must be larger than winning trades.

---

## Cross-Model Patterns

### Pattern 1: Validation Loss ≠ Profitability

**Evidence:**
- OXLCG: Best val loss (0.000048) → -3.38% annualized return
- VGI: 3rd best val loss (0.000093) → WORST return (-10.64% annualized)
- No correlation between validation loss ranking and profitability

**Why This Happens:**

**MSE Loss Optimization:**
```python
loss = mean((predicted_return - actual_return)^2)
```

**MSE focuses on magnitude accuracy, not directional accuracy:**
- Predicting +0.5% when actual is +1.0% → MSE = 0.25%
- Predicting -0.5% when actual is +1.0% → MSE = 2.25%

**Both are poor for trading:**
- First prediction: Correct direction, makes money (good for trading)
- Second prediction: Wrong direction, loses money (bad for trading)
- But MSE is 9x worse for second prediction

**Trading P&L only cares about sign:**
- Predict +0.01%, actual +1.0% → Make money ✅
- Predict +0.99%, actual -1.0% → Lose money ❌

**Conclusion:** MSE is the wrong objective function for trading. Need profit-based or directional accuracy loss.

### Pattern 2: Fast Convergence ≠ Good Results

**Fast Convergers (epochs 3-7):**
- IGI: epoch 3 → -7.98% annualized
- HCXY: epoch 7 → -1.85% annualized

**Slow Convergers (epochs 35-81):**
- OXLCG: epoch 35 → -3.38% annualized
- HYI: epoch 81 → 0.00% annualized

**No clear pattern:** Fast and slow convergers both failed to profit.

### Pattern 3: Prediction Accuracy Doesn't Guarantee Profit

**High Accuracy Models:**
- HYI: 64.71% accuracy → 0% return (never traded)
- OXLCG: 60.87% accuracy → -3.38% annualized

**Paradox:** HYI has highest accuracy but never trades. How?
- Answer: HYI correctly predicts many negative return days
- But also incorrectly avoids many positive return days
- High accuracy comes from correctly predicting "don't trade" (negative return prediction)

### Pattern 4: Win Rate vs. P&L Mismatch

**OXLCG Example:**
- Win Rate: 60.87% (14 wins, 9 losses)
- Total Return: -0.31% ❌

**How can 60% win rate lose money?**
- Average winning trade: Must be smaller than average losing trade
- Example: Win $1 on 14 trades = +$14, Lose $2 on 9 trades = -$18
- Net: -$4 despite 60% win rate

**Conclusion:** Models may be good at identifying small wins but bad at avoiding large losses.

### Pattern 5: Low Trade Frequency

**Trade Frequency Distribution:**
- HYI: 0% (0 trades)
- VGI: 9.8% (5/51 samples)
- IGI: 33% (17/51 samples)
- HCXY: 72% (37/51 samples)
- OXLCG: 100% (23/23 samples)

**Most models avoid trading:**
- Average trade frequency: 43% (excluding OXLCG with different dataset size)
- Models learned to be conservative
- But when they do trade, accuracy is barely above random

---

## Root Cause Analysis

### Hypothesis 1: Training Objective Mismatch ⭐ MOST LIKELY

**Problem:** MSE loss optimizes for prediction magnitude accuracy, not trading profitability.

**Evidence:**
- Models with lowest MSE still lose money
- No correlation between validation loss and returns
- Even 60% win rate (OXLCG) results in losses

**Solution Direction:**
- Replace MSE with profit-based loss function
- Use Sharpe ratio as loss function
- Try directional accuracy loss: `loss = -sign(pred) * sign(actual)`
- Consider classification approach (up/down/neutral)

### Hypothesis 2: Strategy Too Simple ⭐ LIKELY

**Problem:** Binary long/cash strategy doesn't use prediction magnitude.

**Evidence:**
- Models predict continuous returns but strategy only uses sign
- No position sizing based on confidence
- No risk management or stop losses

**Solution Direction:**
- Use prediction magnitude for position sizing
- Implement confidence thresholds (only trade when |prediction| > threshold)
- Add short positions (long/short instead of long/cash)
- Implement dynamic position sizing based on volatility

### Hypothesis 3: GAF Transformation Not Predictive ⭐ POSSIBLE

**Problem:** Gramian Angular Field images may not capture tradeable patterns.

**Evidence:**
- Even models with low loss can't predict profitable trades
- Prediction accuracy barely above random (54.92% average)
- Visual patterns in GAF may not correlate with future returns

**Solution Direction:**
- Try raw price series with LSTM instead of GAF+CNN
- Test alternative image transformations (MTF, RP, etc.)
- Use traditional technical indicators instead of images
- Validate that GAF actually contains predictive information

### Hypothesis 4: Horizon Mismatch ⭐ POSSIBLE

**Problem:** 5-day prediction horizon may be wrong for this task.

**Evidence:**
- Models struggle to predict 5-day returns
- Short-term noise may dominate signal
- Daily rebalancing may be too frequent

**Solution Direction:**
- Test different horizons (1-day, 10-day, 20-day)
- Match horizon to natural market cycles
- Consider multiple horizons simultaneously

### Hypothesis 5: Market Regime Not Considered ❓ UNKNOWN

**Problem:** Models may work in some market conditions but not others.

**Evidence:**
- Test periods are short (23-51 samples per ETF)
- No analysis of performance by market regime
- May have tested during unfavorable regime

**Solution Direction:**
- Analyze performance by volatility regime
- Analyze performance by trend regime (bull/bear/sideways)
- Implement regime-aware models or strategies

### Hypothesis 6: Data Leakage or Look-Ahead Bias ❌ UNLIKELY

**Problem:** Model might be seeing future data during training.

**Assessment:**
- Data split is temporal (70/15/15)
- Test set is truly held-out
- No evidence of leakage in code review
- If leakage existed, models would perform BETTER, not worse

**Conclusion:** Not the issue here.

### Hypothesis 7: Overfitting to Noise ⚠️ POSSIBLE

**Problem:** Models may be fitting to random noise instead of true patterns.

**Evidence:**
- Fast convergence (IGI: epoch 3) suggests finding simple patterns
- But simple patterns don't generalize to profit
- Prediction accuracy barely above random

**Solution Direction:**
- Increase regularization (dropout, weight decay)
- Reduce model capacity
- Use ensemble methods
- Implement cross-validation across time periods

---

## Comparison to Training Results

### OXLCG: Training vs. Backtesting

**Training:**
- Best Val Loss: 0.000048 ✅
- Best Epoch: 35
- Final Train Loss: 0.000056
- Final Val Loss: 0.000057

**Backtesting:**
- Total Return: -0.31% ❌
- Annualized Return: -3.38% ❌
- Sharpe Ratio: -0.33 ❌

**Disconnect:** Model achieves near-perfect loss minimization but loses money in trading.

### VGI: Training vs. Backtesting

**Training:**
- Best Val Loss: 0.000093 ✅ (3rd best)
- Best Epoch: 24
- Final Train Loss: 0.000175
- Final Val Loss: 0.000098

**Backtesting:**
- Total Return: -2.25% ❌ (worst)
- Annualized Return: -10.64% ❌ (worst)
- Sharpe Ratio: -3.13 ❌ (worst)

**Disconnect:** 3rd best validation loss → WORST trading performance. Zero correlation.

---

## Recommendations

### Immediate Next Steps

**1. Investigate Prediction Distributions** (Quick Analysis)
- Plot histogram of predicted returns vs. actual returns
- Analyze prediction bias (are predictions systematically too high/low?)
- Check if models are predicting returns with correct magnitude but wrong sign
- Identify which market conditions lead to correct vs. incorrect predictions

**2. Analyze Individual Trades** (Deep Dive)
- Review the 5 largest losing trades for each model
- Review the 5 largest winning trades for each model
- Identify patterns in when models succeed vs. fail
- Check if certain ETFs or time periods are particularly difficult

**3. Test Alternative Strategies** (Quick Experiment)
```python
# Instead of binary long/cash:
# 1. Threshold strategy
positions = np.where(predictions > threshold, 1.0, 0.0)

# 2. Proportional strategy
positions = np.clip(predictions / max_pred, 0, 1)

# 3. Long/short strategy
positions = np.sign(predictions)  # -1, 0, or +1
```

### Medium-Term Fixes

**4. Change Loss Function** (High Priority)
- Replace MSE with directional accuracy loss
- Implement Sharpe ratio loss function
- Try classification approach (up/down/neutral classes)
- Use ranking loss (pairwise comparisons)

**5. Improve Strategy** (High Priority)
- Implement position sizing based on prediction confidence
- Add transaction cost modeling (0.1% per trade)
- Implement risk management (stop losses, position limits)
- Try ensemble strategies (average predictions across multiple models)

**6. Expand Backtesting** (Medium Priority)
- Test all 25 models (not just top 5)
- Implement walk-forward analysis (rolling train/test)
- Test across different time periods
- Analyze performance by market regime

### Long-Term Architectural Changes

**7. Alternative Approaches** (Evaluate)
- **LSTM on raw price series** instead of GAF+CNN
- **Transformer architecture** for sequence modeling
- **Traditional ML** (XGBoost, Random Forest) on technical indicators
- **Reinforcement Learning** for direct profit optimization
- **Meta-labeling** approach (predict confidence, not return)

**8. Feature Engineering** (Investigate)
- Test alternative image transformations (MTF, Recurrence Plots)
- Add fundamental data (volume, volatility, correlations)
- Include market regime indicators
- Use multiple timeframes simultaneously

**9. Ensemble Methods** (Promising)
- Combine predictions from multiple models
- Weight by model confidence or historical performance
- Use model diversity to improve robustness
- Implement online learning to adapt to changing markets

### Strategic Decisions Required

**Decision 1: Continue with CNN/GAF approach?**
- ❌ **Against:** No evidence that GAF captures tradeable patterns
- ✅ **For:** Haven't tried optimal loss function or strategy yet
- **Recommendation:** Try loss function changes first, then decide

**Decision 2: What is acceptable performance?**
- **Minimum:** Sharpe > 0 (better than cash)
- **Target:** Sharpe > 1.0 (good risk-adjusted return)
- **Realistic:** Sharpe 0.5-0.8 may be achievable with improvements

**Decision 3: What timeline/resources?**
- Quick fixes (1-2 days): Strategy improvements, different thresholds
- Medium changes (1 week): New loss functions, ensemble methods
- Major overhaul (2-4 weeks): New architectures (LSTM, Transformer)

---

## Conclusions

### Key Findings

1. **Validation Loss ≠ Profitability** ⭐ CRITICAL
   - Best validation loss (0.000048) still lost -3.38% annualized
   - No correlation between MSE and trading returns
   - Training objective fundamentally misaligned with trading objective

2. **All Models Unprofitable** ❌
   - Average annualized return: -4.77%
   - Average Sharpe ratio: -1.21
   - Best case: 0% return (HYI never traded)

3. **Prediction Accuracy Barely Above Random** ⚠️
   - Average accuracy: 54.92% (random is 50%)
   - Average win rate: 38.04%
   - Models are not learning strong predictive patterns

4. **Strategy Too Simple** 🔧
   - Binary long/cash doesn't use prediction magnitude
   - No position sizing, risk management, or thresholds
   - Could be losing money on strategy, not model

5. **GAF+CNN Approach Questionable** ❓
   - No evidence that visual patterns predict returns
   - Alternative approaches (LSTM, traditional ML) may work better
   - Need to validate that GAF actually contains predictive information

### Overall Assessment

**Current State:** ❌ **NOT PRODUCTION READY**
- System cannot be deployed for real trading
- All models lose money on average
- Performance is worse than holding cash

**Root Cause:** 🎯 **Training Objective Mismatch**
- MSE loss optimizes wrong thing (magnitude accuracy vs. profitability)
- Need profit-based or directional accuracy loss function

**Path Forward:** 🔧 **Requires Architectural Changes**
1. **Quick Fix:** Change loss function to directional accuracy or Sharpe-based
2. **Medium Fix:** Improve trading strategy (thresholds, position sizing)
3. **Long Fix:** Evaluate alternative approaches (LSTM, traditional ML)

**Time Investment:**
- Original training: 9.12 minutes (successful) ✅
- Validation: Successful (25/25 models) ✅
- Backtesting: Successful (5/5 models) ✅
- **But:** System doesn't achieve goal (profitable trading) ❌

**Recommendation:** Before abandoning CNN/GAF approach, try:
1. Directional accuracy loss function (quick test)
2. Improved trading strategy with thresholds (quick test)
3. Ensemble of top models (medium effort)

If these don't improve results to Sharpe > 0, consider:
- LSTM on raw price data
- Traditional ML on technical indicators
- Accepting that this prediction task may be extremely difficult

---

**Report Generated:** 2026-01-03 21:46:10
**Backtesting Duration:** 3 seconds (5 models)
**Success Rate:** 100% (execution successful, but results negative)
**Critical Finding:** Validation loss does not predict trading profitability
**Status:** ⚠️ **REQUIRES STRATEGIC DECISION ON NEXT DIRECTION**
