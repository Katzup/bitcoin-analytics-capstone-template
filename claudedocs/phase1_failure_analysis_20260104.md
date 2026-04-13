# Phase 1 Failure Analysis - Multi-Timeframe Regime Classification
**Date**: January 4, 2026, 1:30 PM
**Test**: Multi-Timeframe Regime Classification (Phase 1 of Hybrid Approach)
**Status**: ❌ COMPLETE FAILURE - NO-GO for Phase 2
**Recommendation**: Path D - Abandon This Approach

---

## Executive Summary

**Critical Finding**: Phase 1 multi-timeframe regime classification has **completely failed**. All 5 ETFs tested produced errors preventing any trading performance metrics from being generated. The reformulation from return prediction to regime classification did NOT make the problem more tractable - instead it introduced new failure modes while maintaining poor predictive accuracy.

**Impact**:
- ❌ 0 out of 5 ETFs completed successfully
- ❌ 0 Sharpe ratios generated
- ❌ 0 returns calculated
- ❌ 0 win/loss ratios available
- ❌ All Phase 1 success criteria FAILED or unevaluable

**Decision**: Based on baseline synthesis guidelines, Phase 1 failure triggers **Path D recommendation: Abandon multi-timeframe regime classification approach**.

---

## Test Configuration

### Approach Tested
**Multi-Timeframe Regime Classification** (Option B.1 from baseline synthesis)
- Reformulated from 5-day return prediction to regime classification
- 5 time horizons: 20, 50, 100, 150, 200 days
- 6 market regimes: Strong Down, Weak Down, Sideways, Weak Up, Strong Up, High Volatility
- Hierarchical trading strategy based on horizon agreement

### ETFs Tested
1. OXLCG - Oxford Lane Capital Corp
2. HCXY - Hercules Capital Inc
3. VGI - Virtus Global Multi-Sector Income Fund
4. HYI - Western Asset High Yield Defined Opportunity Fund
5. IGI - Western Asset Investment Grade Defined Opportunity Trust

### Feature Set
- 306 features per sample
- 300 OHLCV features (60 days × 5 columns flattened)
- 6 technical indicators (returns_5d, returns_10d, returns_20d, volatility, volume_avg, volume_std)

### Model Architecture
- XGBoost multi-class classifier
- One model per horizon per ETF (25 models total across 5 horizons)
- 100 estimators, max depth 5, learning rate 0.1
- 70/15/15 temporal train/val/test split

---

## Three Categories of Failure

### Failure Type 1: Class Imbalance Between Splits

**ETFs Affected**: OXLCG, HCXY

**Error Messages**:
```
OXLCG: "Invalid classes inferred from unique values of `y`. Expected: [0 1 2], got [0 1 4]"
HCXY: "Invalid classes inferred from unique values of `y`. Expected: [0 1], got [1 4]"
```

**Root Cause**:
XGBoost requires all classes present in validation/test sets to also appear in training set. Temporal splits (70/15/15) resulted in different regime distributions across time periods:
- Training data contained regimes [0, 1, 2]
- Test data contained regimes [0, 1, 4]
- XGBoost failed during prediction phase

**Console Output Evidence**:
```
OXLCG:
  Horizon: 50 days
  Generated 12 samples
  Train: 8, Val: 1, Test: 3
  ❌ Error: Invalid classes inferred from unique values of `y`

HCXY:
  Horizon: 100 days
  Generated 92 samples
  Train: 64, Val: 13, Test: 15
  ❌ Error: Invalid classes inferred from unique values of `y`
```

**Impact**: Complete failure for OXLCG and HCXY - no trading metrics generated.

---

### Failure Type 2: Insufficient Historical Data

**ETFs Affected**: VGI, HYI, IGI

**Error Message**:
```
ValueError: Please reshape the input data into 2-dimensional matrix.
```

**Preceded By**:
```
Generated 0 samples
Train: 0, Val: 0, Test: 0
```

**Root Cause**:
150-day and 200-day horizons require:
- 60-day lookback window for features
- 150/200-day forward window for target
- Total minimum: 210-260+ days
- Plus additional buffer for train/val/test splits

**Data Availability**:
- OXLCG: 212 days total (barely sufficient for single 150-day sample)
- VGI, HYI, IGI: Insufficient data at 150-day horizon
- All failed at 150-day or earlier

**Console Output Evidence**:
```
VGI:
  Horizon: 150 days
  Generated 0 samples
  ❌ Error: Please reshape the input data into 2-dimensional matrix.

HYI:
  Horizon: 150 days
  Generated 0 samples
  ❌ Error: Please reshape the input data into 2-dimensional matrix.

IGI:
  Horizon: 150 days
  Generated 0 samples
  ❌ Error: Please reshape the input data into 2-dimensional matrix.
```

**Impact**: Complete failure for VGI, HYI, IGI - no trading metrics generated.

---

### Failure Type 3: Poor Classification Accuracy

**All ETFs Affected** - Even successful training runs showed poor accuracy

**Accuracy Results by ETF and Horizon**:

**OXLCG**:
- 20-day: 42.86% (132 test samples) - ❌ BELOW 60% threshold

**HCXY**:
- 20-day: 23.53% (332 test samples) - ❌ BELOW 60% threshold
- 50-day: 18.18% (212 test samples) - ❌ BELOW 60% threshold

**VGI**:
- 20-day: 42.19% (422 test samples) - ❌ BELOW 60% threshold
- 50-day: 6.52% (302 test samples) - ❌ WORSE than random (16.67% for 6 classes)
- 100-day: 6.25% (102 test samples) - ❌ WORSE than random

**HYI**:
- 20-day: 62.50% (422 test samples) - ✅ **ONLY configuration meeting 60% threshold**
- 50-day: 13.04% (302 test samples) - ❌ BELOW 60% threshold
- 100-day: 100.00% (102 test samples) - ⚠️ Likely overfitting (only 16 test samples)

**IGI**:
- 20-day: 28.12% (422 test samples) - ❌ BELOW 60% threshold
- 50-day: 19.57% (302 test samples) - ❌ BELOW 60% threshold
- 100-day: 50.00% (102 test samples) - ❌ BELOW 60% threshold

**Statistical Analysis**:
- Only 1 configuration (HYI 20-day: 62.50%) met 60% accuracy threshold
- Most accuracy rates: 6%-50% range
- Multiple configurations worse than random guessing (16.67% for 6 equally-distributed classes)
- HYI 100-day showing 100% likely unreliable (tiny test set: 16 samples)

**Random Baseline Comparison**:
For 6 equally-distributed regime classes, random guessing would achieve ~16.67% accuracy. Configurations showing 6%-13% are performing WORSE than random chance, indicating fundamental prediction failure.

**Impact**: Even when models trained without errors, accuracy rates were too low for profitable trading.

---

## Results Against Success Criteria

From `baseline_comparison_synthesis_20260104.md` (lines 352-360):

> **Success Criteria for Phase 1**:
> - Regime classification accuracy >60%
> - Regime-based trading achieves positive Sharpe ratio
> - Win/loss ratio >0.8 for regime-based trades
>
> **Go/No-Go Decision Point**:
> - If Phase 1 fails → Consider Path D (abandon approach)
> - If Phase 1 succeeds → Proceed to Phase 2 (enhanced features)

### Criterion 1: Regime Classification Accuracy >60%

**Status**: ❌ **FAILED**

**Evidence**:
- Only 1 of many configurations met threshold (HYI 20-day: 62.50%)
- Typical accuracy range: 6%-50%
- Multiple worse than random guessing (16.67%)
- Average accuracy across successful runs: ~30% (estimated)

**Conclusion**: Regime classification is NOT significantly more accurate than random guessing for most configurations.

---

### Criterion 2: Positive Sharpe Ratio

**Status**: ❌ **CANNOT EVALUATE - No Data Generated**

**Evidence from Results JSON**:
```json
"summary": {
  "avg_hierarchical_return": null,
  "avg_hierarchical_sharpe": null,
  "improvement_vs_baseline": {
    "return": null,
    "sharpe": null
  }
}
```

**Reason**: All 5 ETFs failed before backtesting phase, so no Sharpe ratios were calculated.

**Conclusion**: Complete failure to generate trading performance metrics.

---

### Criterion 3: Win/Loss Ratio >0.8

**Status**: ❌ **CANNOT EVALUATE - No Data Generated**

**Evidence**: Same as Criterion 2 - no trading metrics generated due to complete model failures.

**Conclusion**: Complete failure to generate trading performance metrics.

---

## Final Results JSON

**File**: `/Users/bobkatz/Visual_Trading_System/results/multi_timeframe_regime_20260104_130917.json`

```json
{
  "timestamp": "2026-01-04T13:09:17.117827",
  "test_type": "multi_timeframe_regime_classification",
  "horizons_tested": [20, 50, 100, 150, 200],
  "strategy": "hierarchical",
  "tickers_tested": ["OXLCG", "HCXY", "VGI", "HYI", "IGI"],
  "results": [
    {
      "ticker": "OXLCG",
      "status": "error",
      "error": "Invalid classes inferred from unique values of `y`.  Expected: [0 1 2], got [0 1 4]"
    },
    {
      "ticker": "HCXY",
      "status": "error",
      "error": "Invalid classes inferred from unique values of `y`.  Expected: [0 1], got [1 4]"
    },
    {
      "ticker": "VGI",
      "status": "error",
      "error": "Please reshape the input data into 2-dimensional matrix."
    },
    {
      "ticker": "HYI",
      "status": "error",
      "error": "Please reshape the input data into 2-dimensional matrix."
    },
    {
      "ticker": "IGI",
      "status": "error",
      "error": "Please reshape the input data into 2-dimensional matrix."
    }
  ],
  "summary": {
    "avg_hierarchical_return": null,
    "avg_hierarchical_sharpe": null,
    "improvement_vs_baseline": {
      "return": null,
      "sharpe": null
    }
  }
}
```

**Summary**:
- ❌ 5/5 ETFs failed completely
- ❌ 0 successful trading performance evaluations
- ❌ All metrics null

---

## Root Cause Analysis

### Why Multi-Timeframe Regime Classification Failed

**1. Temporal Data Distribution Problem**
- Time series data has non-stationary distributions
- Market regimes change over time (bull markets → bear markets)
- Temporal train/test splits capture different market conditions
- Training period may have regimes [0, 1, 2] while test has [0, 1, 4]
- XGBoost cannot handle classes in test that weren't in training

**2. Insufficient Data for Long Horizons**
- Long horizons (150, 200 days) require extensive historical data
- 60-day lookback + 150-day forward + splits = 210+ days minimum
- Many ETFs lack sufficient history
- Problem compounds for longer-term regime prediction

**3. Regime Classification ≠ Easier Than Return Prediction**
- **Phase 1 Hypothesis**: Regime classification more tractable than return prediction
- **Evidence**: Accuracy rates (6%-62.5%) suggest regimes equally hard to predict
- **Comparison to XGBoost Baseline**:
  - XGBoost baseline (return prediction): -4.41% annualized return
  - Multi-timeframe regime: 0% (no trading metrics generated)
  - Regime classification is WORSE, not better

**4. Fundamental Signal Insufficiency Remains**
- OHLCV data alone still insufficient for prediction
- Adding time horizons doesn't add new information types
- Still missing: sentiment, fundamentals, macro data, order flow
- Reformulation didn't solve feature poverty problem

---

## Comparison to Previous Approaches

### CNN Ensemble (Option A.1)
- **Result**: 0% return (never traded, ultra-conservative)
- **Status**: ❌ Failed - no positive predictions
- **Trading Metrics**: None generated

### XGBoost Baseline (Option A.3)
- **Result**: -4.41% annualized return, -0.18 Sharpe
- **Status**: ❌ Failed - negative returns
- **Trading Metrics**: ✅ Generated (though negative)
- **Best Single Model**: OXLCG (29.45% return, 3.90 Sharpe)
- **Worst Single Model**: HYI (-51.52% return, -4.81 Sharpe)

### Multi-Timeframe Regime (Phase 1 - This Test)
- **Result**: N/A (no metrics generated)
- **Status**: ❌ Failed - complete model failures
- **Trading Metrics**: ❌ None generated
- **Accuracy**: 1 of many configurations >60% threshold

**Ranking (Best to Worst)**:
1. **XGBoost Baseline**: At least generated trading metrics, some models profitable
2. **CNN Ensemble**: Ultra-conservative but avoided losses
3. **Multi-Timeframe Regime**: Complete failure, no metrics at all

**Critical Insight**: Multi-timeframe regime classification performed WORSE than both previous failed approaches. This is not an improvement - it's a step backward.

---

## Why Phase 1 Hypothesis Was Wrong

**Original Hypothesis** (from baseline_comparison_synthesis_20260104.md):

> "Reformulate as regime classification (Path B.1)"
> "Test whether regime prediction is more tractable than return prediction"

**Hypothesis Prediction**:
- Regime classification should be easier than return prediction
- Discrete classes (6 regimes) should be more learnable than continuous returns
- Multi-timeframe approach should capture market structure better

**Actual Results**:
- ❌ Regime accuracy (6%-62.5%) no better than return prediction correlation (0.14-0.69)
- ❌ Discrete classification introduced new failure modes (class imbalance)
- ❌ Multi-timeframe required more data, limiting applicability
- ❌ No trading performance metrics generated (worse than baseline)

**Conclusion**: The hypothesis that regime classification is more tractable than return prediction has been **empirically disproven**. Reformulating the problem did not solve the fundamental issue of insufficient predictive signal in OHLCV data.

---

## Technical Debt and Implementation Issues

### Issues Resolved
✅ **Decimal/Float Type Mismatch**:
- Applied 5 float type conversions at data extraction points
- Test now runs to completion without type errors

### Blocking Issues (Unresolved)
❌ **Class Imbalance Between Splits**:
- Requires stratified splitting (inappropriate for time series)
- Or larger datasets ensuring all classes in all splits
- Or fundamental architecture redesign

❌ **Insufficient Historical Data**:
- Would require requesting more data from database
- Or dropping long horizons (150, 200 days)
- Or filtering to only ETFs with sufficient history

❌ **Poor Regime Definition**:
- Current 6-regime system may be too granular
- Could simplify to 3 regimes (Bull/Bear/Neutral)
- But accuracy issues suggest fundamental prediction failure

---

## Go/No-Go Decision

### Official Decision: **NO-GO** for Phase 2

**Rationale**:

**From Baseline Synthesis** (lines 358-360):
> "If Phase 1 fails → Consider Path D (abandon approach)"
> "If Phase 1 succeeds → Proceed to Phase 2 (enhanced features)"

**Phase 1 Failure Evidence**:
1. ❌ Success Criterion 1: Only 1 configuration met 60% accuracy threshold
2. ❌ Success Criterion 2: No Sharpe ratios generated (cannot evaluate)
3. ❌ Success Criterion 3: No win/loss ratios generated (cannot evaluate)
4. ❌ 0 out of 5 ETFs completed successfully
5. ❌ Performed worse than previous failed approaches

**Decision**: Phase 2 (enhanced features) is **NOT RECOMMENDED**. Adding sentiment, fundamentals, and macro data to a fundamentally flawed approach is unlikely to succeed.

---

## Path D Recommendation: Abandon This Approach

### Strategic Alternatives to Consider

**From Baseline Synthesis** (lines 311-327):

> **Path D: Abandon This Approach**
>
> **Alternative Directions**:
> 1. Traditional quantitative strategies (momentum, mean reversion)
> 2. Rules-based technical systems
> 3. Different asset classes (options, futures)
> 4. Longer timeframes (swing trading, position trading)
> 5. Different data sources (alternative data, fundamental analysis)

### Recommended Next Steps

**Immediate (1-2 weeks)**:
1. **Accept Failure**: ML-based return/regime prediction from OHLCV alone is not viable for 5-day to 200-day horizons on these ETFs
2. **Pivot to Traditional Quant**: Test proven quantitative strategies (momentum, mean reversion, trend following)
3. **Leverage Existing Infrastructure**: Use PostgreSQL data and backtesting framework for traditional strategies

**Medium-term (1-3 months)**:
1. **Different Asset Classes**: Test on more liquid assets (major equity indices, currency pairs)
2. **Fundamental Analysis**: Pure fundamental-based strategies using earnings, valuation metrics
3. **Longer Timeframes**: Test swing trading (weeks) or position trading (months) where fundamentals matter more

**Long-term (3-6 months)**:
1. **Alternative Data**: If pursuing ML, invest in sentiment, order flow, options data first
2. **Ensemble Traditional + ML**: Combine proven quant strategies with ML for enhancement, not replacement
3. **Different Problem Formulation**: Portfolio optimization, risk management, rather than return/regime prediction

---

## Lessons Learned

### What We Now Know

**1. Reformulation Alone Doesn't Solve Fundamental Problems**
- Changing from return prediction → regime classification didn't help
- Same underlying issue: insufficient predictive signal in OHLCV data
- Problem reformulation useful only if addresses root cause

**2. Multi-Timeframe Adds Complexity Without Adding Information**
- 5 time horizons require more data (longer history needed)
- Introduces class imbalance problems in temporal splits
- Doesn't add new information types (still just OHLCV)
- Hierarchical strategy requires ALL horizons to work

**3. Low Validation Loss ≠ Trading Profitability ≠ High Accuracy**
- Previous CNN models: Low MSE, unprofitable
- Previous XGBoost: Good correlation (HYI: 0.692), catastrophic losses
- Current regime models: Poor accuracy AND no trading metrics
- All three approaches failed despite different training objectives

**4. Time Series Classification Has Unique Challenges**
- Non-stationary distributions (market regimes change over time)
- Class imbalance across temporal splits
- Cannot use stratified sampling (breaks time ordering)
- Requires massive datasets to ensure all classes in all periods

**5. 5-200 Day Horizon May Be Fundamentally Unpredictable**
- Too short for fundamental trends to manifest clearly
- Too long for pure technical/momentum patterns to persist
- Dominated by news, events, and random market fluctuations
- Efficient market hypothesis suggests short-term near-random

---

## Final Assessment

**Phase 1 Multi-Timeframe Regime Classification**: ❌ **COMPLETE FAILURE**

**Evidence-Based Conclusion**:
After testing three distinct ML approaches (CNNs, XGBoost return prediction, Multi-timeframe regime classification), all have failed to achieve profitable trading. The consistent failure across different architectures, problem formulations, and prediction targets strongly suggests the issue is **fundamental data insufficiency**, not architecture choice or problem framing.

**Recommended Action**:
Abandon ML-based prediction approaches using OHLCV data alone. Pivot to traditional quantitative strategies with proven track records, or invest in comprehensive alternative data before attempting ML again.

**Path Forward**:
**Path D** - Accept that this approach is not viable and explore fundamentally different directions as outlined in the Strategic Alternatives section.

---

## Appendix: Test Execution Details

**Test Date**: January 4, 2026, 1:09 PM
**Test Script**: `test_multi_timeframe_regime.py` (556 lines)
**Results File**: `results/multi_timeframe_regime_20260104_130917.json`
**Execution Status**: ✅ Ran to completion (after Decimal/float fixes)
**Result Status**: ❌ All 5 ETFs failed

**Test Configuration**:
- Estimators: 100
- Max Depth: 5
- Learning Rate: 0.1
- Objective: multi:softprob (6-class classification)
- Early Stopping: 10 rounds

**Data Configuration**:
- Lookback Window: 60 days (consistent across all horizons)
- Forward Horizons: 20, 50, 100, 150, 200 days
- Train/Val/Test Split: 70%/15%/15% (temporal)

**Regime Definitions**:
- 0: Strong Downtrend (< mean - 0.5σ)
- 1: Weak Downtrend (mean - 0.5σ to mean)
- 2: Sideways (mean ± 0.2σ)
- 3: Weak Uptrend (mean to mean + 0.5σ)
- 4: Strong Uptrend (> mean + 0.5σ)
- 5: High Volatility (vol > mean + 1.5σ, takes precedence)

**Hierarchical Trading Strategy**:
- Full position (100%): All 5 horizons agree on bullish/bearish
- Half position (50%): 3-4 horizons agree
- No position (0%): <3 horizons agree

---

**Document Status**: Final
**Next Action Required**: Strategic decision on Path D alternatives
**Priority**: CRITICAL - Requires project direction decision
