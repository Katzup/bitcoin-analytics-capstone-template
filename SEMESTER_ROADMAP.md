# Semester Roadmap: Post-Midterm Implementation Plan

**Document Date**: 2026-02-17  
**Status**: Evidence-driven plan based on midterm EDA findings  
**Goal**: Move from 41.94% → 57-73% RW percentile through targeted improvements

---

## Executive Summary

Midterm EDA revealed three critical findings:
1. **"Complexity penalty"** - 296K param CNN matched by neutral baseline
2. **"Daily noise"** - 1-day direction signals dominated by noise
3. **"Regime dependency"** - Signal efficacy varies by market conditions

**Strategic pivot**: Stop adding model complexity. Start gating decisions by regime and extending signal horizon.

---

## Phase 1: Regime-Gated Allocator (Weeks 1-2)

### Objective
Implement 2-3 regime detectors and create a gated policy allocator that chooses between buy policies based on regime state.

### Implementation

#### 1.1 Volatility Regime (Most Stable)
```python
# Realized volatility (20-day rolling)
vol_20d = returns.rolling(20).std() * np.sqrt(252)

# Quantile-based regimes (causal - uses only past data)
vol_regime = pd.qcut(vol_20d, q=3, labels=['low', 'medium', 'high'])

# Or threshold-based
vol_regime = np.where(vol_20d > vol_20d.quantile(0.7), 'high',
             np.where(vol_20d < vol_20d.quantile(0.3), 'low', 'medium'))
```

#### 1.2 Trend Regime (Causal Only)
```python
# 200-day MA slope (causal)
ma_200 = price.rolling(200).mean()
ma_slope = ma_200.diff(20) / ma_200.shift(20)  # 20-day slope

# Distance from MA z-score (rolling, causal)
price_to_ma = (price - ma_200) / price.rolling(200).std()

# Trend strength (ADX-like, simplified)
trend_regime = np.where(price_to_ma > 1, 'strong_up',
               np.where(price_to_ma < -1, 'strong_down', 'neutral'))
```

#### 1.3 HMM Regime (2-State)
```python
from hmmlearn.hmm import GaussianHMM

# Fit on training data only
hmm = GaussianHMM(n_components=2, covariance_type='full', random_state=42)
hmm.fit(returns.reshape(-1, 1))

# Predict regimes (causal - fit once, predict rolling)
regimes = hmm.predict(returns.reshape(-1, 1))
```

#### 1.4 Joint Regime: Trend × Vol (4 States)
```python
# Create 4-state regime
regime_joint = np.where((trend == 'up') & (vol == 'low'), 'up_low',
               np.where((trend == 'up') & (vol == 'high'), 'up_high',
               np.where((trend == 'down') & (vol == 'low'), 'down_low',
                                                      'down_high')))
```

### Policy Allocator Design

```python
def regime_gated_allocator(prob_up, regime, base_allocation=100):
    """
    Choose policy based on regime, not daily prediction.
    
    Policies:
    - DCA: base_allocation (steady buy)
    - Buy-more: base_allocation * 1.5 (tilt up in favorable regime)
    - Buy-less: base_allocation * 0.5 (tilt down in unfavorable regime)
    """
    policy_map = {
        'up_low': 'buy_more',      # Best regime: strong trend, low vol
        'up_high': 'dca',          # Good trend but noisy: steady
        'down_low': 'dca',         # Weak but calm: steady accumulation
        'down_high': 'buy_less'    # Worst regime: reduce exposure
    }
    
    policy = policy_map.get(regime, 'dca')
    
    multipliers = {
        'buy_more': 1.5,
        'dca': 1.0,
        'buy_less': 0.5
    }
    
    return base_allocation * multipliers[policy]
```

### Ablation Plan
| Variant | Description | Expected RW |
|---------|-------------|-------------|
| Baseline | Neutral prob_up = 0.5 | 41.94% |
| Vol-only | 3 vol regimes | 43-47% |
| Trend-only | 3 trend regimes | 44-48% |
| HMM | 2-state HMM | 45-50% |
| Joint (trend × vol) | 4 states | 47-53% |

### Deliverables
- [ ] `regime_detector.py` - Vol, trend, HMM implementations
- [ ] `regime_gated_allocator.py` - Policy allocator
- [ ] Ablation results table (5 variants)
- [ ] Section in final report: "Regime-Gated Allocator"

---

## Phase 2: Signal Horizon Upgrade (Weeks 2-3)

### Objective
Reduce daily noise by shifting from 1-day targets to 5-10 day horizons.

### Implementation

#### 2.1 Multi-Day Forward Return Target
```python
# 5-day forward return (instead of 1-day)
forward_return_5d = (price.shift(-5) - price) / price

# Binary label: up if > 0
label_5d = (forward_return_5d > 0).astype(int)
```

#### 2.2 Risk-Adjusted Target
```python
# Sharpe-like: return / vol
forward_return_5d = (price.shift(-5) - price) / price
realized_vol_20d = returns.rolling(20).std() * np.sqrt(252)

risk_adjusted_signal = forward_return_5d / realized_vol_20d
```

#### 2.3 Drawdown Risk Signal
```python
# Running max (causal)
running_max = price.expanding().max()
drawdown = (price - running_max) / running_max

# Elevated drawdown risk
dd_risk = drawdown < -0.15  # 15% drawdown threshold
```

### Weekly Rebalancing Variant
```python
# Even with daily signals, only trade weekly
if day_of_week == 'Monday':
    # Update allocation based on signal
    allocation = compute_allocation(signal)
else:
    # Hold previous allocation
    allocation = previous_allocation
```

### Ablation Plan
| Variant | Description | Expected RW |
|---------|-------------|-------------|
| 1-day (baseline) | Current | 41.94% |
| 5-day horizon | Forward return target | 44-50% |
| 10-day horizon | Forward return target | 45-52% |
| Weekly rebalance | Same signal, trade weekly | 46-54% |
| 5-day + weekly | Combined | 48-57% |

---

## Phase 3: Robustness & Validation (Weeks 3-4)

### Objective
Add structure to prevent overfitting and enable credible reporting.

### Implementation

#### 3.1 Walk-Forward with Multiple Folds
```python
# 5-fold walk-forward
def walk_forward_splits(data, n_splits=5):
    """Generate purged walk-forward splits."""
    n = len(data)
    fold_size = n // n_splits
    
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end + 30  # 30-day purge
        test_end = min(test_start + fold_size, n)
        
        yield slice(0, train_end), slice(test_start, test_end)
```

#### 3.2 Purged Splits Around Boundaries
```python
def purged_train_test_split(X, y, test_size=0.2, purge_gap=30):
    """Split with gap to prevent leakage."""
    n = len(X)
    test_start = int(n * (1 - test_size))
    
    # Purge gap: don't use test_start - purge_gap to test_start
    train_idx = slice(0, test_start - purge_gap)
    test_idx = slice(test_start, n)
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

#### 3.3 Report by Regime + Subperiod
```python
# Performance by regime
for regime in ['up_low', 'up_high', 'down_low', 'down_high']:
    mask = regimes == regime
    regime_return = returns[mask].mean()
    regime_sharpe = returns[mask].mean() / returns[mask].std()
    
# Performance by subperiod (year, quarter)
for year in range(2016, 2025):
    mask = df.index.year == year
    annual_return = returns[mask].sum()
```

### Deliverables
- [ ] `robust_evaluation.py` - Walk-forward + purged splits
- [ ] Performance by regime table
- [ ] Performance by subperiod (year/quarter) table
- [ ] Final report section: "Robustness Analysis"

---

## Phase 4: Optional Point & Figure (Week 4+)

### Objective
Test P&F as feature extraction (NOT CNN input).

### Implementation

#### 4.1 P&F Features (Simple, Causal)
```python
def calculate_pnf_features(prices, box_size=0.02, reversal=3):
    """
    Extract causal features from Point & Figure chart.
    
    Returns:
    - trend_direction: 'up'/'down' column
    - boxes_in_column: count
    - reversal_count: number of reversals
    - time_in_column: days in current column
    """
    # Simple P&F logic (causal)
    trend = []
    boxes = []
    
    current_trend = 'up'
    current_boxes = 0
    
    for i in range(1, len(prices)):
        change = (prices[i] - prices[i-1]) / prices[i-1]
        
        if current_trend == 'up':
            if change >= box_size:
                current_boxes += 1
            elif change <= -box_size * reversal:
                trend.append(current_trend)
                boxes.append(current_boxes)
                current_trend = 'down'
                current_boxes = 1
        else:  # down
            if change <= -box_size:
                current_boxes += 1
            elif change >= box_size * reversal:
                trend.append(current_trend)
                boxes.append(current_boxes)
                current_trend = 'up'
                current_boxes = 1
    
    return {
        'trend_direction': current_trend,
        'boxes_in_column': current_boxes,
        'reversal_count': len(trend),
    }
```

### Ablation Plan
| Variant | Description | Decision Rule |
|---------|-------------|---------------|
| Without P&F | Baseline | Keep if P&F doesn't add value |
| With P&F | Add P&F features | Only if ablation shows +2% RW or better |

**Rule**: If P&F doesn't show lift in first week, drop it immediately. Don't over-invest.

---

## Timeline Summary

| Week | Focus | Key Deliverable | Target RW |
|------|-------|-----------------|-----------|
| 1 | Vol + Trend regimes | Regime detector + ablation | 44-48% |
| 2 | HMM + Joint regimes | Joint regime allocator | 47-53% |
| 3 | 5d/10d horizon + weekly | Multi-day target ablation | 48-57% |
| 4 | Walk-forward + robustness | Robust evaluation framework | 50-60% |
| 4+ | P&F (optional) | P&F feature ablation | TBD |

**Conservative target**: 41.94% → 57% RW percentile  
**Optimistic target**: 41.94% → 73% RW percentile

---

## Success Metrics

### Primary
- [ ] RW SPD percentile > 57% (minimum viable)
- [ ] RW SPD percentile > 60% (competitive)

### Secondary
- [ ] Win rate > 60% (vs DCA)
- [ ] Positive alpha in >50% of walk-forward folds
- [ ] Consistent performance across regimes

### Validation
- [ ] Pass all 3 causality tests
- [ ] Deterministic (seed=42, reproducible)
- [ ] Constraint compliant (w_i ≥ 1e-5, Σw_i = 1.0)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting to regimes | Purged splits, walk-forward validation |
| Regime instability | Multiple regime detectors, ensemble |
| Data leakage | Strict causal implementation, 3 validation tests |
| Time crunch | Prioritize Phase 1-2, Phase 4 is optional |

---

## Files to Create/Update

### New Files
- `regime_detector.py` - Vol, trend, HMM regime detection
- `regime_gated_allocator.py` - Policy allocator
- `robust_evaluation.py` - Walk-forward + purged splits
- `pnf_features.py` - Point & Figure feature extraction (optional)

### Updated Files
- `btc_accumulation_model.ipynb` - Add regime-gated variant
- `EXECUTIVE_SUMMARY.md` - Update with regime results
- `LESSONS_LEARNED.md` - Document regime findings
- Final report - Add "Regime-Gated Allocator" section

---

## Key Principles

1. **Small action space** - 3 policies max (DCA, buy-more, buy-less)
2. **Causal only** - All features use expanding/rolling windows
3. **Test early** - Ablation #0: "Is regime better than neutral?"
4. **Drop fast** - If P&F doesn't add value in 1 week, drop it
5. **Document everything** - Each ablation gets a row in the results table

---

**Quote to guide implementation**:  
*"The best improvements come from better problem framing, not better models."*
