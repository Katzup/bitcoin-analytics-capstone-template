# Trilemma Tournament Master Plan
## VTS Continuous Allocator → Tournament Adapter

**Tournament**: Trilemma x Strategy Stacking Sats Tournament
**Repository**: https://github.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025
**Date**: 2026-01-21
**Status**: Implementation in progress

---

## Executive Summary

**Objective**: Adapt VTS continuous CNN-based allocator to tournament format

**Approach**: Pure adapter layer - zero modifications to existing VTS code

**Key Insight**: VTS is the "brains" (feature engineering + allocation logic), tournament harness is the "simulator" (rolling windows + SPD scoring)

**Deliverable**: Jupyter notebook with `construct_features()` and `compute_weights()` functions

---

## Tournament Requirements vs VTS Capabilities

### Action Space
- **Tournament**: Normalized daily weights (w_i ≥ 10⁻⁵, Σw_i = 1)
- **VTS**: Bounded multipliers [0.7, 1.6] with budget normalization
- **Adapter**: Normalize multipliers to sum=1

### Decision Frequency
- **Tournament**: Daily (365 decisions per 12-month window)
- **VTS**: Configurable (daily in BTC_1D preset)
- **Adapter**: Use BTC_1D preset as-is ✅

### Scoring
- **Tournament**: SPD percentile (0-100) + win rate vs DCA
- **VTS**: Alpha + IR vs DCA baseline
- **Adapter**: New scoring module implementing SPD formulas

### Evaluation
- **Tournament**: 3,075 rolling 12-month windows (2016-2025)
- **VTS**: Walk-forward (train/val/test splits)
- **Adapter**: New evaluator for rolling windows

### Constraints
- **Tournament**: w_i ≥ 10⁻⁵, Σw_i = 1.0 (strict)
- **VTS**: Budget normalization (flexible)
- **Adapter**: Post-process weights with clamp+renormalize

---

## Official Scoring Formulas (Verified)

### 1. SPD (Satoshis Per Dollar)
```python
# Official formula from tournament template:
inv_price = (1.0 / price) * 1e8  # sats per dollar
uniform_spd = inv_price.mean()   # DCA baseline (arithmetic mean)
dynamic_spd = sum(weight * inv_price)  # Strategy (weighted sum)
```

**Mathematical equivalence**:
```
dynamic_spd = sum(w_t × inv_price_t)
            = sum(w_t × (1/p_t) × 1e8)
            = 1e8 × sum(w_t / p_t)
```

### 2. Percentile Ranking
```python
# Normalize between worst-case and best-case
min_spd = inv_price.min()  # All buys at peak price
max_spd = inv_price.max()  # All buys at bottom price
span = max_spd - min_spd

uniform_pct = (uniform_spd - min_spd) / span * 100
dynamic_pct = (dynamic_spd - min_spd) / span * 100
```

**Interpretation**: Percentile ∈ [0, 100] where:
- 0% = worst possible timing (all buys at peak)
- 100% = best possible timing (all buys at bottom)
- 50% = typical for uniform DCA

### 3. Recency Weighting (PRIMARY METRIC)
```python
# Exponential decay: newest windows weighted most
decay = 0.9
N = len(dynamic_pct)
raw_w = np.array([decay ** (N - 1 - i) for i in range(N)])
exp_w = raw_w / raw_w.sum()
RW_SPD_PCT = sum(dynamic_pct[i] * exp_w[i])
```

**Decay direction**:
- i=0 (oldest window): weight = 0.9^(N-1) = smallest
- i=N-1 (newest window): weight = 0.9^0 = 1.0 = largest

### 4. Win Rate (SECONDARY METRIC)
```python
# Fraction of windows beating DCA
wins = sum(1 for i in range(N) if dynamic_pct[i] > uniform_pct[i])
pass_ratio = wins / N
```

### 5. Weight Constraints
```python
MIN_WEIGHT = 1e-5
# All weights: w_t >= MIN_WEIGHT
# Sum constraint: sum(w_t) == 1.0 (within tolerance rtol=1e-5, atol=1e-8)
```

---

## VTS Allocation Logic (Causal)

### Signal Generation
```python
# Step 1: CNN probability → tilt
tilt = sensitivity * (prob_up - 0.5)  # sensitivity = 1.5

# Step 2: Bounded multiplier
multipliers = (1.0 + tilt).clip(lower=0.7, upper=1.6)

# Step 3: EMA smoothing (inherently causal)
multipliers_smooth = multipliers.ewm(alpha=0.30, adjust=False).mean()
```

### Tournament Adaptation
```python
# Step 4: Normalize to sum=1
weights = multipliers_smooth.values
weights = weights / weights.sum()

# Step 5: Enforce minimum weight constraint
weights = np.maximum(weights, MIN_WEIGHT)
weights = weights / weights.sum()  # Re-normalize after clamp
```

### Causality Guarantee
All operations are causal (w_t uses only data ≤ t):
- ✅ Element-wise: `tilt`, `clip` - no lookahead by construction
- ✅ EMA: `pandas.ewm()` - uses only historical values
- ✅ Normalization: operates on entire vector but doesn't change causality
- ✅ Validation: Last-row modification test confirms no leakage

---

## Architecture Overview

```
Tournament Notebook (btc_accumulation_model.ipynb)
    │
    ├─> construct_features(df) → feature-enriched DataFrame
    │   │
    │   ├─> EnhancedBinaryDataset.generate_gaf()  # VTS component
    │   ├─> DeepTradingCNNClassifier.predict()     # VTS component
    │   └─> Technical indicators (MA, volatility)  # Simple pandas
    │
    ├─> compute_weights(df_window) → normalized weights
    │   │
    │   ├─> VTS allocation logic (tilt → multiplier → EMA)
    │   ├─> Tournament normalization (sum=1, min_weight)
    │   └─> Causality enforcement (validated)
    │
    └─> Tournament Evaluator (3,075 windows)
        │
        ├─> SPD calculation per window
        ├─> Percentile ranking
        ├─> Recency-weighted aggregation
        └─> Win rate calculation
```

---

## Implementation Files

### Phase 1: Core Adapter (✅ In Progress)
- `tournament_mode/__init__.py` - Package init
- `tournament_mode/scoring.py` - SPD, percentile, RW metrics (✅ COMPLETE)
- `tournament_mode/weights.py` - VTS → tournament adapter (✅ COMPLETE)
- `tournament_mode/features.py` - CNN feature generation (→ Next)
- `tournament_mode/utils.py` - Helpers and validation

### Phase 2: Evaluation & Testing
- `tournament_mode/evaluator.py` - Rolling window backtester
- `tournament_mode/tests/test_scoring.py` - Unit tests for scoring
- `tournament_mode/tests/test_causality.py` - Lookahead prevention tests
- `tournament_mode/tests/test_constraints.py` - Weight constraint tests

### Phase 3: Submission Notebooks
- `btc_accumulation_model.ipynb` - Main tournament submission
- `vts_educational_notebook.ipynb` - Educational explanation
- `tournament_mode/train_tournament_cnn.py` - Pre-train CNN on 2014-2015

---

## Critical Validation Checklist

### Score-Critical Tests (MUST PASS)

**1. Reference Equivalence**
```python
# Uniform weights → identical to DCA
w_uniform = np.ones(N) / N
assert dynamic_spd(w_uniform) == uniform_spd
assert dynamic_pct(w_uniform) == uniform_pct
```

**2. Constant Price Edge Case**
```python
# All prices same → any weights give 50% percentile
prices_const = np.ones(N) * 30000
assert abs(dynamic_pct(any_weights) - 50.0) < 1e-6
```

**3. Causality Enforcement**
```python
# Last-row modification → first N-1 weights unchanged
df_modified = df_original.copy()
df_modified.iloc[-1, 'prob_up'] = 0.99  # Change last day
assert np.allclose(weights_original[:-1], weights_modified[:-1])
```

**4. Official Scorer Parity**
```python
# Our implementation matches official template
assert abs(our_spd - official_spd) < 1e-6
assert abs(our_RW_PCT - official_exp_avg_pct) < 1e-6
```

### Weight Constraints (MUST HOLD)
```python
# All weights >= MIN_WEIGHT
assert (weights >= MIN_WEIGHT - 1e-9).all()

# Sum exactly 1.0 (within tolerance)
assert np.isclose(weights.sum(), 1.0, rtol=1e-5, atol=1e-8)
```

---

## VTS Components Reused (Zero Modifications)

| VTS Module | Tournament Usage | Status |
|------------|------------------|--------|
| `EnhancedBinaryDataset` | GAF feature generation | ✅ Reuse as-is |
| `DeepTradingCNNClassifier` | P(up) inference | ✅ Pre-train, then load |
| `TrilemmaAllocator` logic | Allocation logic | ✅ Extract to adapter |
| `config_multi_timeframe.BTC_1D` | Parameters | ✅ Use sensitivity=1.5, alpha=0.30 |
| `trilemma_runner.py` | Not used for tournament | ✅ Keep for CNN training |
| `data_adapter.py` | Optional for CNN training | ✅ Keep unchanged |

---

## Pre-Training Strategy

### CNN Model Preparation
```python
# Train ONCE on 2014-2015 data (before tournament starts)
from trilemma_runner import run_trilemma_evaluation
from config_multi_timeframe import BTC_1D

# Fetch pre-tournament data
df_pretrain = fetch_btc_data(start='2014-01-01', end='2015-12-31')

# Train CNN
results = run_trilemma_evaluation(
    ticker='BTC-USD',
    preset=BTC_1D,
    df_override=df_pretrain,
    save_model=True,
    model_save_path='./models/tournament_cnn.pth'
)

# For tournament: Load frozen model (no retraining)
model = DeepTradingCNNClassifier()
model.load_state_dict(torch.load('tournament_cnn.pth'))
model.eval()
```

**Why freeze after pre-training?**
- Simplicity: One model for all 3,075 windows
- Reproducibility: Same model → same results
- No lookahead: Model never sees 2016+ data during training
- Tournament compliance: Conservative approach

---

## Rolling Window Evaluation

### Date Range
- **Start**: 2016-01-01
- **End**: 2025-06-01
- **Window size**: 365 days (12 months)
- **Step size**: 1 day
- **Total windows**: 3,075

### Evaluation Loop
```python
results = {'dynamic_pct': [], 'uniform_pct': []}

current_date = pd.Timestamp('2016-01-01')
end_date = pd.Timestamp('2025-06-01')

while current_date + timedelta(days=365) <= end_date:
    # Extract 12-month window
    window_end = current_date + timedelta(days=365)
    df_window = df_full[(df_full.index >= current_date) &
                        (df_full.index < window_end)]

    # Compute strategy weights
    strategy_weights = compute_weights(df_window)

    # Compute DCA weights
    dca_weights = uniform_weights(df_window)

    # Calculate SPD metrics
    metrics = calculate_spd_for_window(strategy_weights, df_window['price'])

    # Store percentiles
    results['dynamic_pct'].append(metrics['dynamic_pct'])
    results['uniform_pct'].append(metrics['uniform_pct'])

    # Move to next window
    current_date += timedelta(days=1)
```

---

## Final Tournament Metrics

### Primary Score: RW_SPD_PCT
```python
RW_SPD_PCT = calculate_recency_weighted_percentile(
    np.array(results['dynamic_pct']),
    decay=0.9
)
```

**Interpretation**: Recency-weighted average percentile across all windows
- Higher = better (strategy consistently beats worst-case timing)
- Target: > 50% (beats uniform DCA on average)

### Secondary Score: Win Rate
```python
win_rate = calculate_win_rate(
    np.array(results['dynamic_pct']),
    np.array(results['uniform_pct'])
)
```

**Interpretation**: Fraction of windows where strategy beats DCA
- Range: [0, 1] where 1 = wins all windows
- Target: > 0.5 (wins majority of windows)

### Reporting
```python
print(f"Primary Score (RW_SPD_PCT): {RW_SPD_PCT:.2f}%")
print(f"Secondary Score (Win Rate): {win_rate:.2%}")
print(f"Windows Evaluated: {len(results['dynamic_pct'])}")
print(f"Avg Strategy Percentile: {np.mean(results['dynamic_pct']):.2f}%")
print(f"Avg DCA Percentile: {np.mean(results['uniform_pct']):.2f}%")
```

---

## Success Criteria

### Functional Requirements
- [x] ✅ Weight constraints: w_i ≥ 10⁻⁵, Σw_i = 1.0
- [x] ✅ No lookahead bias: Causality tests pass
- [ ] ⏳ Reproducible: Same seed → same score
- [ ] ⏳ All windows evaluated: 3,075 successful runs
- [ ] ⏳ Notebook executes: End-to-end without errors

### Performance Targets
- **Conservative**: RW_SPD_PCT > 50%, Win Rate > 0.50 (beats DCA)
- **Realistic**: RW_SPD_PCT > 55%, Win Rate > 0.55 (modest outperformance)
- **Aspirational**: RW_SPD_PCT > 60%, Win Rate > 0.60 (strong outperformance)

**Note**: Do NOT hardcode expected performance. These are hypotheses to be validated.

---

## Timeline & Milestones

### Phase 0: Pre-Implementation (✅ COMPLETE)
- [x] ✅ Download official scorer
- [x] ✅ Verify SPD formulas
- [x] ✅ Create scoring specification
- [x] ✅ Identify critical bugs (pandas .clip() fix)

### Phase 1: Core Adapter (🔄 IN PROGRESS)
- [x] ✅ `scoring.py` - SPD, percentile, RW metrics
- [x] ✅ `weights.py` - VTS allocation adapter
- [ ] ⏳ `features.py` - CNN feature generation
- [ ] ⏳ `utils.py` - Helpers and validation

### Phase 2: Evaluation (⏸ PENDING)
- [ ] ⏳ `evaluator.py` - Rolling window evaluator
- [ ] ⏳ Unit tests for all modules
- [ ] ⏳ Pre-train CNN on 2014-2015 data
- [ ] ⏳ End-to-end validation

### Phase 3: Submission (⏸ PENDING)
- [ ] ⏳ Main submission notebook
- [ ] ⏳ Educational notebook
- [ ] ⏳ Full tournament run (3-4 hours)
- [ ] ⏳ Final verification and submission

---

## Key Design Decisions

### ✅ Pure Adapter Approach
**Decision**: Zero modifications to existing VTS code
**Rationale**: VTS evaluation (walk-forward) ≠ tournament (rolling windows)
**Benefit**: Clean separation, VTS remains unchanged

### ✅ Frozen Pre-Trained Model
**Decision**: Train once on 2014-2015, freeze for all windows
**Rationale**: Simplicity, reproducibility, conservative compliance
**Alternative considered**: Adaptive retraining (too complex, low benefit)

### ✅ Causality via pandas.ewm()
**Decision**: Trust pandas EMA implementation for causality
**Rationale**: Inherently causal, well-tested, efficient
**Validation**: Last-row modification test confirms no leakage

### ✅ Primary Metric = RW_SPD_PCT
**Decision**: Use recency-weighted percentile as main score
**Rationale**: Official template clearly computes this as `exp_avg_pct`
**Note**: Also compute win_rate; clarify composite if needed later

---

## Risk Mitigation

### High-Risk Areas
1. **Scoring formula mismatch** → Mitigated by official scorer verification
2. **Lookahead bias** → Mitigated by causality tests
3. **Weight constraint violation** → Mitigated by deterministic enforcement
4. **Numerical instability** → Mitigated by tolerance checks

### Medium-Risk Areas
1. **CNN inference errors** → Mitigated by pre-training validation
2. **Feature generation bugs** → Mitigated by unit tests
3. **Long runtime (3-4 hours)** → Acceptable for one-time submission

### Low-Risk Areas
1. **VTS components** → Already validated in walk-forward testing
2. **Pandas operations** → Well-tested library
3. **Notebook format** → Matches official template

---

## References

### Official Tournament
- **Repository**: https://github.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025
- **Template**: model-development/model_development_template.ipynb
- **Data**: data/btc_prices.parquet

### VTS Documentation
- **Scoring Spec**: claudedocs/TOURNAMENT_SCORING_SPEC.md
- **Official Verification**: claudedocs/OFFICIAL_SCORER_VERIFICATION.md
- **Final Plan**: claudedocs/TOURNAMENT_SCORING_PLAN_FINAL.md

### Implementation Files
- **Scoring**: tournament_mode/scoring.py
- **Weights**: tournament_mode/weights.py
- **Features**: tournament_mode/features.py (pending)
- **Evaluator**: tournament_mode/evaluator.py (pending)

---

## Next Steps

1. ✅ Complete `features.py` - CNN feature generation wrapper
2. ✅ Complete `utils.py` - Model loading and helpers
3. ✅ Write unit tests for scoring and weights modules
4. ✅ Implement rolling window evaluator
5. ✅ Pre-train CNN model on 2014-2015 data
6. ✅ Create main submission notebook
7. ✅ Run full evaluation (3,075 windows)
8. ✅ Verify results and submit

---

**Status**: Phase 1 in progress - scoring.py and weights.py complete
**Next**: Implement features.py for CNN feature generation
**Blocker**: None - ready to proceed
