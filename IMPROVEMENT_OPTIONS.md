# Performance Improvement Options: Image-Based Approaches

## Current Baseline
- **GAF (Gramian Angular Field)**: 41.43% RW percentile
- **Neutral (random)**: 41.94% RW percentile
- **Gap**: 0.51 pp (negligible)

## Why GAF Failed
1. **Time horizon mismatch**: 90-day lookback vs 6-18 month BTC cycles
2. **Daily noise**: Signal-to-noise ratio too low for daily allocation
3. **Spatial vs temporal**: CNNs assume translation invariance, time series need sequence modeling
4. **Training set**: 2014-2015 data doesn't capture later market regimes

---

## Alternative Imaging Techniques

### 1. MTF (Markov Transition Field)
**What it is**: Represents time series as transition probabilities between quantile bins

**How it differs from GAF**:
- GAF: Angular encoding preserves temporal correlation via geometry
- MTF: Transition probabilities capture state-to-state dynamics

**Implementation**:
```python
from pyts.image import MarkovTransitionField

mtf = MarkovTransitionField(image_size=90, n_bins=8)
mtf_image = mtf.transform(price_window.reshape(1, -1))[0]
```

**Expected gain**: +2-5 pp
- Captures regime transitions better than GAF
- Still limited by 90-day window
- Doesn't solve fundamental noise problem

**Effort**: 2-3 hours to implement and test

---

### 2. RP (Recurrence Plot)
**What it is**: Visualizes recurrence patterns in phase space reconstruction

**How it differs**:
- Detects periodic patterns and regime changes
- Better for chaotic/non-linear dynamics
- Good for detecting cycles

**Implementation**:
```python
from pyts.image import RecurrencePlot

rp = RecurrencePlot(dimension=3, time_delay=1)
rp_image = rp.transform(price_window.reshape(1, -1))[0]
```

**Expected gain**: +1-4 pp
- Good for detecting BTC's cyclical nature
- But 90-day window still too short for full cycles
- May capture sub-cycles (30-60 day patterns)

**Effort**: 2-3 hours

---

### 3. Multi-Field Ensemble (GAF + MTF + RP)
**Approach**: Train separate CNNs on different representations, ensemble predictions

**Implementation**:
```python
# Generate all three representations
gaf_image = gasf.transform(price_window)
mtf_image = mtf.transform(price_window)
rp_image = rp.transform(price_window)

# Train 3 separate CNNs
prob_gaf = cnn_gaf.predict(gaf_image)
prob_mtf = cnn_mtf.predict(mtf_image)
prob_rp = cnn_rp.predict(rp_image)

# Ensemble (average or weighted)
prob_up = (prob_gaf + prob_mtf + prob_rp) / 3
```

**Expected gain**: +5-10 pp (if patterns are complementary)
- Diversifies signal sources
- Reduces overfitting to single representation
- But 3x training time, 3x model artifacts

**Effort**: 1-2 days (train 3 models, implement ensemble)

---

### 4. Time-Frequency Representations

#### Spectrograms (STFT)
**What it is**: Short-Time Fourier Transform shows frequency content over time

```python
from scipy.signal import spectrogram
f, t, Sxx = spectrogram(price_window, fs=1.0, nperseg=30)
```

**Expected gain**: +2-5 pp
- Captures cyclical patterns (weekly, monthly oscillations)
- Good for volatility regime detection
- But BTC has non-stationary frequency content

#### Wavelet Transforms
**What it is**: Multi-resolution time-frequency analysis

```python
import pywt
coeffs = pywt.wavedec(price_window, 'db4', level=5)
# Reconstruct as 2D image
```

**Expected gain**: +3-6 pp
- Better than STFT for non-stationary signals
- Captures different time scales (trend, cycles, noise)
- Natural fit for multi-scale BTC dynamics

**Effort**: 3-5 hours per approach

---

## Fundamental Improvements (Bigger Impact)

### 5. Longer Lookback with Downsampling ⭐
**Problem**: 90-day window misses macro trends (bull markets are 6-18 months)

**Solution**: Use 180-365 days but downsample to fit 90x90 image

```python
# 365-day window, sample every 4 days → 91 points
long_window = prices[t-365:t:4]
gaf_image = gasf.transform(long_window)
```

**Expected gain**: +5-10 pp
- Captures full bull/bear cycles
- Maintains manageable image size
- Better regime detection

**Risk**: Need more training data (2014-2015 may not be enough)

**Effort**: 1 day to retrain

---

### 6. Weekly Granularity ⭐⭐ (Highest ROI)
**Problem**: Daily prices are too noisy (signal-to-noise ratio low)

**Solution**: Use weekly bars instead of daily

```python
# Resample to weekly
df_weekly = df.resample('W').agg({
    'PriceUSD_coinmetrics': 'last',
    'volume': 'sum'
})

# 90 weeks ≈ 21 months (captures full cycle!)
```

**Expected gain**: +10-20 pp (based on VTS weekly IR=1.59)
- Dramatically improves signal-to-noise
- 90-week window = 21 months (perfect for BTC cycles)
- VTS originally designed for weekly (performed better)

**Catch**: Tournament expects daily weights, so:
- Train on weekly images
- Generate weekly predictions
- Interpolate to daily weights (hold constant for 7 days)

**Effort**: 1-2 days (retrain + interpolation logic)

---

### 7. Multi-Channel Features ⭐
**Problem**: Only using price, ignoring other signals

**Solution**: Multi-channel CNN input

```python
# Channel 1: Normalized price
channel1 = (prices - prices.mean()) / prices.std()

# Channel 2: Returns (momentum signal)
channel2 = np.diff(np.log(prices), prepend=np.nan)

# Channel 3: Volatility (rolling std)
channel3 = pd.Series(prices).rolling(20).std().values

# Channel 4: Volume (if available)
channel4 = volumes

# Stack into 4-channel image
gaf_price = gasf.transform(channel1)
gaf_returns = gasf.transform(channel2)
gaf_vol = gasf.transform(channel3)
gaf_volume = gasf.transform(channel4)

multi_channel = np.stack([gaf_price, gaf_returns, gaf_vol, gaf_volume])
```

**Expected gain**: +5-12 pp
- Richer feature set
- Captures momentum, volatility, volume dynamics
- But need to retrain CNN for 4-channel input

**Effort**: 1-2 days

---

### 8. Regime Detection → Conditional Allocation ⭐⭐⭐ (Best Long-term)
**Approach**: Two-stage model

**Stage 1: Regime Classification** (longer lookback, weekly)
```python
# 365-day window, weekly sampling
regime_image = generate_gaf(prices_365d_weekly)
regime = regime_cnn.predict(regime_image)  # bull/bear/sideways
```

**Stage 2: Regime-Conditional Allocation**
```python
if regime == 'bull':
    # Aggressive allocation (high sensitivity)
    allocation = aggressive_allocator(prob_up)
elif regime == 'bear':
    # Defensive allocation (low sensitivity or constant)
    allocation = defensive_allocator(prob_up)
else:  # sideways
    # Neutral allocation
    allocation = neutral_allocator(prob_up)
```

**Expected gain**: +15-25 pp
- Different strategies for different market conditions
- Bull markets: ride the trend
- Bear markets: preserve capital (low allocation)
- Sideways: neutral or momentum-based

**Effort**: 3-5 days (train regime classifier, design conditional strategies)

---

## Better Architectures (Beyond CNNs)

### 9. Replace CNN with RNN/LSTM
**Why**: Time series have temporal ordering, not spatial patterns

```python
# Instead of CNN on GAF image
model = nn.LSTM(input_size=1, hidden_size=64, num_layers=2)
output = model(price_window.unsqueeze(-1))
```

**Expected gain**: +8-15 pp
- Better suited for sequential data
- Captures long-term dependencies
- No need for image transformation

**Catch**: Requires different architecture, more training data

**Effort**: 2-3 days

---

### 10. Transformer Architecture
**Why**: Attention mechanism captures long-range dependencies

```python
from torch.nn import Transformer

model = Transformer(d_model=64, nhead=8, num_layers=4)
```

**Expected gain**: +10-20 pp (if enough training data)
- State-of-the-art for sequence modeling
- Captures complex patterns
- But needs MUCH more training data (5+ years)

**Effort**: 3-5 days + large training set

---

## Realistic Assessment

### Quick Wins (1-2 days effort)
1. **Weekly granularity**: +10-20 pp (highest ROI) ⭐⭐⭐
2. **Longer lookback (365 days)**: +5-10 pp
3. **Multi-channel features**: +5-12 pp

**Combined potential**: 41.43% → 56-71% RW percentile

### Medium Effort (3-5 days)
4. **Regime detection + conditional**: +15-25 pp ⭐⭐⭐
5. **Multi-field ensemble**: +5-10 pp
6. **LSTM/RNN architecture**: +8-15 pp

**Combined potential**: 41.43% → 64-76% RW percentile (competitive!)

### High Effort (1-2 weeks)
7. **Transformer architecture**: +10-20 pp (with enough data)
8. **Full redesign with macro features**: +20-30 pp

---

## My Recommendation

### If Goal: Quick Performance Boost
**Priority 1**: Switch to weekly granularity (1-2 days)
- Retrain GAF+CNN on weekly data
- Interpolate predictions to daily weights
- Expected: 41.43% → 51-61% RW percentile

**Priority 2**: Add multi-channel features (1-2 days)
- Price + Returns + Volatility + Volume
- Retrain on 4-channel input
- Expected additional: +5-12 pp

**Total effort**: 2-4 days
**Expected result**: ~56-73% RW percentile (competitive)

---

### If Goal: Maximum Performance
**Implement regime detection system** (3-5 days):

```python
# Pseudocode
def allocate(prices_365d):
    # Stage 1: Regime detection (weekly, long lookback)
    regime = detect_regime(prices_365d_weekly)

    # Stage 2: Regime-conditional strategy
    if regime == 'bull':
        # Ride trend aggressively
        prob_up = predict_trend_continuation(prices_90d)
        sensitivity = 2.0  # aggressive
    elif regime == 'bear':
        # Defend capital
        prob_up = 0.5  # neutral
        sensitivity = 0.5  # conservative
    else:
        # Sideways: momentum-based
        prob_up = predict_momentum(prices_30d)
        sensitivity = 1.0

    return compute_weights(prob_up, sensitivity)
```

**Expected**: 56-66% RW percentile (top 25-40%)

---

## What NOT to Do

❌ **MTF/RP without fixing fundamental issues**
- Still has 90-day window problem
- Still has daily noise problem
- Expected gain: +2-5 pp (not worth effort)

❌ **Ensemble of weak models**
- If all models are bad, ensemble is still bad
- Fix individual model quality first

❌ **More complex CNN architecture**
- Adding layers doesn't fix wrong approach
- 296K params → 1M params won't help if signal is noise

---

## Tournament Context

**Important**: Based on README, tournament appears closed (winners announced).

**But for future tournaments or learning**:

### Top Priority Experiments (Ranked by ROI)
1. **Weekly granularity** ⭐⭐⭐ (1-2 days, +10-20 pp)
2. **Regime detection** ⭐⭐⭐ (3-5 days, +15-25 pp)
3. **Multi-channel features** ⭐⭐ (1-2 days, +5-12 pp)
4. **Longer lookback** ⭐ (1 day, +5-10 pp)
5. **LSTM architecture** ⭐ (2-3 days, +8-15 pp)
6. **Multi-field ensemble** (1-2 days, +5-10 pp)

### Skip These
- ❌ MTF/RP alone (low ROI)
- ❌ Spectrogram/wavelets alone (low ROI)
- ❌ Transformer (needs too much data)

---

## Conclusion

**Can we improve with imaging techniques?** Yes, but...

**Quick answer**:
- Other imaging (MTF, RP): +2-5 pp (not enough)
- Better fundamentals (weekly, multi-channel): +15-30 pp (competitive)
- Regime detection: +15-25 pp (best single improvement)

**Best path forward**:
1. Switch to weekly granularity (biggest bang for buck)
2. Add regime detection (captures BTC's cyclical nature)
3. Use multi-channel features (richer signal)

**Realistic outcome**: 56-73% RW percentile (top 25-40% territory)

**Is it worth it?**
- For this tournament: Closed
- For learning: Absolutely
- For future tournaments: Yes, but start with these improvements from day 1

**My honest take**: Image-based approaches CAN work for BTC if:
- Use weekly granularity (not daily)
- Long lookback (6-18 months)
- Regime-aware strategies
- Rich multi-channel features

But simpler approaches (momentum, moving averages, regime detection without images) might perform just as well with less complexity.
