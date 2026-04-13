"""
Multi-Timeframe Configuration Presets
Translated from daily baseline to {1h, 4h, 1d} with BTC 24/7 assumptions
"""

from dataclasses import dataclass
from typing import Literal

# Assumptions:
# - BTC trades 24/7
# - 1d = 1 bar/day, 4h = 6 bars/day, 1h = 24 bars/day
# - Day boundary: 00:00 UTC for alignment
# - Weekly DCA = every 7 days (7 × bars_per_day bars)


@dataclass
class BarConfig:
    """Bar/timeframe specification."""
    timespan: Literal["hour", "day"]
    multiplier: int  # 1h=1, 4h=4, 1d=1
    bars_per_day: int  # 1h=24, 4h=6, 1d=1


@dataclass
class CNNConfig:
    """CNN model parameters (in bars, not days)."""
    lookback_bars: int  # Window for GAF image generation
    horizon_bars: int  # Prediction forward window
    num_epochs: int  # Training epochs


@dataclass
class DCAConfig:
    """DCA allocation parameters."""
    base_dca_amount: float  # $ per DCA period
    rebalance_every_bars: int  # Weekly = 7 × bars_per_day
    sensitivity: float  # Allocation tilt (lower for noisier timeframes)
    min_mult: float  # Minimum allocation multiplier
    max_mult: float  # Maximum allocation multiplier
    smoothing_alpha: float  # EMA smoothing (lower = more smoothing)


@dataclass
class WalkForwardConfig:
    """Walk-forward evaluation windows (in bars)."""
    train_bars: int
    val_bars: int
    test_bars: int
    step_bars: int
    min_train_samples: int


@dataclass
class CostConfig:
    """Transaction cost model (basis points)."""
    fee_bps: float  # Exchange fee (taker-ish)
    slippage_bps: float  # Market impact
    spread_bps: float  # Bid-ask spread

    @property
    def total_bps(self) -> float:
        """Total cost in bps."""
        return self.fee_bps + self.slippage_bps + self.spread_bps

    def apply_to_price(self, price: float, side: str = 'buy') -> float:
        """
        Apply costs to execution price.

        For buys: fill = price × (1 + total_bps/10000)
        For sells: fill = price × (1 - total_bps/10000)
        """
        multiplier = 1 + self.total_bps / 10000 if side == 'buy' else 1 - self.total_bps / 10000
        return price * multiplier


@dataclass
class MultiTimeframePreset:
    """Complete preset for a given timeframe."""
    name: str
    bars: BarConfig
    cnn: CNNConfig
    dca: DCAConfig
    walk_forward: WalkForwardConfig
    costs: CostConfig


# ============================================================================
# PRESETS: {1h, 4h, 1d}
# Translated from daily baseline (90d lookback, 10d horizon, weekly DCA)
# ============================================================================

BTC_1D = MultiTimeframePreset(
    name="btc_1d",
    bars=BarConfig(
        timespan="day",
        multiplier=1,
        bars_per_day=1
    ),
    cnn=CNNConfig(
        lookback_bars=90,  # 90 days
        horizon_bars=10,   # 10 days
        num_epochs=50
    ),
    dca=DCAConfig(
        base_dca_amount=100.0,
        rebalance_every_bars=7,  # Weekly (7 days)
        sensitivity=1.5,  # Moderate tilt (baseline)
        min_mult=0.7,
        max_mult=1.6,
        smoothing_alpha=0.30  # Baseline smoothing
    ),
    walk_forward=WalkForwardConfig(
        train_bars=365,  # 1 year
        val_bars=120,    # ~4 months
        test_bars=60,    # ~2 months
        step_bars=30,    # 1 month
        min_train_samples=50
    ),
    costs=CostConfig(
        fee_bps=10.0,   # 0.10% fee (taker-ish)
        slippage_bps=2.0,  # 0.02% slippage
        spread_bps=1.0     # 0.01% spread
        # Total: ~13 bps (0.13%)
    )
)


BTC_4H = MultiTimeframePreset(
    name="btc_4h",
    bars=BarConfig(
        timespan="hour",
        multiplier=4,
        bars_per_day=6
    ),
    cnn=CNNConfig(
        lookback_bars=540,  # 90 days × 6 bars/day
        horizon_bars=60,    # 10 days × 6 bars/day
        num_epochs=50
    ),
    dca=DCAConfig(
        base_dca_amount=100.0,
        rebalance_every_bars=42,  # Weekly (7 days × 6)
        sensitivity=1.3,  # Slightly lower (noisier probabilities)
        min_mult=0.7,
        max_mult=1.6,
        smoothing_alpha=0.25  # More smoothing than 1d
    ),
    walk_forward=WalkForwardConfig(
        train_bars=2190,   # 365 days × 6
        val_bars=720,      # 120 days × 6
        test_bars=360,     # 60 days × 6
        step_bars=180,     # 30 days × 6
        min_train_samples=50
    ),
    costs=CostConfig(
        fee_bps=10.0,
        slippage_bps=4.0,  # Higher slippage (more frequent)
        spread_bps=2.0
        # Total: ~16 bps (0.16%)
    )
)


BTC_1H = MultiTimeframePreset(
    name="btc_1h",
    bars=BarConfig(
        timespan="hour",
        multiplier=1,
        bars_per_day=24
    ),
    cnn=CNNConfig(
        lookback_bars=2160,  # 90 days × 24 bars/day
        horizon_bars=240,    # 10 days × 24 bars/day
        num_epochs=50
    ),
    dca=DCAConfig(
        base_dca_amount=100.0,
        rebalance_every_bars=168,  # Weekly (7 days × 24)
        sensitivity=1.1,  # Much lower (very noisy)
        min_mult=0.7,
        max_mult=1.6,
        smoothing_alpha=0.18  # Heavy smoothing to control churn
    ),
    walk_forward=WalkForwardConfig(
        train_bars=8760,   # 365 days × 24
        val_bars=2880,     # 120 days × 24
        test_bars=1440,    # 60 days × 24
        step_bars=720,     # 30 days × 24
        min_train_samples=50
    ),
    costs=CostConfig(
        fee_bps=10.0,
        slippage_bps=8.0,  # Highest slippage (most frequent)
        spread_bps=3.0
        # Total: ~21 bps (0.21%)
    )
)


# Preset registry for easy lookup
PRESETS = {
    "1d": BTC_1D,
    "4h": BTC_4H,
    "1h": BTC_1H,
}


def get_preset(timeframe: str) -> MultiTimeframePreset:
    """
    Get configuration preset by timeframe.

    Args:
        timeframe: "1d", "4h", or "1h"

    Returns:
        MultiTimeframePreset configuration

    Raises:
        ValueError if timeframe not recognized
    """
    if timeframe not in PRESETS:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Choose from: {list(PRESETS.keys())}")
    return PRESETS[timeframe]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example: Load 1h preset
    preset_1h = get_preset("1h")

    print(f"Preset: {preset_1h.name}")
    print(f"  Bars: {preset_1h.bars.multiplier}{preset_1h.bars.timespan} ({preset_1h.bars.bars_per_day} bars/day)")
    print(f"  CNN Lookback: {preset_1h.cnn.lookback_bars} bars ({preset_1h.cnn.lookback_bars / preset_1h.bars.bars_per_day:.0f} days)")
    print(f"  DCA Frequency: Every {preset_1h.dca.rebalance_every_bars} bars ({preset_1h.dca.rebalance_every_bars / preset_1h.bars.bars_per_day:.0f} days)")
    print(f"  Sensitivity: {preset_1h.dca.sensitivity} (vs 1d: {BTC_1D.dca.sensitivity})")
    print(f"  Smoothing α: {preset_1h.dca.smoothing_alpha} (vs 1d: {BTC_1D.dca.smoothing_alpha})")
    print(f"  Total Costs: {preset_1h.costs.total_bps:.1f} bps ({preset_1h.costs.total_bps/100:.3f}%)")
    print()

    # Cost application example
    btc_price = 90000.0
    buy_price_1h = preset_1h.costs.apply_to_price(btc_price, side='buy')
    buy_price_1d = BTC_1D.costs.apply_to_price(btc_price, side='buy')

    print(f"BTC Price: ${btc_price:,.2f}")
    print(f"  1h effective buy: ${buy_price_1h:,.2f} (+${buy_price_1h - btc_price:.2f})")
    print(f"  1d effective buy: ${buy_price_1d:,.2f} (+${buy_price_1d - btc_price:.2f})")
    print(f"  1h penalty vs 1d: ${buy_price_1h - buy_price_1d:.2f}")
