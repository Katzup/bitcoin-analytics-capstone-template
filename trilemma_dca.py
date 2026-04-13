"""
Trilemma DCA System - Continuous Allocation Weights for Bitcoin

Converts binary trading signals into continuous allocation weights suitable
for dollar-cost averaging strategies. Designed for Trilemma practicum.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


def get_allocation_multiplier(
    prob_up: float,
    base_allocation: float = 1.0,
    sensitivity: float = 2.0,
    min_mult: float = 0.5,
    max_mult: float = 2.0
) -> float:
    """
    Convert model probability to allocation weight.

    This function transforms binary classification probabilities into continuous
    allocation multipliers, allowing for smooth, risk-aware position sizing
    instead of all-or-nothing trading decisions.

    Args:
        prob_up: Model probability of upward move [0, 1]
        base_allocation: Neutral allocation (1.0 = 100% of normal DCA)
        sensitivity: How much to tilt based on signal (higher = more aggressive)
                    - 1.0 = conservative (±0.5 max tilt)
                    - 2.0 = moderate (±1.0 max tilt)
                    - 3.0 = aggressive (±1.5 max tilt)
        min_mult: Minimum allocation (e.g., 0.5 = never go below 50% of base)
        max_mult: Maximum allocation (e.g., 2.0 = never exceed 2x base)

    Returns:
        allocation_multiplier: Float in [min_mult, max_mult]

    Examples:
        >>> get_allocation_multiplier(0.9, sensitivity=2.0)  # Strong buy signal
        1.8  # Buy 180% of base amount

        >>> get_allocation_multiplier(0.5, sensitivity=2.0)  # Neutral
        1.0  # Buy 100% of base amount

        >>> get_allocation_multiplier(0.2, sensitivity=2.0)  # Bearish
        0.5  # Buy 50% of base amount (capped at min_mult)
    """
    # Convert probability to signal strength
    signal = prob_up - 0.5  # Range: [-0.5, +0.5]

    # Scale by sensitivity
    tilt = sensitivity * signal

    # Apply to base allocation
    multiplier = base_allocation + tilt

    # Clip to bounds
    return np.clip(multiplier, min_mult, max_mult)


def smooth_allocations(
    allocations: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Apply exponential smoothing to allocation weights.

    Reduces week-to-week volatility in allocations, preventing excessive
    churning while maintaining responsiveness to model signals.

    Args:
        allocations: Array of allocation multipliers
        alpha: Smoothing factor [0, 1]
              - 0.0 = maximum smoothing (nearly constant)
              - 1.0 = no smoothing (raw signal)
              - 0.3 = recommended default

    Returns:
        Smoothed allocation array
    """
    smoothed = np.zeros_like(allocations)
    smoothed[0] = allocations[0]

    for i in range(1, len(allocations)):
        smoothed[i] = alpha * allocations[i] + (1 - alpha) * smoothed[i-1]

    return smoothed


class TrilemmaAllocator:
    """
    Generate continuous allocation weights for DCA strategy.

    This class converts model predictions into practicum-ready allocation
    schedules, comparing dynamic DCA against naive baseline.
    """

    def __init__(
        self,
        base_dca_amount: float = 100.0,
        dca_frequency: str = 'weekly',
        sensitivity: float = 2.0,
        min_mult: float = 0.5,
        max_mult: float = 2.0,
        smoothing_alpha: float = 0.3
    ):
        """
        Initialize Trilemma allocator.

        Args:
            base_dca_amount: Base dollar amount to invest per period
            dca_frequency: 'daily', 'weekly', 'biweekly', or 'monthly'
            sensitivity: Signal sensitivity (1.0-3.0)
            min_mult: Minimum allocation multiplier
            max_mult: Maximum allocation multiplier
            smoothing_alpha: EMA smoothing factor
        """
        self.base_amount = base_dca_amount
        self.frequency = dca_frequency
        self.sensitivity = sensitivity
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.smoothing_alpha = smoothing_alpha

    @staticmethod
    def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
        """Compute rolling z-score for normalization."""
        m = s.rolling(window, min_periods=max(5, window // 5)).mean()
        sd = s.rolling(window, min_periods=max(5, window // 5)).std(ddof=1)
        return (s - m) / (sd + 1e-12)

    def generate_allocation_schedule(
        self,
        predictions: List[Dict],
        price_data: pd.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Generate complete DCA allocation schedule from model predictions.

        Args:
            predictions: List of dicts with {'date', 'prob_up'}
            price_data: DataFrame with price history
            start_date: Optional start date for evaluation
            end_date: Optional end date for evaluation

        Returns:
            DataFrame with columns:
                - date: Allocation date
                - price: Bitcoin price
                - prob_up: Model probability
                - allocation_mult_raw: Raw allocation multiplier
                - allocation_mult_smooth: Smoothed multiplier
                - amount_dynamic: Dollar amount to invest (dynamic)
                - amount_naive: Dollar amount to invest (baseline)
                - btc_purchased: Bitcoin quantity purchased
        """
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)

        # Merge with price data
        daily_schedule = pred_df.merge(
            price_data[['date', 'close']],
            on='date',
            how='inner'
        )
        daily_schedule.rename(columns={'close': 'price'}, inplace=True)

        # Filter date range if specified
        if start_date:
            daily_schedule = daily_schedule[daily_schedule['date'] >= start_date]
        if end_date:
            daily_schedule = daily_schedule[daily_schedule['date'] <= end_date]

        # ----------------------------------------------------------------------
        # Blended allocator: z = a*signal + b*disc - c*vol, then tanh map
        # ----------------------------------------------------------------------

        # PARAMETERS (promote to self.* later if needed)
        ma_window = 90
        vol_window = 20
        z_window = 90
        a_signal = 1.0
        b_disc = 1.0
        c_vol = 0.4
        tanh_temp = 3.0  # bigger = smoother/smaller tilts

        # Ensure proper sorting / datetime index
        daily_schedule = daily_schedule.sort_values('date').copy()
        daily_schedule['date'] = pd.to_datetime(daily_schedule['date'])

        # 1) Core components
        daily_schedule['signal'] = daily_schedule['prob_up'] - 0.5

        # Price series
        px = daily_schedule['price'].astype(float)

        # Discount vs trend: positive when below MA
        ma = px.rolling(ma_window, min_periods=max(10, ma_window // 3)).mean()
        daily_schedule['disc'] = (ma - px) / ma

        # Realized volatility (of returns)
        rets = px.pct_change()
        vol = rets.rolling(vol_window, min_periods=max(5, vol_window // 2)).std(ddof=1)
        daily_schedule['vol'] = vol

        # 2) Normalize onto comparable scales (rolling z-score)
        daily_schedule['signal_z'] = self._rolling_zscore(daily_schedule['signal'], z_window)
        daily_schedule['disc_z'] = self._rolling_zscore(daily_schedule['disc'], z_window)
        daily_schedule['vol_z'] = self._rolling_zscore(daily_schedule['vol'], z_window)

        # Fill early NaNs with neutral values so we can compute multipliers everywhere
        daily_schedule[['signal_z', 'disc_z', 'vol_z']] = daily_schedule[['signal_z', 'disc_z', 'vol_z']].fillna(0.0)

        # 3) Combine into blended score
        daily_schedule['z'] = (
            a_signal * daily_schedule['signal_z'] +
            b_disc   * daily_schedule['disc_z'] -
            c_vol    * daily_schedule['vol_z']
        )

        # 4) Map score -> multiplier (smooth squashing, then clip)
        tilt = np.tanh(daily_schedule['z'].values / tanh_temp)  # [-1, 1]
        daily_schedule['allocation_mult_raw'] = np.clip(
            1.0 + self.sensitivity * tilt,
            self.min_mult,
            self.max_mult
        )

        # Check saturation (optional instrumentation)
        hit_min = (daily_schedule['allocation_mult_raw'] <= self.min_mult + 1e-9).mean()
        hit_max = (daily_schedule['allocation_mult_raw'] >= self.max_mult - 1e-9).mean()
        if hit_min > 0.05 or hit_max > 0.05:
            print(f"  [Allocator] Saturation: hit_min={hit_min:.1%}, hit_max={hit_max:.1%}")

        # 5) Apply smoothing (existing)
        daily_schedule['allocation_mult_smooth'] = smooth_allocations(
            daily_schedule['allocation_mult_raw'].values,
            alpha=self.smoothing_alpha
        )

        # DEBUG: Show rows 90-100 where rolling windows should be populated (disabled for WF)
        # if len(daily_schedule) > 0:
        #     debug_cols = ['date', 'price', 'prob_up', 'disc', 'vol', 'z', 'allocation_mult_smooth']
        #     start_idx = min(90, len(daily_schedule) - 10)
        #     print(f"\n  [Blended Allocator] Rows {start_idx}-{start_idx+10} (after rolling windows populate):")
        #     print(daily_schedule[debug_cols].iloc[start_idx:start_idx+10].to_string(index=False))

        # Resample to DCA frequency (daily, weekly, biweekly, monthly)
        if self.frequency == 'daily':
            schedule = daily_schedule.copy()
        elif self.frequency == 'weekly':
            # Buy on Mondays (or first available day of week)
            daily_schedule['date'] = pd.to_datetime(daily_schedule['date'])
            daily_schedule = daily_schedule.set_index('date')
            # Resample to weekly, taking the last value of the week for multiplier
            schedule = daily_schedule.resample('W-MON').agg({
                'price': 'last',
                'prob_up': 'last',
                'allocation_mult_raw': 'last',
                'allocation_mult_smooth': 'last'
            }).reset_index()
            schedule = schedule.dropna()  # Remove any NaN rows
        elif self.frequency == 'biweekly':
            daily_schedule['date'] = pd.to_datetime(daily_schedule['date'])
            daily_schedule = daily_schedule.set_index('date')
            schedule = daily_schedule.resample('2W-MON').agg({
                'price': 'last',
                'prob_up': 'last',
                'allocation_mult_raw': 'last',
                'allocation_mult_smooth': 'last'
            }).reset_index()
            schedule = schedule.dropna()
        elif self.frequency == 'monthly':
            daily_schedule['date'] = pd.to_datetime(daily_schedule['date'])
            daily_schedule = daily_schedule.set_index('date')
            schedule = daily_schedule.resample('MS').agg({  # Month start
                'price': 'last',
                'prob_up': 'last',
                'allocation_mult_raw': 'last',
                'allocation_mult_smooth': 'last'
            }).reset_index()
            schedule = schedule.dropna()
        else:
            raise ValueError(f"Unknown frequency: {self.frequency}")

        # Calculate raw dollar amounts
        schedule['amount_dynamic_raw'] = self.base_amount * schedule['allocation_mult_smooth']
        schedule['amount_naive'] = self.base_amount

        # Budget normalization: ensure dynamic strategy spends same total as naive
        # This makes the comparison fair (no cheating by spending more)
        total_budget = self.base_amount * len(schedule)
        total_dynamic_raw = schedule['amount_dynamic_raw'].sum()
        budget_scale = total_budget / total_dynamic_raw if total_dynamic_raw > 0 else 1.0

        schedule['amount_dynamic'] = schedule['amount_dynamic_raw'] * budget_scale

        # Calculate units purchased (asset-agnostic: could be BTC, SPY shares, etc.)
        schedule['units_purchased'] = schedule['amount_dynamic'] / schedule['price']
        schedule['units_purchased_naive'] = schedule['amount_naive'] / schedule['price']

        # Calculate cumulative metrics
        schedule['cumulative_invested'] = schedule['amount_dynamic'].cumsum()
        schedule['cumulative_invested_naive'] = schedule['amount_naive'].cumsum()
        schedule['cumulative_units'] = schedule['units_purchased'].cumsum()
        schedule['cumulative_units_naive'] = schedule['units_purchased_naive'].cumsum()
        schedule['portfolio_value'] = schedule['cumulative_units'] * schedule['price']
        schedule['portfolio_value_naive'] = schedule['cumulative_units_naive'] * schedule['price']

        # Build daily valuation for accurate drawdown and volatility metrics
        daily_valuation = self._build_daily_valuation(
            price_data=price_data,
            weekly_schedule=schedule,
            start_date=schedule['date'].iloc[0] if len(schedule) > 0 else None,
            end_date=schedule['date'].iloc[-1] if len(schedule) > 0 else None
        )

        return schedule, daily_valuation

    def _build_daily_valuation(
        self,
        price_data: pd.DataFrame,
        weekly_schedule: pd.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Build daily portfolio valuation from weekly buy schedule and daily prices.

        Args:
            price_data: DataFrame with daily prices (date, close columns)
            weekly_schedule: Weekly buy schedule with amount_dynamic, amount_naive
            start_date: Optional start date for valuation window
            end_date: Optional end date for valuation window

        Returns:
            DataFrame with daily portfolio values indexed by date
        """
        # Prepare daily price series
        prices = price_data[['date', 'close']].copy()
        prices['date'] = pd.to_datetime(prices['date'])
        prices = prices.sort_values('date').set_index('date')

        # Prepare weekly schedule
        sched = weekly_schedule[['date', 'amount_dynamic', 'amount_naive']].copy()
        sched['date'] = pd.to_datetime(sched['date'])
        sched = sched.sort_values('date').set_index('date')

        # Create daily spend series (zeros except on buy dates)
        daily_spend_dyn = pd.Series(0.0, index=prices.index)
        daily_spend_naive = pd.Series(0.0, index=prices.index)

        # Assign spend amounts on buy dates (only where they exist in price index)
        buy_dates = sched.index.intersection(prices.index)
        daily_spend_dyn.loc[buy_dates] = sched.loc[buy_dates, 'amount_dynamic'].astype(float).values
        daily_spend_naive.loc[buy_dates] = sched.loc[buy_dates, 'amount_naive'].astype(float).values

        # Compute daily units bought and cumulative holdings
        units_bought_dyn = daily_spend_dyn / prices['close']
        units_bought_naive = daily_spend_naive / prices['close']

        units_dyn = units_bought_dyn.cumsum()
        units_naive = units_bought_naive.cumsum()

        # Daily portfolio values
        value_dyn = units_dyn * prices['close']
        value_naive = units_naive * prices['close']

        # Build output DataFrame
        daily_val = pd.DataFrame({
            'price': prices['close'],
            'spend_dynamic': daily_spend_dyn,
            'spend_naive': daily_spend_naive,
            'units_dynamic': units_dyn,
            'units_naive': units_naive,
            'value_dynamic': value_dyn,
            'value_naive': value_naive,
        }, index=prices.index)

        # Restrict to evaluation window (first buy to last date in schedule)
        if len(buy_dates) > 0:
            eval_start = buy_dates.min()
            eval_end = sched.index.max()
            daily_val = daily_val.loc[eval_start:eval_end]

        return daily_val

    def calculate_metrics(self, schedule: pd.DataFrame, daily_valuation: pd.DataFrame = None) -> Dict:
        """
        Calculate comprehensive performance metrics for practicum report.

        Args:
            schedule: Allocation schedule from generate_allocation_schedule()

        Returns:
            Dictionary of metrics comparing dynamic vs naive DCA
        """
        # Final values
        total_invested_dynamic = schedule['cumulative_invested'].iloc[-1]
        total_invested_naive = schedule['cumulative_invested_naive'].iloc[-1]
        total_units_dynamic = schedule['cumulative_units'].iloc[-1]
        total_units_naive = schedule['cumulative_units_naive'].iloc[-1]
        final_price = schedule['price'].iloc[-1]
        final_value_dynamic = total_units_dynamic * final_price
        final_value_naive = total_units_naive * final_price

        # Average unit cost (cost basis)
        avg_unit_cost_dynamic = total_invested_dynamic / total_units_dynamic if total_units_dynamic > 0 else 0
        avg_unit_cost_naive = total_invested_naive / total_units_naive if total_units_naive > 0 else 0

        # Returns
        total_return_dynamic = (final_value_dynamic - total_invested_dynamic) / total_invested_dynamic
        total_return_naive = (final_value_naive - total_invested_naive) / total_invested_naive
        alpha = total_return_dynamic - total_return_naive

        # Drawdown - computed on daily portfolio values for accuracy
        if daily_valuation is not None and len(daily_valuation) > 0:
            # Use daily valuation for accurate drawdown
            def compute_max_drawdown(value: pd.Series) -> float:
                value = value.dropna()
                if value.empty:
                    return 0.0
                peak = value.cummax()
                dd = (value / peak) - 1.0
                return float(dd.min())  # negative number

            max_drawdown_dynamic = compute_max_drawdown(daily_valuation['value_dynamic'])
            max_drawdown_naive = compute_max_drawdown(daily_valuation['value_naive'])
            max_drawdown = max_drawdown_dynamic  # Report dynamic for primary metric
        else:
            # Fallback to weekly approximation if daily valuation not provided
            running_max = schedule['portfolio_value'].expanding().max()
            drawdown = (schedule['portfolio_value'] - running_max) / running_max
            max_drawdown = drawdown.min()
            max_drawdown_dynamic = max_drawdown
            max_drawdown_naive = 0.0

        # Information Ratio: tracking error vs naive baseline
        # Compute period-to-period returns (weekly if DCA is weekly)
        schedule['dynamic_pct'] = schedule['portfolio_value'].pct_change().fillna(0)
        schedule['naive_pct'] = schedule['portfolio_value_naive'].pct_change().fillna(0)
        schedule['tracking_error'] = schedule['dynamic_pct'] - schedule['naive_pct']

        # Information Ratio (annualized)
        # IR = mean(tracking_error) / std(tracking_error) * sqrt(periods_per_year)
        if len(schedule) > 1 and schedule['tracking_error'].std() > 0:
            # Annualization factor based on actual frequency
            if self.frequency == 'weekly':
                annualization_factor = np.sqrt(52)
            elif self.frequency == 'biweekly':
                annualization_factor = np.sqrt(26)
            elif self.frequency == 'monthly':
                annualization_factor = np.sqrt(12)
            else:  # daily
                annualization_factor = np.sqrt(252)

            information_ratio = (schedule['tracking_error'].mean() / schedule['tracking_error'].std()) * annualization_factor
        else:
            information_ratio = 0

        # Allocation turnover (churn metric)
        allocation_changes = schedule['allocation_mult_smooth'].diff().abs()
        avg_turnover = allocation_changes.mean()
        total_turnover = allocation_changes.sum()

        # Unit accumulation efficiency
        units_per_dollar_dynamic = total_units_dynamic / total_invested_dynamic if total_invested_dynamic > 0 else 0
        units_per_dollar_naive = total_units_naive / total_invested_naive if total_invested_naive > 0 else 0
        accumulation_improvement = (units_per_dollar_dynamic - units_per_dollar_naive) / units_per_dollar_naive if units_per_dollar_naive > 0 else 0

        return {
            # Dynamic DCA metrics
            'total_invested_dynamic': total_invested_dynamic,
            'total_units_accumulated_dynamic': total_units_dynamic,
            'avg_unit_cost_dynamic': avg_unit_cost_dynamic,
            'final_value_dynamic': final_value_dynamic,
            'total_return_pct_dynamic': total_return_dynamic * 100,

            # Naive DCA metrics
            'total_invested_naive': total_invested_naive,
            'total_units_accumulated_naive': total_units_naive,
            'avg_unit_cost_naive': avg_unit_cost_naive,
            'final_value_naive': final_value_naive,
            'total_return_pct_naive': total_return_naive * 100,

            # Comparative metrics
            'alpha_vs_naive': alpha * 100,
            'unit_accumulation_improvement': accumulation_improvement * 100,
            'unit_cost_improvement': ((avg_unit_cost_naive - avg_unit_cost_dynamic) / avg_unit_cost_naive) * 100 if avg_unit_cost_naive > 0 else 0,

            # Risk metrics
            'max_drawdown_pct': max_drawdown * 100,
            'information_ratio': information_ratio,

            # Efficiency metrics
            'avg_allocation_turnover': avg_turnover,
            'total_allocation_turnover': total_turnover,
            'num_periods': len(schedule),
            'date_range': f"{schedule['date'].iloc[0]} to {schedule['date'].iloc[-1]}"
        }

    def print_report(self, metrics: Dict):
        """Print formatted practicum report."""
        print("=" * 80)
        print("TRILEMMA DCA SYSTEM - PERFORMANCE REPORT")
        print("=" * 80)
        print(f"\nEvaluation Period: {metrics['date_range']}")
        print(f"Number of DCA Periods: {metrics['num_periods']}")
        print(f"Base DCA Amount: ${self.base_amount:.2f} per period")
        print()

        print("DYNAMIC DCA (Model-Driven)")
        print("-" * 40)
        print(f"  Total Invested:     ${metrics['total_invested_dynamic']:,.2f}")
        print(f"  Units Accumulated:  {metrics['total_units_accumulated_dynamic']:.8f}")
        print(f"  Avg Unit Cost:      ${metrics['avg_unit_cost_dynamic']:,.2f}")
        print(f"  Final Value:        ${metrics['final_value_dynamic']:,.2f}")
        print(f"  Total Return:       {metrics['total_return_pct_dynamic']:+.2f}%")
        print()

        print("NAIVE DCA (Baseline)")
        print("-" * 40)
        print(f"  Total Invested:     ${metrics['total_invested_naive']:,.2f}")
        print(f"  Units Accumulated:  {metrics['total_units_accumulated_naive']:.8f}")
        print(f"  Avg Unit Cost:      ${metrics['avg_unit_cost_naive']:,.2f}")
        print(f"  Final Value:        ${metrics['final_value_naive']:,.2f}")
        print(f"  Total Return:       {metrics['total_return_pct_naive']:+.2f}%")
        print()

        print("COMPARATIVE PERFORMANCE")
        print("-" * 40)
        print(f"  Alpha vs Naive:             {metrics['alpha_vs_naive']:+.2f}%")
        print(f"  Unit Accumulation Boost:    {metrics['unit_accumulation_improvement']:+.2f}%")
        print(f"  Unit Cost Improvement:      {metrics['unit_cost_improvement']:+.2f}%")
        print()

        print("RISK METRICS")
        print("-" * 40)
        print(f"  Max Drawdown:               {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Information Ratio:          {metrics['information_ratio']:.2f}")
        print()

        print("EFFICIENCY METRICS")
        print("-" * 40)
        print(f"  Avg Allocation Turnover:    {metrics['avg_allocation_turnover']:.4f}")
        print(f"  Total Turnover:             {metrics['total_allocation_turnover']:.2f}")
        print("=" * 80)


def main():
    """Example usage."""
    print("Trilemma DCA Allocation System")
    print("See trilemma_runner.py for complete implementation")


if __name__ == '__main__':
    main()
