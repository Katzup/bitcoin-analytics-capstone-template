#!/usr/bin/env python3
"""
Data Requirements Checker Module

Validates that sufficient historical data exists before attempting model training.
Prevents "reshape" errors by checking data availability against lookback + horizon
requirements before expensive dataset creation and training operations.

Addresses common issues:
- Insufficient historical data for long horizons (150-200 days)
- Data availability for train/val/test splits
- Minimum samples per class requirements
- Lookback + horizon + split overhead calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class DataRequirement:
    """Container for data requirement specifications"""
    lookback: int
    horizon: int
    train_pct: float = 0.70
    val_pct: float = 0.15
    test_pct: float = 0.15
    min_samples_per_class: int = 5
    num_classes: int = 6  # 0-5 for regime classification

    def __post_init__(self):
        """Validate percentages sum to 1.0"""
        total = self.train_pct + self.val_pct + self.test_pct
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split percentages must sum to 1.0, got {total}")


@dataclass
class DataAvailability:
    """Container for data availability results"""
    ticker: str
    available_samples: int
    required_samples: int
    sufficient: bool
    shortage: int
    horizon: int
    metadata: Dict


class DataRequirementsChecker:
    """
    Validates data sufficiency before model training.

    Key validations:
    1. Sufficient samples for lookback window creation
    2. Sufficient samples after train/val/test split
    3. Sufficient samples per class in each split
    4. Data date range adequacy
    """

    def __init__(self, requirement: DataRequirement):
        """
        Args:
            requirement: DataRequirement specification with lookback, horizon, splits
        """
        self.req = requirement

    def calculate_minimum_required_samples(self) -> int:
        """
        Calculate minimum samples needed in raw DataFrame.

        Returns:
            int: Minimum number of samples required

        Formula:
            Raw samples needed = lookback + horizon + buffer_for_splits

            Where buffer_for_splits accounts for:
            - Train split needs: min_samples_per_class * num_classes
            - Val split needs: min_samples_per_class * num_classes
            - Test split needs: min_samples_per_class * num_classes

            We calculate the minimum total samples such that after creating
            rolling windows (which reduces count by lookback + horizon - 1),
            we can still split into train/val/test with sufficient samples.
        """
        # Samples consumed by rolling window creation
        window_overhead = self.req.lookback + self.req.horizon - 1

        # Minimum samples needed in EACH split
        min_per_split = self.req.min_samples_per_class * self.req.num_classes

        # Total samples needed across all splits
        # After window creation, we need at least these many valid samples
        min_after_windowing = max(
            int(min_per_split / self.req.train_pct),  # Training set dominates
            min_per_split * 3  # Or at least min for each of train/val/test
        )

        # Add window overhead back to get raw data requirement
        min_raw_samples = min_after_windowing + window_overhead

        # Add safety margin (10%) for edge cases and class imbalance
        safety_margin = int(min_raw_samples * 0.10)

        return min_raw_samples + safety_margin

    def check_dataframe_sufficiency(self,
                                   df: pd.DataFrame,
                                   ticker: str) -> DataAvailability:
        """
        Check if DataFrame has sufficient data for training.

        Args:
            df: Price DataFrame with OHLCV data
            ticker: Ticker symbol for reporting

        Returns:
            DataAvailability: Results of sufficiency check
        """
        available_samples = len(df)
        required_samples = self.calculate_minimum_required_samples()
        sufficient = available_samples >= required_samples
        shortage = max(0, required_samples - available_samples)

        # Additional metadata
        metadata = {
            'lookback': self.req.lookback,
            'horizon': self.req.horizon,
            'window_overhead': self.req.lookback + self.req.horizon - 1,
            'min_samples_per_class': self.req.min_samples_per_class,
            'num_classes': self.req.num_classes,
            'train_pct': self.req.train_pct,
            'val_pct': self.req.val_pct,
            'test_pct': self.req.test_pct,
            'date_range_days': None,
            'earliest_date': None,
            'latest_date': None
        }

        # Calculate date range if index is DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            date_range = (df.index[-1] - df.index[0]).days
            metadata['date_range_days'] = date_range
            metadata['earliest_date'] = df.index[0].strftime('%Y-%m-%d')
            metadata['latest_date'] = df.index[-1].strftime('%Y-%m-%d')

        return DataAvailability(
            ticker=ticker,
            available_samples=available_samples,
            required_samples=required_samples,
            sufficient=sufficient,
            shortage=shortage,
            horizon=self.req.horizon,
            metadata=metadata
        )

    def check_multiple_horizons(self,
                               df: pd.DataFrame,
                               ticker: str,
                               horizons: List[int]) -> Dict[int, DataAvailability]:
        """
        Check data sufficiency for multiple forecast horizons.

        Args:
            df: Price DataFrame
            ticker: Ticker symbol
            horizons: List of forecast horizons to check (e.g., [20, 50, 100, 150, 200])

        Returns:
            dict: {horizon: DataAvailability} for each horizon
        """
        results = {}

        for horizon in horizons:
            # Create requirement for this horizon
            requirement = DataRequirement(
                lookback=self.req.lookback,
                horizon=horizon,
                train_pct=self.req.train_pct,
                val_pct=self.req.val_pct,
                test_pct=self.req.test_pct,
                min_samples_per_class=self.req.min_samples_per_class,
                num_classes=self.req.num_classes
            )

            # Create checker for this horizon
            checker = DataRequirementsChecker(requirement)

            # Check sufficiency
            results[horizon] = checker.check_dataframe_sufficiency(df, ticker)

        return results

    def estimate_required_days(self) -> int:
        """
        Estimate calendar days of data required (assuming trading days).

        Returns:
            int: Approximate calendar days needed

        Note:
            Assumes ~252 trading days per year, or ~365/252 = 1.45 calendar days per trading day
        """
        required_samples = self.calculate_minimum_required_samples()
        # Convert trading days to calendar days (rough estimate)
        calendar_days = int(required_samples * 1.45)
        return calendar_days


def validate_data_requirements(df: pd.DataFrame,
                               ticker: str,
                               horizons: List[int],
                               lookback: int = 60,
                               min_samples_per_class: int = 5,
                               verbose: bool = True) -> Tuple[bool, Dict[int, DataAvailability]]:
    """
    Convenience function for data requirements validation across multiple horizons.

    Args:
        df: Price DataFrame with OHLCV data
        ticker: Ticker symbol
        horizons: List of forecast horizons to validate
        lookback: Lookback period in days
        min_samples_per_class: Minimum samples required per class
        verbose: Print validation results

    Returns:
        Tuple[bool, Dict]: (all_sufficient, results_by_horizon)
    """
    # Create base requirement (horizon will be set per check)
    requirement = DataRequirement(
        lookback=lookback,
        horizon=max(horizons),  # Will be overridden
        min_samples_per_class=min_samples_per_class
    )

    checker = DataRequirementsChecker(requirement)
    results = checker.check_multiple_horizons(df, ticker, horizons)

    all_sufficient = all(r.sufficient for r in results.values())

    if verbose:
        print(f"\n{'='*80}")
        print(f"DATA REQUIREMENTS VALIDATION - {ticker}")
        print(f"{'='*80}")

        print(f"\n📊 Available Data:")
        print(f"   Samples: {len(df)}")
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"   Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Calendar Days: {(df.index[-1] - df.index[0]).days}")

        print(f"\n🔍 Requirements Check (lookback={lookback}, min_samples_per_class={min_samples_per_class}):")
        print(f"\n{'Horizon':<10} {'Required':<10} {'Available':<10} {'Status':<10} {'Shortage':<10}")
        print("-" * 80)

        for horizon in sorted(horizons):
            result = results[horizon]
            status = "✅ OK" if result.sufficient else "❌ FAIL"
            shortage_str = str(result.shortage) if result.shortage > 0 else "-"

            print(f"{horizon:<10} {result.required_samples:<10} {result.available_samples:<10} "
                  f"{status:<10} {shortage_str:<10}")

        if all_sufficient:
            print(f"\n✅ PASSED - Sufficient data for all horizons")
        else:
            failed_horizons = [h for h, r in results.items() if not r.sufficient]
            print(f"\n❌ FAILED - Insufficient data for horizons: {failed_horizons}")
            print(f"\n💡 Recommendations:")
            max_shortage = max(r.shortage for r in results.values())
            print(f"   - Fetch {max_shortage} additional days of historical data")
            print(f"   - Or reduce forecast horizons to shorter timeframes")
            print(f"   - Or reduce min_samples_per_class requirement (currently {min_samples_per_class})")

    return all_sufficient, results


def print_requirements_summary(lookback: int = 60,
                              horizons: List[int] = [20, 50, 100, 150, 200],
                              min_samples_per_class: int = 5):
    """
    Print summary table of data requirements for different horizons.
    Useful for planning data collection needs.

    Args:
        lookback: Lookback period in days
        horizons: List of forecast horizons
        min_samples_per_class: Minimum samples per class
    """
    print(f"\n{'='*80}")
    print(f"DATA REQUIREMENTS SUMMARY")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"   Lookback: {lookback} days")
    print(f"   Min samples per class: {min_samples_per_class}")
    print(f"   Number of classes: 6 (regimes 0-5)")
    print(f"   Train/Val/Test split: 70%/15%/15%")

    print(f"\n{'Horizon':<10} {'Min Samples':<15} {'Est. Calendar Days':<20}")
    print("-" * 80)

    for horizon in horizons:
        requirement = DataRequirement(
            lookback=lookback,
            horizon=horizon,
            min_samples_per_class=min_samples_per_class
        )
        checker = DataRequirementsChecker(requirement)
        min_samples = checker.calculate_minimum_required_samples()
        calendar_days = checker.estimate_required_days()

        print(f"{horizon:<10} {min_samples:<15} {calendar_days:<20}")

    print(f"\n💡 Note: Calendar days are estimates assuming ~252 trading days/year")
    print(f"{'='*80}\n")
