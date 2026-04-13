#!/usr/bin/env python3
"""
Analyze temporal performance pattern from tournament results.
"""

import pandas as pd
import numpy as np

# Load detailed results
results = pd.read_csv('tournament_results_detailed.csv')
results['start'] = pd.to_datetime(results['start'])
results['end'] = pd.to_datetime(results['end'])
results['year'] = results['start'].dt.year

# Yearly aggregation
yearly = results.groupby('year').agg({
    'pct_strategy': ['mean', 'median', 'std', 'min', 'max'],
    'pct_uniform': ['mean', 'median'],
    'window_idx': 'count'
}).round(2)

yearly.columns = ['_'.join(col).strip() for col in yearly.columns.values]
yearly = yearly.rename(columns={'window_idx_count': 'n_windows'})

print("=" * 80)
print("TEMPORAL PERFORMANCE ANALYSIS")
print("=" * 80)
print("\nYearly VTS Strategy Performance:")
print(yearly.to_string())

# Calculate win rate by year
win_by_year = results.groupby('year').apply(
    lambda x: (x['pct_strategy'] > x['pct_uniform']).mean()
).round(4)

print("\n" + "=" * 80)
print("YEARLY WIN RATES")
print("=" * 80)
for year, rate in win_by_year.items():
    print(f"{year}: {rate:.2%} of windows")

# Performance trend
print("\n" + "=" * 80)
print("PERFORMANCE TREND")
print("=" * 80)

early_period = results[results['year'].isin([2016, 2017, 2018])]['pct_strategy'].mean()
mid_period = results[results['year'].isin([2019, 2020, 2021])]['pct_strategy'].mean()
late_period = results[results['year'].isin([2022, 2023, 2024])]['pct_strategy'].mean()

print(f"\nEarly period (2016-2018): {early_period:.2f}% avg percentile")
print(f"Mid period   (2019-2021): {mid_period:.2f}% avg percentile")
print(f"Late period  (2022-2024): {late_period:.2f}% avg percentile")
print(f"\nTrend: {early_period:.2f}% → {mid_period:.2f}% → {late_period:.2f}%")
print(f"Total decline: {late_period - early_period:.2f} percentage points")

# Volatility analysis
print("\n" + "=" * 80)
print("PERFORMANCE STABILITY")
print("=" * 80)

early_std = results[results['year'].isin([2016, 2017, 2018])]['pct_strategy'].std()
late_std = results[results['year'].isin([2022, 2023, 2024])]['pct_strategy'].std()

print(f"\nEarly period std dev: {early_std:.2f}%")
print(f"Late period std dev:  {late_std:.2f}%")
print(f"Stability change: {((late_std - early_std) / early_std * 100):.1f}%")

# Best/worst windows
print("\n" + "=" * 80)
print("EXTREME WINDOWS")
print("=" * 80)

best = results.nlargest(5, 'pct_strategy')[['start', 'end', 'pct_strategy', 'pct_uniform']]
worst = results.nsmallest(5, 'pct_strategy')[['start', 'end', 'pct_strategy', 'pct_uniform']]

print("\nBest 5 Windows:")
print(best.to_string(index=False))

print("\nWorst 5 Windows:")
print(worst.to_string(index=False))

# Quartile analysis
print("\n" + "=" * 80)
print("PERFORMANCE DISTRIBUTION BY PERIOD")
print("=" * 80)

for period, years in [
    ("2016-2018", [2016, 2017, 2018]),
    ("2019-2021", [2019, 2020, 2021]),
    ("2022-2024", [2022, 2023, 2024])
]:
    subset = results[results['year'].isin(years)]['pct_strategy']
    print(f"\n{period}:")
    print(f"  Min:    {subset.min():.2f}%")
    print(f"  Q1:     {subset.quantile(0.25):.2f}%")
    print(f"  Median: {subset.median():.2f}%")
    print(f"  Q3:     {subset.quantile(0.75):.2f}%")
    print(f"  Max:    {subset.max():.2f}%")
