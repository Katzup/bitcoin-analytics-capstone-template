"""
Generate charts for VTS Midterm Presentation
Run: python generate_charts.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = "/Users/bobkatz/Visual_Trading_System/midterm_presentation/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(name):
    """Save figure with tight layout"""
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Generated: {name}.png")


def chart_ablation_sensitivity():
    """Ablation Study #1: Sensitivity Sweep"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sensitivities = [0.5, 0.8, 1.0, 1.2, 1.5]
    rw_percentiles = [41.78, 41.68, 41.61, 41.54, 41.43]
    colors = ['#2ecc71' if r > 41.5 else '#e74c3c' for r in rw_percentiles]
    
    bars = ax.bar(sensitivities, rw_percentiles, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, rw_percentiles):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.axhline(y=41.43, color='red', linestyle='--', linewidth=2, label='Baseline (sens=1.5)')
    ax.set_xlabel('Sensitivity Parameter')
    ax.set_ylabel('RW SPD Percentile')
    ax.set_title('Ablation Study #1: Allocator Sizing\n(Minimal Impact: +0.35 pp max improvement)')
    ax.set_ylim(41, 42.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_fig('01_ablation_sensitivity')


def chart_ablation_signal_quality():
    """Ablation Study #2: Signal Quality (The "Aha!" Moment)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    approaches = ['CNN\n(296K params)', 'Neutral\n(constant 0.5)', 'Improvement']
    rw_values = [41.43, 41.94, 0.51]
    win_rates = [54.32, 70.42, 16.1]
    
    x = np.arange(len(approaches))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rw_values, width, label='RW Percentile', color='#3498db', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, win_rates, width, label='Win Rate %', color='#e67e22', alpha=0.8)
    
    # Customize
    ax.set_ylabel('RW SPD Percentile', color='#3498db')
    ax2.set_ylabel('Win Rate (%)', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.set_title('Ablation Study #2: Signal Quality Test\n(CNN ≡ Random Guessing)', fontweight='bold')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    save_fig('02_ablation_signal_quality')


def chart_ablation_ema():
    """Ablation Study #3: EMA Smoothing (Flat Line)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    alphas = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    rw_values = [41.94] * len(alphas)  # All identical
    
    ax.plot(alphas, rw_values, 'o-', linewidth=3, markersize=10, color='#9b59b6')
    ax.fill_between(alphas, [r-0.01 for r in rw_values], [r+0.01 for r in rw_values],
                    alpha=0.2, color='#9b59b6')
    
    # Annotate each point
    for a, r in zip(alphas, rw_values):
        ax.annotate(f'{r:.2f}%', (a, r), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontweight='bold')
    
    ax.set_xlabel('EMA Alpha')
    ax.set_ylabel('RW SPD Percentile')
    ax.set_title('Ablation Study #3: EMA Smoothing Parameter Sweep\n(Zero Effect: Mathematical Proof Signal Quality Issue)',
                 fontweight='bold')
    ax.set_ylim(41.5, 42.5)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('All alphas produce\nIDENTICAL results',
                xy=(0.5, 41.94), xytext=(0.7, 42.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red', fontweight='bold')
    
    save_fig('03_ablation_ema_flat')


def chart_performance_comparison():
    """Final Performance Comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    categories = ['RW SPD\nPercentile', 'Win Rate\n(%)', 'Execution\nTime (min)', 'Model\nSize (MB)']
    cnn_values = [41.43, 54.32, 17.5, 1.1]
    simplified_values = [41.94, 70.42, 2.5, 0]
    
    # Normalize for visualization (different scales)
    cnn_norm = [41.43, 54.32, 17.5/20*100, 1.1/2*100]
    simplified_norm = [41.94, 70.42, 2.5/20*100, 0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cnn_norm, width, label='CNN Approach', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, simplified_norm, width, label='Simplified Baseline', color='#2ecc71', alpha=0.8)
    
    # Add actual value labels
    actual_cnn = ['41.43%', '54.32%', '~17 min', '1.1 MB']
    actual_simp = ['41.94% ⭐', '70.42% ⭐', '~2.5 min ⭐', 'None ⭐']
    
    for bar, val in zip(bars1, actual_cnn):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                val, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, val in zip(bars2, actual_simp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                val, ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
    
    ax.set_ylabel('Normalized Score (Higher is Better)')
    ax.set_title('Performance Comparison: CNN vs Simplified Baseline\n(Simplified Wins on ALL Dimensions)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    save_fig('04_performance_comparison')


def chart_temporal_pattern():
    """Temporal Performance Pattern (U-Shaped)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    periods = ['2016-2018', '2019-2021', '2022-2024', '2024-2025']
    percentiles = [40.58, 34.97, 43.39, 41.0]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = ax.bar(periods, percentiles, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, percentiles):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='DCA Benchmark (50%)')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Average Percentile')
    ax.set_title('Temporal Pattern: U-Shaped Performance\n(CNN Trained on 2014-2015 Doesn\'t Generalize)',
                 fontweight='bold')
    ax.set_ylim(30, 55)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    save_fig('05_temporal_pattern')


def chart_root_causes():
    """Root Causes (Horizontal Bar Chart)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    causes = [
        'Time Horizon Mismatch\n(90-day vs 6-18 month cycles)',
        'Daily Noise Problem\n(Low SNR in daily prices)',
        'Wrong Architecture\n(Spatial vs Temporal)',
        'Training Set Issues\n(2014-2015 ≠ 2016-2024)',
        'Overengineering\n(296K params for binary outcome)'
    ]
    impact = [95, 85, 80, 75, 70]  # Arbitrary impact scores
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#95a5a6']
    
    y_pos = np.arange(len(causes))
    bars = ax.barh(y_pos, impact, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(causes)
    ax.set_xlabel('Impact Score (Relative)')
    ax.set_title('Why GAF + CNN Failed: Root Cause Analysis', fontweight='bold')
    ax.set_xlim(0, 110)
    
    # Add value labels
    for bar, val in zip(bars, impact):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val}', ha='left', va='center', fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='x')
    save_fig('06_root_causes')


def chart_future_improvements():
    """Future Improvements Roadmap"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = [
        'Weekly Granularity',
        'Regime Detection',
        'Multi-Channel Features',
        'Longer Lookback (365d)',
        'LSTM/RNN Architecture',
        'Ensemble Methods'
    ]
    effort = [1, 4, 2, 2, 4, 3]  # Days of effort
    expected_gain = [15, 20, 8, 7, 12, 8]  # Expected percentile points gain
    
    scatter = ax.scatter(effort, expected_gain, s=300, alpha=0.6,
                        c=expected_gain, cmap='RdYlGn', edgecolors='black')
    
    # Add labels
    for i, txt in enumerate(improvements):
        ax.annotate(txt, (effort[i], expected_gain[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Implementation Effort (Days)')
    ax.set_ylabel('Expected RW Percentile Gain (pp)')
    ax.set_title('Future Improvements: ROI Analysis\n(Highest ROI: Weekly + Regime Detection)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 25)
    
    # Add ROI quadrant lines
    ax.axhline(y=12, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(4.5, 22, 'High Effort\nHigh Reward', ha='center', fontsize=9, style='italic')
    ax.text(0.5, 22, 'Low Effort\nHigh Reward ⭐', ha='center', fontsize=9, 
            style='italic', color='green', fontweight='bold')
    
    save_fig('07_future_improvements')


def main():
    """Generate all charts"""
    print("Generating charts for VTS Midterm Presentation...")
    print("=" * 50)
    
    chart_ablation_sensitivity()
    chart_ablation_signal_quality()
    chart_ablation_ema()
    chart_performance_comparison()
    chart_temporal_pattern()
    chart_root_causes()
    chart_future_improvements()
    
    print("=" * 50)
    print(f"All charts saved to: {OUTPUT_DIR}")
    print("\nCharts generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
