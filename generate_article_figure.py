"""Generate Figure 1 for Medium/Substack articles: Strategy Performance Comparison."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

strategies = ['CNN + GAF\n(Deep Learning)', 'Neutral DCA\n(Baseline)', 'OLS Z-Score\n(Regime-Adaptive)']
rw_pct = [41.43, 41.94, 44.95]
win_rate = [54.32, 70.42, 66.94]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Colors: CNN=muted red, Neutral=gray, OLS=teal
colors = ['#d97373', '#999999', '#2a9d8f']
edge_colors = ['#b85555', '#777777', '#1f7a6e']

# Panel 1: RW Percentile
bars1 = ax1.bar(strategies, rw_pct, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.6)
ax1.set_ylabel('Rolling Window Percentile (%)', fontsize=11)
ax1.set_title('Tournament Metric (RW-SPD)', fontsize=13, fontweight='bold')
ax1.set_ylim(38, 47)
ax1.axhline(y=41.94, color='#999999', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(2.35, 42.1, 'neutral baseline', fontsize=8, color='#777777', style='italic')
for bar, val in zip(bars1, rw_pct):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 2: Win Rate
bars2 = ax2.bar(strategies, win_rate, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.6)
ax2.set_ylabel('Win Rate vs DCA (%)', fontsize=11)
ax2.set_title('Win Rate (% of Windows Beating DCA)', fontsize=13, fontweight='bold')
ax2.set_ylim(45, 78)
ax2.axhline(y=50, color='#cccccc', linestyle=':', alpha=0.5, linewidth=1)
ax2.text(2.35, 50.5, 'coin flip', fontsize=8, color='#aaaaaa', style='italic')
for bar, val in zip(bars2, win_rate):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Annotations
ax1.annotate('+3.01 pp', xy=(2, 44.95), xytext=(2, 46.2),
            fontsize=10, fontweight='bold', color='#1f7a6e', ha='center',
            arrowprops=dict(arrowstyle='->', color='#1f7a6e', lw=1.5))

fig.suptitle('Bitcoin Accumulation Strategy Comparison (2016–2025)',
             fontsize=14, fontweight='bold', y=1.02)
fig.text(0.5, -0.02, 'Evaluated across 3,076 rolling windows on sats-per-dollar metric | Trilemma Foundation Tournament',
         ha='center', fontsize=9, color='#666666', style='italic')

plt.tight_layout()
plt.savefig('/Users/bobkatz/Visual_Trading_System/final_report/figures/article_strategy_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: final_report/figures/article_strategy_comparison.png")
