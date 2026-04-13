"""Generate table images for Substack article."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Table 1: Experiment Ledger ---
fig1, ax1 = plt.subplots(figsize=(13, 3.5))
ax1.axis('off')

cols = ['Experiment', 'RW Impact', 'Decision', 'Reason']
data = [
    ['OLS z-score (\u00b12\u03c3 clip)', '+3.01 pp', 'ADOPT', 'Clear improvement, regime-adaptive'],
    ['Turnover governor', '+0.00 pp', 'ADOPT', '~12% turnover reduction, operational stability'],
    ['Vol-amplitude scaling', '+0.14 pp', 'REJECT', 'Win rate degraded, SPD advantage shrank'],
    ['Weekly granularity', '-0.90 pp', 'REJECT', 'Worse than daily; gain was from z-window'],
    ['Fee sensitivity', 'N/A', 'DOCUMENT', 'Proved percentile-invariant'],
]

table1 = ax1.table(cellText=data, colLabels=cols, loc='center', cellLoc='left',
                   colWidths=[0.20, 0.10, 0.12, 0.58])
table1.auto_set_font_size(False)
table1.set_fontsize(11)
table1.scale(1.0, 1.8)

for (row, col), cell in table1.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2a9d8f')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f0f0f0')
    else:
        cell.set_facecolor('white')
    cell.set_edgecolor('#cccccc')
    if col == 2 and row > 0:
        cell.set_text_props(fontweight='bold')

fig1.suptitle('Experiment Ledger', fontsize=14, fontweight='bold', y=0.98)
fig1.tight_layout()
fig1.savefig('final_report/figures/table_experiment_ledger.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: table_experiment_ledger.png")
plt.close(fig1)

# --- Table 2: Performance Ladder ---
fig2, ax2 = plt.subplots(figsize=(9, 2.5))
ax2.axis('off')

cols2 = ['Configuration', 'RW Percentile', 'Win Rate']
data2 = [
    ['Neutral (constant prob=0.5)', '41.94%', '70.42%'],
    ['OLS raw (no normalization)', '43.25%', '91.48%'],
    ['OLS z-score winsorized', '44.95%', '66.94%'],
]

table2 = ax2.table(cellText=data2, colLabels=cols2, loc='center', cellLoc='left',
                   colWidths=[0.50, 0.25, 0.25])
table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1.0, 1.8)

# Center text in columns 1 and 2 (RW Percentile, Win Rate)
for (row, col), cell in table2.get_celld().items():
    if col in (1, 2):
        cell._loc = 'center'

for (row, col), cell in table2.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2a9d8f')
        cell.set_text_props(color='white', fontweight='bold')
    elif row == 3:
        cell.set_facecolor('#d4edda')
        cell.set_text_props(fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f0f0f0')
    else:
        cell.set_facecolor('white')
    cell.set_edgecolor('#cccccc')

fig2.suptitle('Performance Ladder', fontsize=14, fontweight='bold', y=0.98)
fig2.tight_layout()
fig2.savefig('final_report/figures/table_performance_ladder.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: table_performance_ladder.png")
plt.close(fig2)
