"""
Redraw fig_residual_deciles.pdf from the already-computed
results/forensics_2026-09-14_relabel/residual_by_decile_agg.csv (produced by
clocks_and_residuals.py::part_c_residual_curve). Pure replot, no rerun of any
model/inference/analysis step.

Requested layout: 2x2 facets (FD001-FD004), 6.5x4.5in, 9pt font, shared/unified
y-axis range, x-label "Lifetime fraction decile", y-label "Median |s_t| (Split CP)",
no suptitle. Overwrites the existing fig_residual_deciles.pdf/.png in place.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results' / 'forensics_2026-09-14_relabel'

plt.rcParams.update({'font.size': 9})

agg = pd.read_csv(OUT / 'residual_by_decile_agg.csv')

fds = ['FD001', 'FD002', 'FD003', 'FD004']
ymax = agg['median_abs_s'].max() * 1.08
ymin = 0.0

fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=True, sharey=True)

for ax, fd in zip(axes.flat, fds):
    sub = agg[agg['fd'] == fd].sort_values('bin')
    deciles = sub['bin'].to_numpy() + 1  # 1-indexed decile number
    ax.plot(deciles, sub['median_abs_s'], marker='o', markersize=3, linewidth=1.2)
    ax.set_title(fd, fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel('Lifetime fraction decile')
for ax in axes[:, 0]:
    ax.set_ylabel('Median |s_t| (Split CP)')

fig.tight_layout()
fig.savefig(OUT / 'fig_residual_deciles.pdf')
fig.savefig(OUT / 'fig_residual_deciles.png', dpi=150)
plt.close(fig)
print('wrote', OUT / 'fig_residual_deciles.pdf')
