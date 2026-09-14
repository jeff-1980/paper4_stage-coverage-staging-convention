"""Redraws all 3 v4 figures: data/per_unit_clocks_final_v4.csv (from
build_canonical.py) and tables/A2_decile_v4.csv (from aggregate.py, run that
first)."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
DATA = REPO / 'data'
TABLES = REPO / 'tables'
FIGURES = REPO / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9})
FDS = ['FD001', 'FD002', 'FD003', 'FD004']


def fig_clocks():
    clocks = pd.read_csv(DATA / 'per_unit_clocks_final_v4.csv')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(clocks['t_knee_norm'].dropna(), bins=30, alpha=0.6,
            label='t_knee/T (RUL-clip knee)', density=True)
    ax.hist(clocks['t_cp_norm'], bins=30, alpha=0.6,
            label='t_cp/T (CUSUM change point, h=7,k=0.7)', density=True)
    ax.set_xlabel('Normalized position within unit trajectory')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_clocks_v4.pdf')
    fig.savefig(FIGURES / 'fig_clocks_v4.png', dpi=150)
    plt.close(fig)
    print('wrote fig_clocks_v4.pdf')


def fig_residual_deciles():
    df = pd.read_csv(TABLES / 'A2_decile_v4.csv')
    ymax = df['median_s_t'].max() * 1.08
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=True, sharey=True)
    for ax, fd in zip(axes.flat, FDS):
        sub = df[df['fd'] == fd].sort_values('decile')
        ax.plot(sub['decile'], sub['median_s_t'], marker='o', markersize=3, linewidth=1.2)
        ax.set_title(fd, fontsize=9)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel('Lifetime fraction decile')
    for ax in axes[:, 0]:
        ax.set_ylabel('Median |s_t| (Split CP)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_residual_deciles_v4.pdf')
    fig.savefig(FIGURES / 'fig_residual_deciles_v4.png', dpi=150)
    plt.close(fig)
    print('wrote fig_residual_deciles_v4.pdf')


def fig_failure_deciles():
    df = pd.read_csv(TABLES / 'A2_decile_v4.csv')
    ymax = df['failure_rate'].max() * 1.08
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=True, sharey=True)
    for ax, fd in zip(axes.flat, FDS):
        sub = df[df['fd'] == fd].sort_values('decile')
        ax.plot(sub['decile'], sub['failure_rate'], marker='o', markersize=3, linewidth=1.2, color='C3')
        ax.set_title(fd, fontsize=9)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel('Lifetime fraction decile')
    for ax in axes[:, 0]:
        ax.set_ylabel('Split CP failure rate (1-ECR)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_failure_deciles_v4.pdf')
    fig.savefig(FIGURES / 'fig_failure_deciles_v4.png', dpi=150)
    plt.close(fig)
    print('wrote fig_failure_deciles_v4.pdf')


if __name__ == '__main__':
    fig_clocks()
    fig_residual_deciles()
    fig_failure_deciles()
