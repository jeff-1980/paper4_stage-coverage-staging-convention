"""
Forensics 2026-09-14 (follow-up): per-unit "clock" comparison + residual-vs-t_norm curve.

(a) RUL-clip knee position t_knee/T_unit: analytic, no rerun needed. The underlying rul
    column is `(max_cycle - cycle).clip(upper=125)` (verified against
    item2_six_method_comparison.py::load_split_and_scaler, line
    "train_df['rul'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)").
    Combined with the fixed window stride=1 and SEQ=30 used by predict_engine, sample
    index t (0-indexed within a unit's n_unit-length eval array, as saved in
    per_sample_relabeled.csv) has raw (unclipped) remaining life = n_unit-1-t exactly.
    The knee -- where clipping stops binding -- is at raw_remaining(t)=125, i.e.
    t_knee = n_unit-126 for n_unit>126 (else the whole observed window is already past
    the knee; reported as NaN and counted separately, not silently zeroed).
    Computed directly from results/forensics_2026-09-14_relabel/per_sample_relabeled.csv
    (n_unit column) -- no rerun.

(b) CUSUM(h=7) change-point position t_cp/T_unit: also read directly from the existing
    per_sample_relabeled.csv's stage_cusum column (h_mult==7.0, any method -- stage_cusum
    is identical across methods at fixed h since it derives from the shared base model's
    scores, not from any one method's interval). t_cp = count of samples with
    stage_cusum != 'late' for that unit, since assign_stage() (conformal/mondrian_cp.py:29)
    defines late as [t_cp, n_total) -- i.e. every t < t_cp is early or middle.

(c) Split CP standardized residual |s_t| by t_norm decile, per FD: NOT recoverable from
    the saved CSV (only interval widths/coverage were saved, not raw mu/sigma). Requires a
    light rerun -- but only base-model MC-Dropout inference on eval units, no calibration
    state, no Mondrian buffers (much cheaper than the full relabel run). Reuses the
    already-trained weights from forensics_2026-09-11/models/, no retraining.

Does not modify main.tex or any existing script/paper-text.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'forensics_2026-09-11'))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import item2_six_method_comparison as i2

REPO = i2.REPO
MODELS = REPO / 'results' / 'forensics_2026-09-11' / 'models'
IN_CSV = REPO / 'results' / 'forensics_2026-09-14_relabel' / 'per_sample_relabeled.csv'
OUT = REPO / 'results' / 'forensics_2026-09-14_relabel'
FDS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [0, 1, 2, 3, 4]
RUL_CAP = 125


def part_ab_clocks():
    df = pd.read_csv(IN_CSV)
    sub = df[(df['method'] == 'Split CP') & (df['h_mult'] == 7.0)].copy()

    # one row per (fd,seed,unit): n_unit, t_cp (count of non-late samples)
    unit_rows = []
    for (fd, seed, unit), g in sub.groupby(['fd', 'seed', 'unit']):
        n_unit = int(g['n_unit'].iloc[0])
        t_cp = int((g['stage_cusum'] != 'late').sum())
        if n_unit > RUL_CAP + 1:
            t_knee = n_unit - RUL_CAP - 1
        else:
            t_knee = np.nan
        unit_rows.append(dict(fd=fd, seed=seed, unit=unit, n_unit=n_unit,
                               t_knee=t_knee, t_cp=t_cp,
                               t_knee_frac=(t_knee / n_unit if not np.isnan(t_knee) else np.nan),
                               t_cp_frac=t_cp / n_unit))
    units = pd.DataFrame(unit_rows)
    units.to_csv(OUT / 'unit_clocks.csv', index=False)

    print('=== t_knee/T_unit and t_cp/T_unit distributions, by FD ===')
    summary_rows = []
    for fd in FDS:
        u = units[units['fd'] == fd]
        n_no_knee = u['t_knee_frac'].isna().sum()
        for label, col in [('t_knee/T', 't_knee_frac'), ('t_cp/T', 't_cp_frac')]:
            vals = u[col].dropna()
            row = dict(fd=fd, metric=label, n=len(vals),
                       mean=vals.mean(), median=vals.median(),
                       q1=vals.quantile(0.25), q3=vals.quantile(0.75))
            summary_rows.append(row)
            print(f'{fd} {label}: n={row["n"]} mean={row["mean"]:.3f} median={row["median"]:.3f} '
                  f'Q1={row["q1"]:.3f} Q3={row["q3"]:.3f}'
                  + (f'  (n_units_with_no_knee={n_no_knee})' if label == 't_knee/T' else ''))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / 'clocks_summary.csv', index=False)

    # dual histogram, pooled across FD/seed/unit (single figure, not faceted -- per instruction)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 31)
    ax.hist(units['t_knee_frac'].dropna(), bins=bins, alpha=0.55, label=r'$t_{knee}/T_{unit}$ (RUL-clip knee)', color='#5DADE2')
    ax.hist(units['t_cp_frac'].dropna(), bins=bins, alpha=0.55, label=r'$t_{cp}/T_{unit}$ (CUSUM, $h=7\sigma_0$)', color='#E74C3C')
    ax.set_xlabel('Normalized position within held-out trajectory')
    ax.set_ylabel('Count (units, pooled over FD/seed)')
    ax.set_title('Two notions of "when does the interesting part of life start"')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_clocks.pdf')
    fig.savefig(OUT / 'fig_clocks.png', dpi=150)
    plt.close(fig)
    print(f'\nwrote fig_clocks.pdf (n_units total={len(units)}, '
          f'n_units_with_no_rul_knee={units["t_knee_frac"].isna().sum()})')
    return units, summary


def part_c_residual_curve():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    records = []

    for fd in FDS:
        for seed in SEEDS:
            tag = f'{fd}_seed{seed}'
            meta_path = MODELS / f'meta_{tag}.json'
            weight_path = MODELS / f'best_{tag}.pt'
            if not meta_path.exists() or not weight_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            dfu, model_train_units, cal_units, eval_units, split_hash = i2.load_split_and_scaler(fd, seed)
            assert split_hash == meta['split_hash']

            model = i2.BayesianLSTM(input_dim=len(i2.SENSOR_COLS)).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()

            for u in eval_units:
                s = i2.get_engine_series(dfu, u, model, device)
                if s is None:
                    continue
                mus, sigs, y_true = s
                n = len(mus)
                scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
                t_norm = (np.arange(n) + 1) / n
                bin_idx = np.clip(np.digitize(t_norm, bin_edges) - 1, 0, n_bins - 1)
                for b in range(n_bins):
                    mask = bin_idx == b
                    if mask.sum() == 0:
                        continue
                    records.append(dict(fd=fd, seed=seed, unit=int(u), bin=b,
                                         median_abs_s=float(np.median(scores[mask])),
                                         n=int(mask.sum())))
            print(f'[{tag}] residual-decile pass done, n_eval_units={len(eval_units)}')

    rec_df = pd.DataFrame(records)
    rec_df.to_csv(OUT / 'residual_by_decile_raw.csv', index=False)

    # aggregate: median-of-medians per (fd, bin) across seeds/units (weighted by n within
    # each unit's own median is already a per-unit summary; take the across-unit median)
    agg = rec_df.groupby(['fd', 'bin'])['median_abs_s'].median().reset_index()
    agg.to_csv(OUT / 'residual_by_decile_agg.csv', index=False)

    # faceted-by-FD line plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for ax, fd in zip(axes, FDS):
        sub = agg[agg['fd'] == fd].sort_values('bin')
        vals = [sub[sub['bin'] == b]['median_abs_s'].values[0] if b in sub['bin'].values else np.nan
                for b in range(n_bins)]
        ax.plot(bin_centers, vals, marker='o', color='#2E86C1')
        ax.set_title(fd)
        ax.set_xlabel(r'$t_{norm}$ decile')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'Median $|s_t|$ (Split CP nonconformity score)')
    fig.suptitle('Standardized residual magnitude across normalized lifetime position, by dataset')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_residual_deciles.pdf')
    fig.savefig(OUT / 'fig_residual_deciles.png', dpi=150)
    plt.close(fig)
    print('\nwrote fig_residual_deciles.pdf')

    print('\n=== median |s_t| by t_norm decile, by FD ===')
    for fd in FDS:
        sub = agg[agg['fd'] == fd].sort_values('bin')
        print(f'{fd}: ' + ' '.join(f'{v:.3f}' for v in sub['median_abs_s'].values))

    return agg


if __name__ == '__main__':
    part_ab_clocks()
    part_c_residual_curve()
