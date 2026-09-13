"""
Forensics 2026-09-11 — Item 1: headline claim A reproduction.

Reproduces the original Table 1 methodology EXACTLY as implemented in
experiments/run_c1.py + conformal/standard_cp.py::conditional_ecr:
  - stage assignment = FIXED normalized-lifetime split (t_norm<0.33 / <0.67 / >=0.67),
    t_norm = 1 - y_true/AVG_LIFETIME[fd]  (AVG_LIFETIME hardcoded, verified against
    actual train-set mean max_cycle: FD001=206.3, FD002=206.8, FD003=247.2, FD004=246.0
    -- matches run_c1.py's dict to 1 decimal).
  - NOT CUSUM-based. This matches original code, not the paper's §4.3 description.
Run across seeds 0-4 using models trained by train_multiseed.py.

Also runs a secondary CUSUM-based sensitivity variant (item 3 rule: code value
h=7sigma0 differs from paper-stated h=10sigma0, so sensitivity across
h in {5,7,10} sigma0 is required) for comparison only -- not the primary claim-A test.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest

from conformal.standard_cp import split_cp_intervals, compute_scores
from conformal.mondrian_cp import detect_cp_cusum

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / 'results' / 'forensics_2026-09-11' / 'models'
OUT = REPO / 'results' / 'forensics_2026-09-11'
ALPHA = 0.10

AVG_LIFETIME = {'FD001': 206, 'FD002': 206, 'FD003': 247, 'FD004': 247}

FDS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [0, 1, 2, 3, 4]


def conditional_ecr_fixed(y_true, lo, hi, t_norm):
    covered = (y_true >= lo) & (y_true <= hi)
    mpiw = (hi - lo).mean()
    stages = {
        'early': t_norm < 0.33,
        'middle': (t_norm >= 0.33) & (t_norm < 0.67),
        'late': t_norm >= 0.67,
    }
    result = {'global': {'ecr': covered.mean(), 'mpiw': mpiw,
                          'n': len(y_true), 'n_covered': int(covered.sum())}}
    for name, mask in stages.items():
        if mask.sum() == 0:
            continue
        result[name] = {
            'ecr': covered[mask].mean(),
            'mpiw': (hi - lo)[mask].mean(),
            'n': int(mask.sum()),
            'n_covered': int(covered[mask].sum()),
        }
    return result


def cusum_stage_ecr(y_true, mu, sigma, lo, hi, h_mult):
    scores = np.minimum(np.abs(y_true - mu) / np.maximum(sigma, 1e-3), 10.0)
    t_cp = detect_cp_cusum(scores.tolist(), k_mult=0.7*h_mult/7.0, h_mult=h_mult)
    n = len(scores)
    idx = np.arange(n)
    covered = (y_true >= lo) & (y_true <= hi)
    stages = {
        'early': idx < t_cp // 2,
        'middle': (idx >= t_cp // 2) & (idx < t_cp),
        'late': idx >= t_cp,
    }
    out = {}
    for name, mask in stages.items():
        if mask.sum() == 0:
            continue
        out[name] = {'ecr': float(covered[mask].mean()), 'n': int(mask.sum())}
    return out, int(t_cp)


def run():
    rows = []
    cusum_rows = []
    for fd in FDS:
        for seed in SEEDS:
            tag = f'{fd}_seed{seed}'
            meta_path = MODELS / f'meta_{tag}.json'
            if not meta_path.exists():
                print(f'[{tag}] missing, skip')
                continue
            meta = json.loads(meta_path.read_text())

            mu = np.load(MODELS / f'mu_{tag}.npy')
            sigma = np.load(MODELS / f'sigma_{tag}.npy')
            y = np.load(MODELS / f'true_{tag}.npy')
            cal_mu = np.load(MODELS / f'cal_mu_{tag}.npy')
            cal_sigma = np.load(MODELS / f'cal_sigma_{tag}.npy')
            cal_y = np.load(MODELS / f'cal_true_{tag}.npy')

            cal_scores = compute_scores(cal_mu, cal_sigma, cal_y)
            lo, hi, q_hat = split_cp_intervals(cal_scores, mu, sigma, alpha=ALPHA)

            T = AVG_LIFETIME[fd]
            t_norm = 1.0 - np.clip(y / T, 0, 1)
            ecr_result = conditional_ecr_fixed(y, lo, hi, t_norm)

            for stage, r in ecr_result.items():
                n_covered = r['n_covered']
                try:
                    pval = binomtest(n_covered, r['n'], 1-ALPHA, alternative='less').pvalue
                except Exception:
                    pval = float('nan')
                rows.append({
                    'fd': fd, 'seed': seed, 'stage': stage,
                    'ecr': r['ecr'], 'mpiw': r['mpiw'], 'n': r['n'], 'pval': pval,
                    'split_hash': meta['split_hash'], 'git_commit': meta['git_commit'],
                    'q_hat': q_hat,
                })

            # h-sensitivity (item 3), primary staging method is fixed-t_norm above;
            # this CUSUM variant is informational only.
            for h_mult in [5.0, 7.0, 10.0]:
                stage_ecr, t_cp = cusum_stage_ecr(y, mu, sigma, lo, hi, h_mult)
                for stage, r in stage_ecr.items():
                    cusum_rows.append({
                        'fd': fd, 'seed': seed, 'h_mult': h_mult, 'stage': stage,
                        'ecr': r['ecr'], 'n': r['n'], 't_cp': t_cp,
                    })

            print(f'[{tag}] global_ecr={ecr_result["global"]["ecr"]:.3f} '
                  f'middle_ecr={ecr_result.get("middle",{}).get("ecr",float("nan")):.3f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'item1_standard_cp_multiseed.csv', index=False)

    dfc = pd.DataFrame(cusum_rows)
    dfc.to_csv(OUT / 'item1_h_sensitivity.csv', index=False)

    print('\n=== ITEM 1 SUMMARY (fixed-tnorm, matches original run_c1.py method) ===')
    mid = df[df['stage'] == 'middle']
    for fd in FDS:
        sub = mid[mid['fd'] == fd]
        print(f'{fd}: middle_ecr per seed = {sub.sort_values("seed")["ecr"].round(3).tolist()}  '
              f'mean={sub["ecr"].mean():.3f}  n_seeds_p<0.05={ (sub["pval"]<0.05).sum() }/5')

    print('\n=== H-SENSITIVITY (CUSUM-based staging, informational) ===')
    for h in [5.0, 7.0, 10.0]:
        sub = dfc[(dfc['h_mult'] == h) & (dfc['stage'] == 'middle')]
        print(f'h={h}sigma0: middle_ecr mean(all fd/seed)={sub["ecr"].mean():.3f}  '
              f'range=[{sub["ecr"].min():.3f},{sub["ecr"].max():.3f}]')


if __name__ == '__main__':
    run()
