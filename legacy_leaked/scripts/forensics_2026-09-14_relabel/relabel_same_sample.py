"""
Forensics 2026-09-14: same-sample relabeling correction.

Problem being fixed: the 2026-09-11/09-12 "reversal" finding compared TWO DIFFERENT
evaluation populations under the two staging conventions --
  - fixed lifetime-fraction convention: item1, the OFFICIAL held-out test set
    (one point per unit, from cmapss_loader-style test_FD00X.txt)
  - CUSUM convention: item2, full-trajectory HELD-OUT EVAL UNITS (different units,
    different population, drawn from train_{fd}.txt via load_split_and_scaler)
That is not a controlled same-sample comparison -- it confounds "which convention" with
"which population." This script fixes it: BOTH staging conventions are computed on the
SAME per-sample intervals (three methods x 4 FD x 5 seeds), reusing the already-trained
weights from forensics_2026-09-11/models/ (no retraining -- pure relabeling + inference).

Per-timestep t_norm for the fixed convention is computed per-unit (not from a shared
per-FD average lifetime, and not from the model's clipped RUL target): for a held-out
eval unit with n prediction windows (t=0..n-1), we define RUL_true(t) = n-1-t (raw
remaining window count, unclipped) and T_unit = n, so t_norm(t) = 1 - RUL_true(t)/T_unit
= (t+1)/n. This gives a well-populated early/middle/late split for every unit (unlike the
official single-point test set, which structurally has no early points at all under any
similar convention using a shared clipped-RUL reference -- see appendix_official_test.tex).

Three methods only (no ACI/EnbPI): MC-Dropout, Split CP (='Standard CP' in the underlying
code), Mondrian CP (referred to as the third method throughout; this repository uses the
name Mondrian CP consistently). h in {5,7,10} for Mondrian CP; h=7 is primary/default
reported elsewhere, all three are computed here.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'forensics_2026-09-11'))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import norm

import item2_six_method_comparison as i2

REPO = i2.REPO
MODELS = REPO / 'results' / 'forensics_2026-09-11' / 'models'
OUT = REPO / 'results' / 'forensics_2026-09-14_relabel'
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10
H_MULTS = [5.0, 7.0, 10.0]
FDS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [0, 1, 2, 3, 4]


def fixed_stage_labels(n):
    """Per-unit t_norm = (t+1)/n for t=0..n-1 (see module docstring)."""
    t_norm = (np.arange(n) + 1) / n
    stages = np.where(t_norm < 0.33, 'early', np.where(t_norm < 0.67, 'middle', 'late'))
    return stages


def process_unit(mus, sigs, y_true, cal_scores_all, mondrians, fd, seed, unit):
    """Compute per-sample width/covered for 3 methods, and both stage labelings, for one
    held-out eval unit. Returns a list of per-sample dict rows."""
    n = len(mus)
    sig_c = np.maximum(sigs, 1e-3)

    # -- fixed lifetime-fraction stage labels (per-unit, unclipped) --
    stages_fixed = fixed_stage_labels(n)

    # -- CUSUM stage labels, one per h_mult (reuses the exact same detect_cp_cusum /
    #    assign_stage as elsewhere in the codebase) --
    scores = np.minimum(np.abs(y_true - mus) / sig_c, 10.0)
    stages_cusum = {}
    t_cp_by_h = {}
    for h in H_MULTS:
        k_mult = 0.7 * h / 7.0
        t_cp = i2.detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h)
        stages_cusum[h] = np.array(i2.stage_labels_from_tcp(n, t_cp))
        t_cp_by_h[h] = t_cp

    # -- MC-Dropout --
    z = norm.ppf(1 - ALPHA / 2)
    lo_mc, hi_mc = mus - z * sig_c, mus + z * sig_c
    width_mc = hi_mc - lo_mc
    covered_mc = (y_true >= lo_mc) & (y_true <= hi_mc)

    # -- Split CP --
    nC = len(cal_scores_all)
    q = np.quantile(cal_scores_all, np.ceil((nC + 1) * (1 - ALPHA)) / nC, method='higher')
    lo_sp, hi_sp = mus - q * sig_c, mus + q * sig_c
    width_sp = hi_sp - lo_sp
    covered_sp = (y_true >= lo_sp) & (y_true <= hi_sp)

    rows = []
    # MC-Dropout and Split CP have no h-dependence in their OWN interval construction, but
    # their reported per-stage ECR under the CUSUM convention still depends on h through the
    # BUCKETING alone (which samples land in which stage). Save one row per h so the
    # h-sensitivity table can cover all 3 methods, not just Mondrian CP.
    for t in range(n):
        base = dict(fd=fd, seed=seed, unit=int(unit), t=t, n_unit=n,
                     stage_fixed=stages_fixed[t])
        for h in H_MULTS:
            rows.append(dict(base, method='MC-Dropout', h_mult=h,
                              width=float(width_mc[t]), covered=bool(covered_mc[t]),
                              is_inf=bool(np.isinf(width_mc[t])), stage_cusum=stages_cusum[h][t]))
            rows.append(dict(base, method='Split CP', h_mult=h,
                              width=float(width_sp[t]), covered=bool(covered_sp[t]),
                              is_inf=bool(np.isinf(width_sp[t])), stage_cusum=stages_cusum[h][t]))

    # -- Mondrian CP, one pass per h_mult (each with its OWN warmed instance and its OWN
    #    CUSUM stage labels, since h changes both the calibration buffering and the stage
    #    assignment consistently) --
    for h in H_MULTS:
        mondrian = mondrians[h]
        lo_list, hi_list = [], []
        for t in range(n):
            stg = stages_cusum[h][t]
            l, hh = mondrian.predict_interval(float(t), mus[t], sigs[t], stg)
            mondrian.update(float(t), y_true[t], mus[t], sigs[t], stg)
            lo_list.append(l); hi_list.append(hh)
        lo_m, hi_m = np.array(lo_list), np.array(hi_list)
        width_m = hi_m - lo_m
        covered_m = (y_true >= lo_m) & (y_true <= hi_m)
        for t in range(n):
            rows.append(dict(fd=fd, seed=seed, unit=int(unit), t=t, n_unit=n,
                              stage_fixed=stages_fixed[t], method='Mondrian CP', h_mult=h,
                              width=float(width_m[t]), covered=bool(covered_m[t]),
                              is_inf=bool(np.isinf(width_m[t])), stage_cusum=stages_cusum[h][t]))
    return rows


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_rows = []
    for fd in FDS:
        for seed in SEEDS:
            tag = f'{fd}_seed{seed}'
            meta_path = MODELS / f'meta_{tag}.json'
            weight_path = MODELS / f'best_{tag}.pt'
            if not meta_path.exists() or not weight_path.exists():
                print(f'[{tag}] missing, skip')
                continue
            meta = json.loads(meta_path.read_text())

            df, model_train_units, cal_units, eval_units, split_hash = i2.load_split_and_scaler(fd, seed)
            assert split_hash == meta['split_hash']

            model = i2.BayesianLSTM(input_dim=len(i2.SENSOR_COLS)).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()

            cal_series = []
            cal_scores_all = []
            for u in cal_units:
                s = i2.get_engine_series(df, u, model, device)
                if s is None:
                    continue
                mus, sigs, y_true = s
                cal_series.append(s)
                sc = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
                cal_scores_all.extend(sc.tolist())
            cal_scores_all = np.array(cal_scores_all)

            T_est = i2.AVG_LIFETIME[fd]
            mondrians = {h: i2.warm_mondrian(cal_series, T_est, h) for h in H_MULTS}

            for u in eval_units:
                s = i2.get_engine_series(df, u, model, device)
                if s is None:
                    continue
                mus, sigs, y_true = s
                rows = process_unit(mus, sigs, y_true, cal_scores_all, mondrians, fd, seed, u)
                all_rows.extend(rows)

            print(f'[{tag}] n_eval_units={len(eval_units)} done, running total rows={len(all_rows)}')

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT / 'per_sample_relabeled.csv', index=False)
    print(f'\nSaved {len(df_out)} per-sample rows -> per_sample_relabeled.csv')


if __name__ == '__main__':
    run()
