"""
Stage 0-R canonical per-sample data source. Single construction of all four
methods' intervals, replacing per_sample_clean.csv (relabel_clean.py) and
per_sample_mechanism.csv (mechanism_clean.py), which independently re-ran
MC-Dropout inference and diverged because dropout draws were never seeded
(see inference_clean.py::deterministic_seed docstring for the root-cause trace:
same model, same unit, two independent calls to model.predict_mc gave mu
differing by up to 1.97 and sigma by up to 0.63, because dropout masks were
drawn from PyTorch's uncontrolled global RNG state).

WARM-UP / BUFFER / ORDERING RULES (also restated in README.md (repo root)):
  - MC-Dropout inference: model.predict_mc(n_samples=50), dropout draws seeded
    via inference_clean.deterministic_seed(fd, seed, unit) = fd_num*1e7 +
    seed*1e5 + unit -- called immediately before EVERY get_engine_series_clean
    invocation (cal AND eval units alike), making every unit's mu/sigma
    reproducible regardless of call order or process. mu/sigma are computed
    ONCE per (fd,seed,unit) and reused for all downstream h values -- MC-Dropout
    and Split CP intervals never depend on h at all.
  - CUSUM: k_mult=0.7 held FIXED across h in {5.0, 7.0, 10.0} (NOT the
    linked rule k=0.7*h/7 used in earlier forensics rounds). h=7.0 is primary
    (unsuffixed columns); h=5.0/10.0 give the _h5/_h10 suffixed columns for
    the h-sensitivity table. detect_cp_cusum from conformal.mondrian_cp,
    unmodified logic, called once per (unit, h).
  - Split CP: single calibration quantile q_hat from ALL cal_units' scores
    pooled (np.quantile, method='higher'), alpha=0.10. Independent of h.
  - Mondrian CP warm-up: for EACH h in {5,7,10}, ONE MondirianCP(alpha=0.10,
    W_max=200, total_lifetime_est=AVG_LIFETIME[fd]) online-base instance and
    one separate frozen-base instance are warmed by feeding every cal_unit's
    (mu,sigma,y_true) triple with k_mult=0.7 fixed, IN THE ORDER cal_units
    APPEARS IN meta_{fd}_seed{seed}.json (the order fixed by train_clean.py's
    original random split). 6 total warmed instances per (fd,seed): {online,
    frozen} x {h=5,7,10}.
  - Mondrian online: for each h, the SAME warmed instance is shared and
    continuously updated (.update() after every .predict_interval()) across
    ALL eval_units, IN THE ORDER eval_units APPEARS IN meta_*.json. Order
    matters for this mode by construction and is fixed as above.
  - Mondrian frozen: for each (eval unit, h), copy.deepcopy() that h's warmed
    frozen base fresh, call .predict_interval() only (never .update()). Order
    among eval units does not matter for this mode (independent deepcopy per
    unit, no shared mutable state).
  - Unbounded-interval threshold: np.isinf(width), width=hi-lo.
  - Per-engine aggregation excludes unbounded (is_inf) samples for the
    "finite-only" ECR number; a separate "unbounded-as-covered" number is also
    reported (aggregate.py::table_main), since an infinite-width interval
    covers any finite true value by definition.

Staging conventions written into every row:
  - stage_fixed         : per-unit t_norm=(t+1)/n, thresholds 0.33/0.67.
  - stage_cusum          : CUSUM at h=7 (primary), k=0.7.
  - stage_cusum_h5/_h10  : CUSUM at h=5/h=10, k=0.7 (same fixed k).
  - stage_trainpct       : each unit's raw cycle number vs. that SEED's own
    train_units' 33rd/67th raw max_cycle percentile (no eval-unit self-
    reference, no residuals).
  - stage_matched        : filled in a second pass (build_matched_labels)
    using each FD's pooled CUSUM(h=7) stage-occupancy proportions as FIXED
    chronological cut points applied to every unit's own trajectory.
"""
import sys, os, json, copy, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `models.*`/`conformal.*` resolve to this scripts/ dir

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import norm

from models.bayesian_lstm import BayesianLSTM
from inference_clean import (
    load_split_and_scaler_clean, get_engine_series_clean, deterministic_seed,
    assign_stage, AVG_LIFETIME, ALPHA, MODELS, REPO,
)
from conformal.mondrian_cp import detect_cp_cusum, MondirianCP

OUT = REPO / 'data'
OUT.mkdir(parents=True, exist_ok=True)

FDS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [0, 1, 2, 3, 4]
H_MULTS = [5.0, 7.0, 10.0]
H_PRIMARY = 7.0  # main table reads this h's columns (unsuffixed column names)
K_FIXED = 0.7    # k held fixed across all three h values -- NOT linked to h
SEQ = 30


def h_suffix(h):
    return '' if h == H_PRIMARY else f'_h{int(h)}'


def warm_mondrian_kfixed(cal_series_list, T_est, h_mult, k_mult=K_FIXED):
    mondrian = MondirianCP(alpha=ALPHA, W_max=200, total_lifetime_est=T_est)
    for mus, sigs, y_true in cal_series_list:
        scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
        t_cp = detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h_mult)
        n = len(mus)
        for t in range(n):
            stg = assign_stage(t, t_cp, n)
            mondrian.update(float(t), y_true[t], mus[t], sigs[t], stg)
    return mondrian


def run_mondrian_online_kfixed(mondrian, mus, sigs, y_true, h_mult, k_mult=K_FIXED):
    scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
    t_cp = detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h_mult)
    lo, hi, stages = [], [], []
    n = len(mus)
    for t in range(n):
        stg = assign_stage(t, t_cp, n)
        l, h = mondrian.predict_interval(float(t), mus[t], sigs[t], stg)
        mondrian.update(float(t), y_true[t], mus[t], sigs[t], stg)
        lo.append(l); hi.append(h); stages.append(stg)
    return np.array(lo), np.array(hi), stages, t_cp


def run_mondrian_frozen_kfixed(mondrian_warmed, mus, sigs, y_true, h_mult, k_mult=K_FIXED):
    mondrian = copy.deepcopy(mondrian_warmed)
    scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
    t_cp = detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h_mult)
    lo, hi, stages = [], [], []
    n = len(mus)
    for t in range(n):
        stg = assign_stage(t, t_cp, n)
        l, h = mondrian.predict_interval(float(t), mus[t], sigs[t], stg)
        lo.append(l); hi.append(h); stages.append(stg)
    return np.array(lo), np.array(hi), stages, t_cp


HEADER_COMMENT = """\
# Stage 0-R canonical per-sample data source (per_sample_final.csv).
# Generated by build_canonical.py -- see its module docstring and
# README.md (repo root) for the complete warm-up/buffer/ordering rules.
# Fixes the MC-Dropout seeding bug that caused per_sample_clean.csv
# (relabel_clean.py) and per_sample_mechanism.csv (mechanism_clean.py) to
# silently diverge on Mondrian CP (online) by ~0.01-0.02 ECR.
# CUSUM: k=0.7 fixed (NOT linked to h) across h in {5.0, 7.0, 10.0}.
# h=7.0 columns are unsuffixed (primary/main-table); h=5.0/10.0 columns carry
# _h5/_h10 suffixes (h-sensitivity table only).
# fixed_seed = deterministic_seed(fd, seed, unit) applied before every
# MC-Dropout draw (cal and eval units alike) -- see inference_clean.py.
"""


def detect_cusum_tracked(scores, h_mult, k_mult=K_FIXED):
    if len(scores) < 20:
        return len(scores) // 2, False
    mu0 = np.mean(scores[:10])
    sigma = np.std(scores[:10]) + 1e-8
    k = k_mult * sigma
    h = h_mult * sigma
    cusum = 0.0
    cp = len(scores) - 1
    triggered = False
    for i, s in enumerate(scores):
        cusum = max(0.0, cusum + (s - mu0) - k)
        if cusum > h:
            cp = i
            triggered = True
            break
    return cp, triggered


def fixed_stage_labels(n):
    t_norm = (np.arange(n) + 1) / n
    return np.where(t_norm < 0.33, 'early', np.where(t_norm < 0.67, 'middle', 'late'))


def process_fd_seed(fd, seed, device):
    tag = f'{fd}_seed{seed}'
    meta = json.loads((MODELS / f'meta_{tag}.json').read_text())
    df, train_u, val_u, cal_units, eval_units, split_hash = load_split_and_scaler_clean(fd, seed)
    assert split_hash == meta['split_hash']

    model = BayesianLSTM(input_dim=len(meta['scaler_stats']['data_min_'])).to(device)
    model.load_state_dict(torch.load(MODELS / f'best_{tag}.pt', map_location=device))
    model.eval()

    train_max_cycle = df[df['unit'].isin(train_u)].groupby('unit')['cycle'].max()
    p33 = float(np.percentile(train_max_cycle.values, 33))
    p67 = float(np.percentile(train_max_cycle.values, 67))

    cal_series = []
    for u in cal_units:
        s = get_engine_series_clean(df, u, model, device,
                                     fixed_seed=deterministic_seed(fd, seed, u))
        if s is not None:
            cal_series.append(s)
    cal_scores_all = np.concatenate([
        np.minimum(np.abs(y - m) / np.maximum(sg, 1e-3), 10.0) for m, sg, y in cal_series
    ])
    nC = len(cal_scores_all)
    q_hat = float(np.quantile(cal_scores_all, np.ceil((nC + 1) * (1 - ALPHA)) / nC, method='higher'))

    T_est = AVG_LIFETIME[fd]
    mondrian_online_base = {h: warm_mondrian_kfixed(cal_series, T_est, h) for h in H_MULTS}
    mondrian_frozen_base = {h: warm_mondrian_kfixed(cal_series, T_est, h) for h in H_MULTS}

    z = norm.ppf(1 - ALPHA / 2)

    rows = []
    clock_rows = []
    for u in eval_units:
        s = get_engine_series_clean(df, u, model, device,
                                     fixed_seed=deterministic_seed(fd, seed, u))
        if s is None:
            continue
        mus, sigs, y_true = s
        n = len(mus)
        sig_c = np.maximum(sigs, 1e-3)
        scores = np.minimum(np.abs(y_true - mus) / sig_c, 10.0)

        # stage_fixed, stage_trainpct: h-independent
        stage_fixed = fixed_stage_labels(n)
        sub = df[df['unit'] == u]
        cycles = sub['cycle'].values
        label_cycles = cycles[SEQ - 1:]
        assert len(label_cycles) == n
        stage_trainpct = np.where(label_cycles < p33, 'early',
                                   np.where(label_cycles < p67, 'middle', 'late'))

        # MC-Dropout, Split CP: h-independent
        lo_mc, hi_mc = mus - z * sig_c, mus + z * sig_c
        width_mc = hi_mc - lo_mc
        covered_mc = (y_true >= lo_mc) & (y_true <= hi_mc)

        lo_sp, hi_sp = mus - q_hat * sig_c, mus + q_hat * sig_c
        width_sp = hi_sp - lo_sp
        covered_sp = (y_true >= lo_sp) & (y_true <= hi_sp)

        # per-h: stage_cusum, Mondrian online/frozen
        per_h = {}
        for h in H_MULTS:
            t_cp, triggered = detect_cusum_tracked(scores.tolist(), h)
            stage_cusum_h = np.array([assign_stage(t, t_cp, n) for t in range(n)])

            lo_o, hi_o, _, _ = run_mondrian_online_kfixed(mondrian_online_base[h], mus, sigs, y_true, h)
            width_o = hi_o - lo_o
            covered_o = (y_true >= lo_o) & (y_true <= hi_o)

            lo_f, hi_f, _, _ = run_mondrian_frozen_kfixed(mondrian_frozen_base[h], mus, sigs, y_true, h)
            width_f = hi_f - lo_f
            covered_f = (y_true >= lo_f) & (y_true <= hi_f)

            per_h[h] = dict(stage_cusum=stage_cusum_h, t_cp=t_cp, triggered=triggered,
                             lo_o=lo_o, hi_o=hi_o, width_o=width_o, covered_o=covered_o,
                             lo_f=lo_f, hi_f=hi_f, width_f=width_f, covered_f=covered_f)

            if h == H_PRIMARY:
                t_knee_idx = n - 126 if n > 126 else np.nan
                t_knee_norm = t_knee_idx / n if not np.isnan(t_knee_idx) else np.nan
                clock_rows.append(dict(fd=fd, seed=seed, unit=int(u), n_unit=n,
                                        t_knee_idx=t_knee_idx, t_knee_norm=t_knee_norm,
                                        t_cp_idx=t_cp, t_cp_norm=t_cp / n, cusum_triggered=triggered))

        for t in range(n):
            row = dict(
                fd=fd, seed=seed, unit=int(u), t=t, n_unit=n, cycle=int(label_cycles[t]),
                mu=float(mus[t]), sigma=float(sig_c[t]), y_true=float(y_true[t]),
                s_t=float(scores[t]), t_norm=(t + 1) / n,
                stage_fixed=stage_fixed[t], stage_trainpct=stage_trainpct[t], stage_matched=None,
                q_hat=q_hat,
                lo_mc=float(lo_mc[t]), hi_mc=float(hi_mc[t]),
                width_mc=float(width_mc[t]), covered_mc=bool(covered_mc[t]),
                lo_sp=float(lo_sp[t]), hi_sp=float(hi_sp[t]),
                width_sp=float(width_sp[t]), covered_sp=bool(covered_sp[t]),
            )
            for h in H_MULTS:
                sfx = h_suffix(h)
                d = per_h[h]
                row[f'stage_cusum{sfx}'] = d['stage_cusum'][t]
                row[f'lo_mo{sfx}'] = float(d['lo_o'][t]); row[f'hi_mo{sfx}'] = float(d['hi_o'][t])
                row[f'width_mo{sfx}'] = float(d['width_o'][t]); row[f'covered_mo{sfx}'] = bool(d['covered_o'][t])
                row[f'is_inf_mo{sfx}'] = bool(np.isinf(d['width_o'][t]))
                row[f'lo_mf{sfx}'] = float(d['lo_f'][t]); row[f'hi_mf{sfx}'] = float(d['hi_f'][t])
                row[f'width_mf{sfx}'] = float(d['width_f'][t]); row[f'covered_mf{sfx}'] = bool(d['covered_f'][t])
                row[f'is_inf_mf{sfx}'] = bool(np.isinf(d['width_f'][t]))
            rows.append(row)

    return rows, clock_rows, dict(fd=fd, seed=seed, p33=p33, p67=p67)


def build_matched_labels(csv_path):
    """Second pass: pooled CUSUM(h=7) stage-occupancy proportions per FD, used
    as fixed chronological cut points on t_norm to fill stage_matched."""
    df = pd.read_csv(csv_path, comment='#')
    stage_matched = np.empty(len(df), dtype=object)
    props = {}
    for fd in FDS:
        sub = df[df['fd'] == fd]
        p_early = (sub['stage_cusum'] == 'early').mean()
        p_middle = (sub['stage_cusum'] == 'middle').mean()
        props[fd] = (p_early, p_middle)
        mask = (df['fd'] == fd).values
        t_norm = df.loc[mask, 't_norm'].values
        cum_early = p_early
        cum_middle = p_early + p_middle
        stage_matched[mask] = np.where(t_norm < cum_early, 'early',
                                        np.where(t_norm < cum_middle, 'middle', 'late'))
    df['stage_matched'] = stage_matched
    return df, props


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_path = OUT / 'per_sample_final_tmp.csv'
    final_path = OUT / 'per_sample_final.csv'
    clock_path = OUT / 'per_unit_clocks_final.csv'
    thresh_path = OUT / 'trainpct_thresholds_final.csv'
    if tmp_path.exists():
        tmp_path.unlink()

    first_write = True
    first_write_clock = True
    thresholds = []
    total = 0
    for fd in FDS:
        for seed in SEEDS:
            t0 = time.time()
            rows, clock_rows, thresh = process_fd_seed(fd, seed, device)
            pd.DataFrame(rows).to_csv(tmp_path, mode='a', header=first_write, index=False)
            pd.DataFrame(clock_rows).to_csv(clock_path, mode='a', header=first_write_clock, index=False)
            first_write = False
            first_write_clock = False
            thresholds.append(thresh)
            total += len(rows)
            print(f'[{fd}_seed{seed}] n_eval_units={len(clock_rows)} rows={len(rows)} '
                  f'done in {time.time()-t0:.0f}s, total rows so far={total}')

    pd.DataFrame(thresholds).to_csv(thresh_path, index=False)
    print(f'\nPhase 1 done: {total} rows -> {tmp_path}')

    df_final, props = build_matched_labels(tmp_path)
    with open(final_path, 'w') as f:
        f.write(HEADER_COMMENT)
    df_final.to_csv(final_path, mode='a', index=False)
    tmp_path.unlink()

    md5 = hashlib.md5(final_path.read_bytes()).hexdigest()
    print(f'Phase 2 done: stage_matched filled using CUSUM(h=7) occupancy {props}')
    print(f'Wrote {final_path}, {len(df_final)} rows, md5={md5}')
    with open(OUT / 'per_sample_final.md5', 'w') as f:
        f.write(md5 + '\n')
    return md5


if __name__ == '__main__':
    run()
