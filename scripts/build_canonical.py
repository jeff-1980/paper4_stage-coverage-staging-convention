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
  - CUSUM (Stage 0-R2 fix): baseline window is the first max(floor(0.2*n),20)
    scores of a unit's OWN trajectory (was hardcoded scores[:10] before this
    round). An instance that never crosses the threshold now returns cp=n
    exactly (was cp=n-1) -- assign_stage(t, n, n) leaves the LATE bucket
    genuinely empty for that instance, rather than containing one leftover
    timestep. k_mult=0.7 held FIXED across h in {5.0, 7.0, 10.0} (NOT the
    linked rule k=0.7*h/7 used in earlier forensics rounds). h=7.0 is primary
    (unsuffixed columns); h=5.0/10.0 give the _h5/_h10 suffixed columns for
    the h-sensitivity table. detect_cp_cusum is now the ONLY implementation
    (conformal.mondrian_cp) -- this file's own former duplicate loop
    (detect_cusum_tracked) is now a thin wrapper around it, closing off the
    exact kind of two-implementations drift this rebuild fixed for MC-Dropout.
  - Split CP: single calibration quantile q_hat from ALL cal_units' scores
    pooled (np.quantile, method='higher'), alpha=0.10. Independent of h.
  - Mondrian CP online: for EACH h in {5,7,10}, ONE MondirianCP(alpha=0.10,
    W_max=200, total_lifetime_est=AVG_LIFETIME[fd]) instance is warmed by
    feeding every cal_unit's (mu,sigma,y_true) triple with k_mult=0.7 fixed,
    IN THE ORDER cal_units APPEARS IN meta_{fd}_seed{seed}.json (the order
    fixed by train_clean.py's original random split). That SAME warmed
    instance is then shared and continuously updated (.update() after every
    .predict_interval()) across ALL eval_units, IN THE ORDER eval_units
    APPEARS IN meta_*.json -- order matters for this mode by construction.
    Internally, AdaptiveLambdaCP's predict_interval() appends a pseudo-score
    of +inf with unnormalized weight 1.0 to the exponentially-decaying buffer
    weights before taking the weighted quantile (see
    conformal/adaptive_lambda_cp.py::_weights/predict_interval) -- this is
    what produces genuinely unbounded intervals when the real buffer's
    weighted mass at the target quantile level is thin (small/young buffer,
    or a stage whose calibration scores are unusually tight).
  - Mondrian CP (split) [Stage 0-R2 redefinition, was 'frozen']: NOT an
    online/adaptive instance at all. build_mondrian_split_quantiles() fills
    each of the 3 stage buckets ONCE from ALL cal_units' scores (uniform
    weight, no lambda, no time index), then takes the standard split-CP
    empirical quantile ceil((m+1)(1-alpha))/m per stage. run_mondrian_split()
    applies that fixed per-stage quantile to every eval sample -- no
    .update(), no rollback, no shared mutable state; eval-unit order is
    provably irrelevant. The previous 'frozen' definition (deepcopy of the
    online AdaptiveLambdaCP instance, predict-only) was still an
    adaptive/exponentially-weighted machine underneath; this is the
    classical, non-adaptive Mondrian conformal predictor the name implies.
  - Unbounded-interval threshold: np.isinf(width), width=hi-lo. For Mondrian
    (split), width is inf when a stage's calibration bucket is empty or its
    quantile level exceeds 1 (standard split-CP small-sample rule).
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


def build_mondrian_split_quantiles(cal_series_list, h_mult, k_mult=K_FIXED, alpha=ALPHA):
    """'Mondrian CP (split)': standard (non-adaptive) Mondrian conformal
    prediction. Calibration nonconformity scores are partitioned into 3 stage
    buckets (via CUSUM on each cal unit's OWN trajectory, same detect_cp_cusum
    as everywhere else), each bucket filled ONCE from ALL calibration units
    (uniform weight -- every calibration score in a stage counts equally, no
    exponential decay, no lambda, no time-index rollback), then the standard
    split-CP empirical quantile ceil((m+1)(1-alpha))/m (method='higher') is
    computed per stage. This REPLACES the previous 'frozen' definition
    (deepcopy of the online AdaptiveLambdaCP instance, predict-only) -- that
    was still an online/adaptive machine underneath, just not fed eval-time
    feedback; this is genuinely the classical, non-adaptive Mondrian split CP.
    Returns {stage: q_hat}, q_hat=np.inf if a stage's calibration bucket is
    empty or the quantile level exceeds 1 (standard split-CP small-sample rule)."""
    buckets = {'early': [], 'middle': [], 'late': []}
    for mus, sigs, y_true in cal_series_list:
        scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
        t_cp = detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h_mult)
        n = len(mus)
        for t in range(n):
            stg = assign_stage(t, t_cp, n)
            buckets[stg].append(scores[t])
    q = {}
    for stg in ['early', 'middle', 'late']:
        m = len(buckets[stg])
        if m == 0:
            q[stg] = np.inf
            continue
        level = np.ceil((m + 1) * (1 - alpha)) / m
        q[stg] = np.inf if level > 1.0 else float(np.quantile(np.array(buckets[stg]), level, method='higher'))
    return q


def run_mondrian_split(q_by_stage, mus, sigs, y_true, h_mult, k_mult=K_FIXED):
    """Applies the fixed, once-computed per-stage quantile to every eval
    sample. No .update(), no time-index rollback, no weighting -- order among
    eval units or timesteps does not matter (verified by construction: this
    function has no mutable state at all)."""
    scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
    t_cp = detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h_mult)
    n = len(mus)
    lo, hi, stages = [], [], []
    for t in range(n):
        stg = assign_stage(t, t_cp, n)
        q = q_by_stage[stg]
        lo.append(mus[t] - q * sigs[t])
        hi.append(mus[t] + q * sigs[t])
        stages.append(stg)
    return np.array(lo), np.array(hi), stages, t_cp


HEADER_COMMENT = """\
# Stage 0-R2 canonical per-sample data source (per_sample_final_v2.csv).
# Generated by build_canonical.py -- see its module docstring and
# README.md (repo root) for the complete warm-up/buffer/ordering rules.
# v2 vs v1 (per_sample_final.csv, superseded but kept on disk): three fixes,
# no retraining --
#   1. MC-Dropout total variance corrected to mean(sigma_b^2) + Var(mu_b)
#      (models/bayesian_lstm.py::predict_mc; was sigma_mean^2 + Var(mu_b)).
#   2. CUSUM baseline window widened to max(floor(0.2n),20) (was hardcoded
#      [:10]); an instance that never triggers now gets cp=n exactly, i.e. a
#      genuinely empty late bucket (was cp=n-1, one leftover late timestep).
#   3. 'Mondrian CP (frozen)' redefined as 'Mondrian CP (split)': a classical,
#      non-adaptive Mondrian split conformal predictor (fill each stage's
#      calibration bucket once, uniform weight, standard empirical quantile)
#      -- replaces the old definition, which was a deepcopy of the online
#      AdaptiveLambdaCP instance and still adaptive/exponentially-weighted
#      underneath even though it wasn't fed eval-time feedback. Column names
#      (mf/covered_mf/etc.) are unchanged; what they mean is not.
# CUSUM: k=0.7 fixed (NOT linked to h) across h in {5.0, 7.0, 10.0}.
# h=7.0 columns are unsuffixed (primary/main-table); h=5.0/10.0 columns carry
# _h5/_h10 suffixes (h-sensitivity table only).
# fixed_seed = deterministic_seed(fd, seed, unit) applied before every
# MC-Dropout draw (cal and eval units alike) -- see inference_clean.py.
"""


def detect_cusum_tracked(scores, h_mult, k_mult=K_FIXED):
    """Thin wrapper around conformal.mondrian_cp.detect_cp_cusum (the shared,
    single-implementation detector) that also reports whether the threshold
    was ever crossed. Previously this function carried its OWN duplicate
    CUSUM loop (hardcoded baseline window scores[:10], fallback cp=n-1) that
    could silently drift out of sync with the shared implementation used by
    the Mondrian warm/online/frozen(split) functions -- exactly the kind of
    two-implementations bug this rebuild is fixing elsewhere. Now: single
    implementation, called once."""
    n = len(scores)
    cp = detect_cp_cusum(scores, k_mult=k_mult, h_mult=h_mult)
    triggered = cp < n
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
    mondrian_split_q = {h: build_mondrian_split_quantiles(cal_series, h) for h in H_MULTS}

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

            lo_f, hi_f, _, _ = run_mondrian_split(mondrian_split_q[h], mus, sigs, y_true, h)
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
    tmp_path = OUT / 'per_sample_final_v2_tmp.csv'
    final_path = OUT / 'per_sample_final_v2.csv'
    clock_path = OUT / 'per_unit_clocks_final_v2.csv'
    thresh_path = OUT / 'trainpct_thresholds_final_v2.csv'
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
    with open(OUT / 'per_sample_final_v2.md5', 'w') as f:
        f.write(md5 + '\n')
    return md5


if __name__ == '__main__':
    run()
