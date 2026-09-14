"""
Stage 0-R, step 2: unified/fixed inference library, built on the clean (leak-free)
models trained by train_clean.py.

Fixes applied vs. the old experiments/forensics_2026-09-11/item2_six_method_comparison.py
(verification item 5):
  1. MC-dropout: unified to n_samples=50 (matches models.bayesian_lstm.BayesianLSTM's
     own predict_mc default) by directly calling model.predict_mc() on a BATCHED
     tensor of all of a unit's sliding windows, instead of the old per-timestep
     python loop that called model() twice per mc-sample per timestep (mc=15,
     2x15=30 forward passes per timestep -- also much slower).
  2. Variance decomposition: predict_mc() already computes
     sigma_total = sqrt(sigma_mean**2 + var(mu_samples)), i.e. aleatoric mean +
     epistemic (between-sample) variance of the mean prediction. Reusing it means
     this is now correct by construction, not reimplemented.
  3. Window/label alignment: the old get_engine_series paired window k (covering
     raw rows k..k+SEQ-1) with label y_u[SEQ+k] (row k+SEQ) -- one row past the
     window's own last frame, AND only generated L-SEQ windows total instead of
     the L-SEQ+1 that experiments/forensics_2026-09-11/train_multiseed.py's
     extract_sequences() produces (i.e. it silently dropped the final window,
     the one ending exactly at the unit's last observed row). Fixed here to
     mirror extract_sequences() exactly: window k covers rows k..k+SEQ-1, label
     is y_u[k+SEQ-1] (the window's own last frame), for k=0..L-SEQ inclusive.

Mondrian CP: two explicit modes (verification item 4), both available:
  - mode='online'  : current/original behavior. One shared MondirianCP instance
    is warmed on cal_units, then continuously updated (.update() called after
    every predict_interval()) across ALL eval_units in sequence -- state persists
    across engine boundaries, exactly as before.
  - mode='frozen'  : calibrate once on cal_units, then for EACH eval engine
    independently, deep-copy that warmed instance and call ONLY predict_interval()
    (never .update()) -- no eval-time feedback ever enters any buffer, and no
    engine's evaluation can affect another's. Note: AdaptiveLambdaCP.predict_interval()
    itself calls _nr_update(), which re-estimates lambda from whatever is already
    in .buf on every call -- in frozen mode .buf never changes after warm-up (no
    .update() calls), so lambda may still drift call-to-call, but strictly as a
    function of the fixed calibration-set buffer, never of eval-time y_true.
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `models.*`/`conformal.*` resolve to this scripts/ dir

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from models.bayesian_lstm import BayesianLSTM
from conformal.mondrian_cp import MondirianCP, detect_cp_cusum, assign_stage

REPO = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
DATA_DIR = REPO / 'data' / 'cmapss'  # raw NASA C-MAPSS files -- not included, see README.md
MODELS = REPO / 'models'

COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SENSOR_COLS = ['op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SEQ = 30
ALPHA = 0.10
AVG_LIFETIME = {'FD001': 206, 'FD002': 206, 'FD003': 247, 'FD004': 247}


def load_split_and_scaler_clean(fd, seed):
    """Rebuilds the exact same 4-way split + scaler used by train_clean.py, from
    the manifest (meta_*.json), so downstream inference operates on identical
    partitions with no re-derivation risk."""
    tag = f'{fd}_seed{seed}'
    meta = json.loads((MODELS / f'meta_{tag}.json').read_text())

    train_df = pd.read_csv(DATA_DIR / f'train_{fd}.txt', sep=r'\s+', header=None, names=COLS)
    max_cycle = train_df.groupby('unit')['cycle'].max()
    train_df = train_df.merge(max_cycle.rename('max_cycle'), on='unit')
    train_df['rul'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)
    train_df = train_df.drop(columns='max_cycle')

    scaler = MinMaxScaler()
    scaler.data_min_ = np.array(meta['scaler_stats']['data_min_'])
    scaler.data_max_ = np.array(meta['scaler_stats']['data_max_'])
    rng = scaler.data_max_ - scaler.data_min_
    rng[rng == 0] = 1.0
    scaler.scale_ = 1.0 / rng
    scaler.min_ = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = len(SENSOR_COLS)
    train_df[SENSOR_COLS] = scaler.transform(train_df[SENSOR_COLS])

    return (train_df, meta['train_units'], meta['val_units'],
            meta['cal_units'], meta['eval_units'], meta['split_hash'])


def deterministic_seed(fd, seed, unit):
    """Stable, cross-process integer seed from (fd, seed, unit). Does NOT use
    Python's built-in hash() on strings, which is randomized per-process
    (PYTHONHASHSEED) unless disabled -- that randomization is the root cause
    traced in the Stage 0-R cross-table discrepancy: two independent script
    runs' MC-Dropout draws for the "same" unit were never actually the same
    draw, because nothing seeded them at all (not even hash-based)."""
    fd_num = int(fd[2:])  # 'FD001' -> 1
    return fd_num * 10_000_000 + seed * 100_000 + int(unit)


def get_engine_series_clean(df, unit, model, device, n_samples=50, seq=SEQ,
                             fixed_seed=None):
    """Fixed window/label alignment (see module docstring). Returns (mus, sigmas,
    y_true), all length L-seq+1 for a unit with L observed rows.

    fixed_seed: if given, torch.manual_seed(fixed_seed) is called immediately
    before the MC-Dropout forward passes, making this call reproducible across
    processes. Pass deterministic_seed(fd, seed, unit). If None (the default,
    kept for backward compatibility with pre-Stage-0R-v3 scripts), dropout
    draws are NOT reproducible -- every call gets fresh, uncontrolled
    randomness from PyTorch's global RNG state, which is what caused two
    independently-run scripts to silently diverge on the "same" unit."""
    sub = df[df['unit'] == unit]
    X_u = sub[SENSOR_COLS].values
    y_u = sub['rul'].values
    L = len(X_u)
    if L < seq + 15:
        return None

    n_windows = L - seq + 1
    windows = np.stack([X_u[k:k + seq] for k in range(n_windows)]).astype(np.float32)
    y_true = y_u[seq - 1:].astype(np.float32)  # y_true[k] = y_u[k+seq-1], matches window k's last frame
    assert len(y_true) == n_windows

    x_t = torch.tensor(windows).to(device)
    if fixed_seed is not None:
        torch.manual_seed(fixed_seed)
    with torch.no_grad():
        mu_mean, sigma_total = model.predict_mc(x_t, n_samples=n_samples)
    return mu_mean, sigma_total, y_true


def warm_mondrian_clean(cal_series_list, T_est, h_mult):
    k_mult = 0.7 * h_mult / 7.0
    mondrian = MondirianCP(alpha=ALPHA, W_max=200, total_lifetime_est=T_est)
    for mus, sigs, y_true in cal_series_list:
        scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
        t_cp = detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h_mult)
        n = len(mus)
        for t in range(n):
            stg = assign_stage(t, t_cp, n)
            mondrian.update(float(t), y_true[t], mus[t], sigs[t], stg)
    return mondrian


def run_mondrian_online(mondrian, mus, sigs, y_true, h_mult):
    """mode='online': mondrian is the SAME shared instance across all eval units
    for this (fd,seed,h) -- caller passes the same object every time, and this
    function mutates it via .update()."""
    scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
    k_mult = 0.7 * h_mult / 7.0
    t_cp = detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h_mult)
    lo, hi, stages = [], [], []
    n = len(mus)
    for t in range(n):
        stg = assign_stage(t, t_cp, n)
        l, h = mondrian.predict_interval(float(t), mus[t], sigs[t], stg)
        mondrian.update(float(t), y_true[t], mus[t], sigs[t], stg)
        lo.append(l); hi.append(h); stages.append(stg)
    return np.array(lo), np.array(hi), stages, t_cp


def run_mondrian_frozen(mondrian_warmed, mus, sigs, y_true, h_mult):
    """mode='frozen': deep-copies the warmed instance so this eval engine's run
    cannot affect, or be affected by, any other eval engine's run. Never calls
    .update() -- no eval-time y_true ever enters any buffer."""
    mondrian = copy.deepcopy(mondrian_warmed)
    scores = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
    k_mult = 0.7 * h_mult / 7.0
    t_cp = detect_cp_cusum(scores.tolist(), k_mult=k_mult, h_mult=h_mult)
    lo, hi, stages = [], [], []
    n = len(mus)
    for t in range(n):
        stg = assign_stage(t, t_cp, n)
        l, h = mondrian.predict_interval(float(t), mus[t], sigs[t], stg)
        lo.append(l); hi.append(h); stages.append(stg)
    return np.array(lo), np.array(hi), stages, t_cp
