"""
Forensics 2026-09-11 — Item 2: headline claim B reproduction (Table 2, six methods).

Does NOT repeat the leakage found in experiments/run_c3.py (test_units drawn from the
train-unit index range -- see FINDINGS_LOG.md). Instead uses a proper 3-way engine-level
split, entirely disjoint:
  - train_units  (60% of units): used by train_multiseed.py to fit the LSTM (mu_head/sigma_head)
  - cal_units    (next 20%)    : used ONLY to fit each CP method's calibration state
  - eval_units   (last 20%)    : held out from both training and calibration; the only units
                                  ECR/MPIW are measured on
All three sets are engine-disjoint, drawn from train_{fd}.txt with the SAME seeded shuffle as
train_multiseed.py, so train_units here == train_units the model was actually fit on (verified
via split_hash match against meta_{tag}.json). cal_units/eval_units are then carved out of what
train_multiseed.py called "cal_units" (which the model never saw).

Full per-cycle trajectories (not the single-window official test set) are required because
stage-wise (CUSUM-based) evaluation needs a within-engine time series. The official
test_FD00X.txt only gives one truncated window per engine (see cmapss_loader.py), so eval_units
are evaluated on their train-file trajectories instead -- this is the only way to get a
genuinely held-out AND full-trajectory evaluation set from C-MAPSS.

Six methods, all wrapping the SAME per-timestep (mu_t, sigma_t) from the SAME trained model:
  MC-Dropout (uncal.) | Standard CP | SPCI-S | ACI | EnbPI | Mondrian CP
Stage assignment: CUSUM-based (assign_stage + detect_cp_cusum), matching how Mondrian CP itself
defines stages. Primary run at the code's actual default h_mult=7.0; sensitivity at
h_mult in {5.0, 10.0} also recorded (item 3 rule).
"""
import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import norm, binomtest, wilcoxon

from models.bayesian_lstm import BayesianLSTM
from conformal.mondrian_cp import MondirianCP, detect_cp_cusum, assign_stage
from conformal.adaptive_lambda_cp import AdaptiveLambdaCP
from conformal.standard_cp import compute_scores, split_cp_intervals

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.path.expanduser('~/论文4/data/cmapss'))
MODELS = REPO / 'results' / 'forensics_2026-09-11' / 'models'
OUT = REPO / 'results' / 'forensics_2026-09-11'
ALPHA = 0.10
SEQ = 30

COLS = (['unit','cycle','op1','op2','op3'] +
        [f's{i}' for i in range(1, 22)])
SENSOR_COLS = ['s2','s3','s4','s6','s7','s8',
               's9','s11','s12','s13','s14',
               's15','s17','s20','s21']

FDS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [0, 1, 2, 3, 4]
H_MULTS = [7.0, 5.0, 10.0]   # 7.0 = code default (primary); 5.0/10.0 = item-3 sensitivity


def load_split_and_scaler(fd, seed):
    """Reproduce the exact unit shuffle used by train_multiseed.py, then carve the
    20% 'cal_units' region further into cal-fit (first half) and eval-holdout (second half)."""
    from sklearn.preprocessing import MinMaxScaler
    train_df = pd.read_csv(DATA_DIR / f'train_{fd}.txt', sep=r'\s+', header=None, names=COLS)
    max_cycle = train_df.groupby('unit')['cycle'].max()
    train_df = train_df.merge(max_cycle.rename('max_cycle'), on='unit')
    train_df['rul'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)
    train_df = train_df.drop(columns='max_cycle')

    scaler = MinMaxScaler()
    train_df[SENSOR_COLS] = scaler.fit_transform(train_df[SENSOR_COLS])

    units = train_df['unit'].unique()
    rng = np.random.RandomState(seed)
    units = units.copy()
    rng.shuffle(units)

    split = int(len(units) * 0.8)
    model_train_units = units[:split]
    cal_pool = units[split:]

    split_hash = hashlib.sha256(
        (','.join(map(str, sorted(model_train_units.tolist()))) + '|' +
         ','.join(map(str, sorted(cal_pool.tolist())))).encode()
    ).hexdigest()[:16]

    half = len(cal_pool) // 2
    cal_units = cal_pool[:half] if half > 0 else cal_pool
    eval_units = cal_pool[half:] if half > 0 else cal_pool

    return train_df, model_train_units, cal_units, eval_units, split_hash


def predict_engine(model, X_u, device, seq=SEQ, mc=15):
    mus, sigs = [], []
    model.train()
    for t in range(seq, len(X_u)):
        x_t = torch.tensor(X_u[t-seq:t][None], dtype=torch.float32).to(device)
        with torch.no_grad():
            m_list = [model(x_t)[0].item() for _ in range(mc)]
            s_list = [model(x_t)[1].item() for _ in range(mc)]
        mus.append(float(np.mean(m_list)))
        sigs.append(float(np.mean(s_list)))
    return np.array(mus), np.array(sigs)


def get_engine_series(df, unit, model, device):
    sub = df[df['unit'] == unit]
    X_u = sub[SENSOR_COLS].values
    y_u = sub['rul'].values
    if len(X_u) < SEQ + 15:
        return None
    mus, sigs = predict_engine(model, X_u, device)
    y_true = y_u[SEQ:]
    return mus, sigs, y_true


def run_mc_dropout(mus, sigs, y_true):
    z = norm.ppf(1 - ALPHA/2)
    sig_c = np.maximum(sigs, 1e-3)
    lo, hi = mus - z*sig_c, mus + z*sig_c
    return lo, hi


def run_standard_cp(mus, sigs, y_true, cal_scores):
    n = len(cal_scores)
    q = np.quantile(cal_scores, np.ceil((n+1)*(1-ALPHA))/n, method='higher')
    sig_c = np.maximum(sigs, 1e-3)
    return mus - q*sig_c, mus + q*sig_c


def run_spci_s(mus, sigs, y_true, cal_scores, W=200):
    Wc = min(W, len(cal_scores))
    tail = cal_scores[-Wc:]
    q = np.quantile(tail, np.ceil((Wc+1)*(1-ALPHA))/Wc, method='higher')
    sig_c = np.maximum(sigs, 1e-3)
    return mus - q*sig_c, mus + q*sig_c


_PRIVATE_CLASS_CACHE = {}


def _load_private_workspace_class(module_file, class_name):
    """ACI/EnbPI are not in the anonymized public repo (see FINDINGS_LOG.md --
    they exist only in the author's private workspace at ~/论文4/conformal/).
    Load them directly by file path so as not to touch/copy anything into the
    audited repo."""
    key = (module_file, class_name)
    if key in _PRIVATE_CLASS_CACHE:
        return _PRIVATE_CLASS_CACHE[key]
    import importlib.util
    path = os.path.expanduser(f'~/论文4/conformal/{module_file}')
    spec = importlib.util.spec_from_file_location(module_file[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    _PRIVATE_CLASS_CACHE[key] = cls
    return cls


def run_aci(mus, sigs, y_true, cal_scores):
    ACI = _load_private_workspace_class('aci_cp.py', 'ACI')
    aci = ACI(alpha=ALPHA, gamma=0.005)
    aci.calibrate(list(cal_scores))
    lo, hi = [], []
    for i in range(len(mus)):
        l, h = aci.predict_interval(mus[i], sigs[i])
        lo.append(l); hi.append(h)
        aci.update(mus[i], sigs[i], y_true[i])
    return np.array(lo), np.array(hi)


def run_enbpi(mus, sigs, y_true, cal_mu, cal_sig, cal_y, W=200):
    EnbPI = _load_private_workspace_class('enbpi.py', 'EnbPI')
    en = EnbPI(alpha=ALPHA, W=W)
    en.calibrate(cal_mu, cal_sig, cal_y)
    lo, hi = [], []
    for i in range(len(mus)):
        l, h = en.predict_interval(mus[i], sigs[i])
        lo.append(l); hi.append(h)
        en.update(mus[i], sigs[i], y_true[i])
    return np.array(lo), np.array(hi)


def warm_mondrian(cal_series_list, T_est, h_mult):
    """Reproduce run_c3.py's calibration-set warm-up ('Mondrian CP在校准集上预热',
    run_c3.py:~185-216): feed every cal engine's own CUSUM-staged trajectory into
    ONE shared MondrianCP instance before it is ever used for prediction. Without
    this, each stage's small per-stage buffer starts empty and the conformal
    +inf-appended quantile (AdaptiveLambdaCP.predict_interval) is selected almost
    every time n_buf<~10, producing spurious infinite-width intervals -- an
    artifact of my reproduction harness, not of the original method."""
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


def run_macp(mondrian, mus, sigs, y_true, h_mult):
    """mondrian: a pre-warmed (see warm_mondrian) instance, shared and continuously
    updated across the whole eval_units sequence for this (fd, seed, h_mult) --
    matching run_c3.py's single shared `mondrian` object reused across all
    test_units in its loop."""
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


def stage_labels_from_tcp(n, t_cp):
    return [assign_stage(t, t_cp, n) for t in range(n)]


def stage_ecr_mpiw(y_true, lo, hi, stages):
    covered = (y_true >= lo) & (y_true <= hi)
    mpiw_arr = hi - lo
    out = {}
    for stg in ['global', 'early', 'middle', 'late']:
        mask = np.ones(len(y_true), bool) if stg == 'global' else np.array([s == stg for s in stages])
        if mask.sum() == 0:
            continue
        out[stg] = {'ecr': float(covered[mask].mean()),
                     'mpiw': float(mpiw_arr[mask].mean()),
                     'n': int(mask.sum())}
    return out


AVG_LIFETIME = {'FD001': 206, 'FD002': 206, 'FD003': 247, 'FD004': 247}


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for fd in FDS:
        for seed in SEEDS:
            tag = f'{fd}_seed{seed}'
            meta_path = MODELS / f'meta_{tag}.json'
            weight_path = MODELS / f'best_{tag}.pt'
            if not meta_path.exists() or not weight_path.exists():
                print(f'[{tag}] missing model/meta, skip')
                continue
            meta = json.loads(meta_path.read_text())

            df, model_train_units, cal_units, eval_units, split_hash = load_split_and_scaler(fd, seed)
            assert split_hash == meta['split_hash'], f'{tag}: split_hash mismatch, model not trained on this split!'

            model = BayesianLSTM(input_dim=len(SENSOR_COLS)).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()

            # ── Fit calibration state from cal_units (held out from training) ──
            # cal_series cached once (inference is expensive); reused both for the
            # batch methods' cal_scores_all AND for warming Mondrian's per-stage buffers.
            cal_series = []
            cal_scores_all = []
            cal_mu_all, cal_sig_all, cal_y_all = [], [], []
            for u in cal_units:
                series = get_engine_series(df, u, model, device)
                if series is None:
                    continue
                mus, sigs, y_true = series
                cal_series.append((mus, sigs, y_true))
                s = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
                cal_scores_all.extend(s.tolist())
                cal_mu_all.extend(mus.tolist()); cal_sig_all.extend(sigs.tolist()); cal_y_all.extend(y_true.tolist())
            cal_scores_all = np.array(cal_scores_all)
            cal_mu_all = np.array(cal_mu_all); cal_sig_all = np.array(cal_sig_all); cal_y_all = np.array(cal_y_all)

            T_est = AVG_LIFETIME[fd]

            # ── Warm one shared MondrianCP per h_mult on cal_units (run_c3.py-style
            #    pre-warm), then keep updating it continuously across eval_units ──
            mondrians = {h: warm_mondrian(cal_series, T_est, h) for h in H_MULTS}

            # ── Evaluate on eval_units (never seen by training or calibration) ──
            for u in eval_units:
                series = get_engine_series(df, u, model, device)
                if series is None:
                    continue
                mus, sigs, y_true = series
                n = len(mus)

                for h_mult in H_MULTS:
                    lo_m, hi_m, stages_m, t_cp = run_macp(mondrians[h_mult], mus, sigs, y_true, h_mult)
                    r = stage_ecr_mpiw(y_true, lo_m, hi_m, stages_m)
                    for stg, v in r.items():
                        rows.append(dict(fd=fd, seed=seed, unit=int(u), method='Mondrian CP', h_mult=h_mult,
                                          stage=stg, ecr=v['ecr'], mpiw=v['mpiw'], n=v['n'], t_cp=t_cp))

                # stage labels for the non-Mondrian-CP methods: use the SAME primary-h CUSUM t_cp
                # (h_mult=7.0, the code default) so all six methods are compared on identical
                # stage definitions.
                scores_primary = np.minimum(np.abs(y_true - mus) / np.maximum(sigs, 1e-3), 10.0)
                t_cp_primary = detect_cp_cusum(scores_primary.tolist(), k_mult=0.7, h_mult=7.0)
                stages_primary = stage_labels_from_tcp(n, t_cp_primary)

                lo, hi = run_mc_dropout(mus, sigs, y_true)
                r = stage_ecr_mpiw(y_true, lo, hi, stages_primary)
                for stg, v in r.items():
                    rows.append(dict(fd=fd, seed=seed, unit=int(u), method='MC-Dropout (uncal.)',
                                      h_mult=7.0, stage=stg, ecr=v['ecr'], mpiw=v['mpiw'], n=v['n'], t_cp=t_cp_primary))

                lo, hi = run_standard_cp(mus, sigs, y_true, cal_scores_all)
                r = stage_ecr_mpiw(y_true, lo, hi, stages_primary)
                for stg, v in r.items():
                    rows.append(dict(fd=fd, seed=seed, unit=int(u), method='Standard CP',
                                      h_mult=7.0, stage=stg, ecr=v['ecr'], mpiw=v['mpiw'], n=v['n'], t_cp=t_cp_primary))

                lo, hi = run_spci_s(mus, sigs, y_true, cal_scores_all)
                r = stage_ecr_mpiw(y_true, lo, hi, stages_primary)
                for stg, v in r.items():
                    rows.append(dict(fd=fd, seed=seed, unit=int(u), method='SPCI-S',
                                      h_mult=7.0, stage=stg, ecr=v['ecr'], mpiw=v['mpiw'], n=v['n'], t_cp=t_cp_primary))

                lo, hi = run_aci(mus, sigs, y_true, cal_scores_all)
                r = stage_ecr_mpiw(y_true, lo, hi, stages_primary)
                for stg, v in r.items():
                    rows.append(dict(fd=fd, seed=seed, unit=int(u), method='ACI',
                                      h_mult=7.0, stage=stg, ecr=v['ecr'], mpiw=v['mpiw'], n=v['n'], t_cp=t_cp_primary))

                lo, hi = run_enbpi(mus, sigs, y_true, cal_mu_all, cal_sig_all, cal_y_all)
                r = stage_ecr_mpiw(y_true, lo, hi, stages_primary)
                for stg, v in r.items():
                    rows.append(dict(fd=fd, seed=seed, unit=int(u), method='EnbPI',
                                      h_mult=7.0, stage=stg, ecr=v['ecr'], mpiw=v['mpiw'], n=v['n'], t_cp=t_cp_primary))

            print(f'[{tag}] n_cal_units={len(cal_units)} n_eval_units={len(eval_units)} done')

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / 'item2_six_method_multiseed.csv', index=False)
    print(f'\nSaved {len(df_out)} rows -> item2_six_method_multiseed.csv')


if __name__ == '__main__':
    run()
