"""
Stage 0-R, step 1: clean retrain fixing verification-item-1 (early-stop-val leakage).

Engine-level 4-way mutually exclusive split per (fd, seed):
  train (60%) / early_stop_val (15%) / calibration (12.5%) / evaluation (12.5%)
Proportions chosen to keep calibration/evaluation set sizes comparable to the old
pipeline's cal-fit/eval halves (~10-12.5% each), while carving out a genuinely
separate early-stop-val slice that the old pipeline never had (it used the
calibration set itself for early stopping -- verification item 1).

Fixes applied vs. the old experiments/forensics_2026-09-11/train_multiseed.py:
  1. scaler.fit() only on train_units' rows; val/cal/eval use .transform() only.
  2. val_loader is built from early_stop_val units, never from calibration or
     evaluation units. Early stopping (both stages) uses ONLY this loader.
  3. All 4 partitions are disjoint by construction (single shuffle+cumulative-split
     of the unit array) and this is asserted before training.

Writes, per (fd, seed): best_{tag}.pt (final model), meta_{tag}.json (manifest:
split hash over all 4 partitions, scaler stats source+values, RMSE/NLL, n per
partition, git commit). Does NOT touch results/forensics_2026-09-11/ (old,
leaky pipeline) -- this is a fully separate results tree so old and new can be
compared side by side.
"""
import sys, os, json, hashlib, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `models.*` resolves to this scripts/ dir

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from models.bayesian_lstm import BayesianLSTM, gaussian_nll_loss

REPO = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
DATA_DIR = REPO / 'data' / 'cmapss'  # raw NASA C-MAPSS files -- not included, see README.md
OUT = REPO / 'models'
OUT.mkdir(parents=True, exist_ok=True)

COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SENSOR_COLS = ['op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

SPLIT_FRACS = {'train': 0.60, 'val': 0.15, 'cal': 0.125, 'eval': 0.125}
assert abs(sum(SPLIT_FRACS.values()) - 1.0) < 1e-9

CFG = {
    'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.2,
    'lr_stage1': 1e-3, 'epochs_stage1': 50, 'patience_stage1': 10,
    'lr_stage2': 3e-4, 'epochs_stage2': 100, 'patience_stage2': 15,
    'batch_size': 256,
}


def load_dataset_clean(fd, seed, max_rul=125):
    train_df = pd.read_csv(DATA_DIR / f'train_{fd}.txt', sep=r'\s+', header=None, names=COLS)

    def add_rul(df):
        max_cycle = df.groupby('unit')['cycle'].max()
        df = df.merge(max_cycle.rename('max_cycle'), on='unit')
        df['rul'] = (df['max_cycle'] - df['cycle']).clip(upper=max_rul)
        return df.drop(columns='max_cycle')

    train_df = add_rul(train_df)

    units = train_df['unit'].unique().copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(units)
    n = len(units)
    n_train = int(round(n * SPLIT_FRACS['train']))
    n_val = int(round(n * SPLIT_FRACS['val']))
    n_cal = int(round(n * SPLIT_FRACS['cal']))
    # eval takes the remainder so the 4 counts always sum to n exactly
    n_eval = n - n_train - n_val - n_cal

    train_units = units[:n_train]
    val_units = units[n_train:n_train + n_val]
    cal_units = units[n_train + n_val:n_train + n_val + n_cal]
    eval_units = units[n_train + n_val + n_cal:]

    assert len(set(train_units) & set(val_units)) == 0
    assert len(set(train_units) & set(cal_units)) == 0
    assert len(set(train_units) & set(eval_units)) == 0
    assert len(set(val_units) & set(cal_units)) == 0
    assert len(set(val_units) & set(eval_units)) == 0
    assert len(set(cal_units) & set(eval_units)) == 0
    assert len(train_units) + len(val_units) + len(cal_units) + len(eval_units) == n

    def part_hash(u):
        return ','.join(map(str, sorted(u.tolist())))

    split_hash = hashlib.sha256(
        '|'.join([part_hash(train_units), part_hash(val_units),
                   part_hash(cal_units), part_hash(eval_units)]).encode()
    ).hexdigest()[:16]

    # -- fix #1: scaler fit ONLY on train_units' rows --
    scaler = MinMaxScaler()
    train_mask = train_df['unit'].isin(train_units)
    scaler.fit(train_df.loc[train_mask, SENSOR_COLS])
    scaler_stats = {
        'fit_source': 'train_units_only',
        'n_rows_fit': int(train_mask.sum()),
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist(),
    }
    train_df[SENSOR_COLS] = scaler.transform(train_df[SENSOR_COLS])

    def extract_sequences(df, unit_list, seq_len=30):
        X_list, y_list = [], []
        for u in unit_list:
            sub = df[df['unit'] == u][SENSOR_COLS].values
            rul = df[df['unit'] == u]['rul'].values
            for i in range(len(sub) - seq_len + 1):
                X_list.append(sub[i:i + seq_len])
                y_list.append(rul[i + seq_len - 1])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    train_X, train_y = extract_sequences(train_df, train_units)
    val_X, val_y = extract_sequences(train_df, val_units)
    cal_X, cal_y = extract_sequences(train_df, cal_units)
    eval_X, eval_y = extract_sequences(train_df, eval_units)

    return dict(
        train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y,
        cal_X=cal_X, cal_y=cal_y, eval_X=eval_X, eval_y=eval_y,
        split_hash=split_hash, scaler_stats=scaler_stats,
        train_units=train_units.tolist(), val_units=val_units.tolist(),
        cal_units=cal_units.tolist(), eval_units=eval_units.tolist(),
    )


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_loader(X, y, batch_size, shuffle):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def train_one(fd, seed, device):
    tag = f'{fd}_seed{seed}'
    t0 = time.time()
    set_seed(seed)
    d = load_dataset_clean(fd, seed)

    train_loader = make_loader(d['train_X'], d['train_y'], CFG['batch_size'], True)
    val_loader = make_loader(d['val_X'], d['val_y'], CFG['batch_size'], False)  # fix #2: NOT cal_X

    model = BayesianLSTM(input_dim=d['train_X'].shape[2], hidden_dim=CFG['hidden_dim'],
                          num_layers=CFG['num_layers'], dropout=CFG['dropout']).to(device)

    for p in model.sigma_head.parameters():
        p.requires_grad_(False)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=CFG['lr_stage1'], weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    best_rmse = float('inf')
    pat = 0
    stage1_path = OUT / f'stage1_{tag}.pt'
    for epoch in range(1, CFG['epochs_stage1'] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt1.zero_grad()
            mu, _ = model(xb)
            loss = torch.nn.functional.mse_loss(mu, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                mu, _ = model(xb.to(device))
                preds.extend(mu.cpu().numpy())
                trues.extend(yb.cpu().numpy())
        rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(trues)) ** 2)))
        sch1.step(rmse)
        if rmse < best_rmse - 0.1:
            best_rmse = rmse
            pat = 0
            torch.save(model.state_dict(), stage1_path)
        else:
            pat += 1
            if pat >= CFG['patience_stage1']:
                break
    model.load_state_dict(torch.load(stage1_path, map_location=device))

    for p in model.sigma_head.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.AdamW([
        {'params': model.lstm.parameters(), 'lr': CFG['lr_stage2']},
        {'params': model.mu_head.parameters(), 'lr': CFG['lr_stage2']},
        {'params': model.sigma_head.parameters(), 'lr': CFG['lr_stage2'] * 0.3},
    ], weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CFG['epochs_stage2'])
    best_val = float('inf')
    pat = 0
    best_path = OUT / f'best_{tag}.pt'
    for epoch in range(1, CFG['epochs_stage2'] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt2.zero_grad()
            mu, sigma = model(xb)
            loss = gaussian_nll_loss(mu, sigma, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, sigma = model(xb)
                v_losses.append(gaussian_nll_loss(mu, sigma, yb).item())
        vl = float(np.mean(v_losses))
        sch2.step()
        if vl < best_val - 1e-4:
            best_val = vl
            pat = 0
            torch.save(model.state_dict(), best_path)
        else:
            pat += 1
            if pat >= CFG['patience_stage2']:
                break
    model.load_state_dict(torch.load(best_path, map_location=device))
    stage1_path.unlink(missing_ok=True)

    # RMSE/NLL on the held-out evaluation split, for the old-vs-new performance
    # comparison the user asked for -- NOT used for any model-selection decision.
    model.eval()
    eval_t = torch.tensor(d['eval_X']).to(device)
    eval_y_t = torch.tensor(d['eval_y']).to(device)
    with torch.no_grad():
        mu_eval, sigma_eval = model(eval_t)
        eval_rmse = float(torch.sqrt(torch.mean((mu_eval - eval_y_t) ** 2)).item())
        eval_nll = float(gaussian_nll_loss(mu_eval, sigma_eval, eval_y_t).item())

    dt = time.time() - t0
    git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                 capture_output=True, text=True).stdout.strip()
    meta = {
        'fd': fd, 'seed': seed, 'split_hash': d['split_hash'],
        'git_commit': git_commit, 'stage1_best_rmse': best_rmse, 'stage2_best_nll': best_val,
        'eval_rmse_final': eval_rmse, 'eval_nll_final': eval_nll,
        'elapsed_sec': round(dt, 1),
        'n_train': len(d['train_X']), 'n_val': len(d['val_X']),
        'n_cal': len(d['cal_X']), 'n_eval': len(d['eval_X']),
        'n_train_units': len(d['train_units']), 'n_val_units': len(d['val_units']),
        'n_cal_units': len(d['cal_units']), 'n_eval_units': len(d['eval_units']),
        'split_fracs': SPLIT_FRACS,
        'scaler_stats': d['scaler_stats'],
        'train_units': d['train_units'], 'val_units': d['val_units'],
        'cal_units': d['cal_units'], 'eval_units': d['eval_units'],
    }
    with open(OUT / f'meta_{tag}.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[{tag}] done in {dt:.0f}s  split_hash={d["split_hash"]}  '
          f'stage1_rmse={best_rmse:.2f} stage2_nll={best_val:.4f} '
          f'eval_rmse={eval_rmse:.2f} eval_nll={eval_nll:.4f}')
    return meta


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    fds = ['FD001', 'FD002', 'FD003', 'FD004']
    seeds = [0, 1, 2, 3, 4]
    for fd in fds:
        for seed in seeds:
            tag = f'{fd}_seed{seed}'
            if (OUT / f'meta_{tag}.json').exists():
                print(f'[{tag}] already done, skipping')
                continue
            train_one(fd, seed, device)
    print('ALL DONE')
