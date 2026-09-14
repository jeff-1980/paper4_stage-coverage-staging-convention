"""
Forensics 2026-09-11 — item 1/2 prerequisite: multi-seed model training.

NON-INVASIVE ADDITION. Does not modify any existing repo file.
Re-implements the exact split/window logic of data/cmapss_loader.py::load_dataset
and the exact training regime of models/train_lstm.py::train_one_fd, with the
only deviation being that the RNG seed is a parameter instead of the hardcoded
np.random.seed(42) baked into cmapss_loader.py line 53. Architecture,
hyperparameters (CFG below, copied verbatim from models/train_lstm.py), epoch
budget, and early-stopping patience are unchanged from the original.

Split is by engine unit (train_units/cal_units), never by timestamp — this
matches the original loader and was verified leak-free by inspection
(data/cmapss_loader.py:52-59) before this script was written.

Outputs go to results/forensics_2026-09-11/ only. Existing results/*.npy
(the original seed=42 run) are untouched.
"""
import sys, os, json, hashlib, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from models.bayesian_lstm import BayesianLSTM, gaussian_nll_loss

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.path.expanduser('~/论文4/data/cmapss'))
OUT = REPO / 'results' / 'forensics_2026-09-11' / 'models'
OUT.mkdir(parents=True, exist_ok=True)

COLS = (['unit','cycle','op1','op2','op3'] +
        [f's{i}' for i in range(1, 22)])
SENSOR_COLS = ['s2','s3','s4','s6','s7','s8',
               's9','s11','s12','s13','s14',
               's15','s17','s20','s21']

CFG = {
    'hidden_dim'  : 128,
    'num_layers'  : 2,
    'dropout'     : 0.2,
    'lr_stage1'   : 1e-3,
    'epochs_stage1': 50,
    'patience_stage1': 10,
    'lr_stage2'   : 3e-4,
    'epochs_stage2': 100,
    'patience_stage2': 15,
    'batch_size'  : 256,
}


def load_dataset_seeded(fd, seed, max_rul=125):
    """Byte-for-byte copy of data/cmapss_loader.py::load_dataset, seed parameterized."""
    train_df = pd.read_csv(DATA_DIR / f'train_{fd}.txt',
                            sep=r'\s+', header=None, names=COLS)

    def add_rul(df):
        max_cycle = df.groupby('unit')['cycle'].max()
        df = df.merge(max_cycle.rename('max_cycle'), on='unit')
        df['rul'] = df['max_cycle'] - df['cycle']
        df['rul'] = df['rul'].clip(upper=max_rul)
        return df.drop(columns='max_cycle')

    train_df = add_rul(train_df)

    scaler = MinMaxScaler()
    train_df[SENSOR_COLS] = scaler.fit_transform(train_df[SENSOR_COLS])

    units = train_df['unit'].unique()
    rng = np.random.RandomState(seed)
    units = units.copy()
    rng.shuffle(units)

    split = int(len(units) * 0.8)
    train_units = units[:split]
    cal_units = units[split:]
    split_hash = hashlib.sha256(
        (','.join(map(str, sorted(train_units.tolist()))) + '|' +
         ','.join(map(str, sorted(cal_units.tolist())))).encode()
    ).hexdigest()[:16]

    def extract_sequences(df, unit_list, seq_len=30):
        X_list, y_list = [], []
        for u in unit_list:
            sub = df[df['unit'] == u][SENSOR_COLS].values
            rul = df[df['unit'] == u]['rul'].values
            for i in range(len(sub) - seq_len + 1):
                X_list.append(sub[i:i+seq_len])
                y_list.append(rul[i+seq_len-1])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    train_X, train_y = extract_sequences(train_df, train_units)
    cal_X, cal_y = extract_sequences(train_df, cal_units)

    test_df = pd.read_csv(DATA_DIR / f'test_{fd}.txt',
                           sep=r'\s+', header=None, names=COLS)
    test_df[SENSOR_COLS] = scaler.transform(test_df[SENSOR_COLS])
    rul_df = pd.read_csv(DATA_DIR / f'RUL_{fd}.txt', header=None, names=['rul'])

    seq_len = 30
    test_X_list, test_y_list = [], []
    for u in test_df['unit'].unique():
        sub = test_df[test_df['unit'] == u][SENSOR_COLS].values
        true_rul = rul_df.iloc[u-1]['rul']
        if len(sub) >= seq_len:
            test_X_list.append(sub[-seq_len:])
        else:
            pad = np.zeros((seq_len - len(sub), len(SENSOR_COLS)))
            test_X_list.append(np.vstack([pad, sub]))
        test_y_list.append(min(true_rul, max_rul))

    test_X = np.array(test_X_list, dtype=np.float32)
    test_y = np.array(test_y_list, dtype=np.float32)

    return (train_X, train_y, cal_X, cal_y, test_X, test_y, split_hash)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_loader(X, y, batch_size, shuffle):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=0, pin_memory=True)


def train_one(fd, seed, device):
    tag = f'{fd}_seed{seed}'
    t0 = time.time()
    set_seed(seed)
    train_X, train_y, cal_X, cal_y, test_X, test_y, split_hash = load_dataset_seeded(fd, seed)

    train_loader = make_loader(train_X, train_y, CFG['batch_size'], True)
    val_loader = make_loader(cal_X, cal_y, CFG['batch_size'], False)

    model = BayesianLSTM(input_dim=train_X.shape[2],
                          hidden_dim=CFG['hidden_dim'],
                          num_layers=CFG['num_layers'],
                          dropout=CFG['dropout']).to(device)

    # Stage 1: MSE pretrain mu, sigma_head frozen
    for p in model.sigma_head.parameters():
        p.requires_grad_(False)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=CFG['lr_stage1'], weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min', factor=0.5,
                                                        patience=5, min_lr=1e-6)
    best_rmse = float('inf')
    pat = 0
    stage1_path = OUT / f'stage1_{tag}.pt'
    for epoch in range(1, CFG['epochs_stage1']+1):
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
        rmse = float(np.sqrt(np.mean((np.array(preds)-np.array(trues))**2)))
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

    # Stage 2: joint NLL
    for p in model.sigma_head.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.AdamW([
        {'params': model.lstm.parameters(), 'lr': CFG['lr_stage2']},
        {'params': model.mu_head.parameters(), 'lr': CFG['lr_stage2']},
        {'params': model.sigma_head.parameters(), 'lr': CFG['lr_stage2']*0.3},
    ], weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CFG['epochs_stage2'])
    best_val = float('inf')
    pat = 0
    best_path = OUT / f'best_{tag}.pt'
    for epoch in range(1, CFG['epochs_stage2']+1):
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

    model.eval()
    test_t = torch.tensor(test_X).to(device)
    cal_t = torch.tensor(cal_X).to(device)
    mu_te, sig_te = model.predict_mc(test_t, n_samples=50)
    mu_ca, sig_ca = model.predict_mc(cal_t, n_samples=50)

    np.save(OUT / f'mu_{tag}.npy', mu_te)
    np.save(OUT / f'sigma_{tag}.npy', sig_te)
    np.save(OUT / f'true_{tag}.npy', test_y)
    np.save(OUT / f'cal_mu_{tag}.npy', mu_ca)
    np.save(OUT / f'cal_sigma_{tag}.npy', sig_ca)
    np.save(OUT / f'cal_true_{tag}.npy', cal_y)

    stage1_path.unlink(missing_ok=True)
    # best_path (best_{tag}.pt) is intentionally KEPT on disk: item 2's
    # held-out-engine online inference (cal-subset split, never seen by
    # training) needs the actual trained weights, not just point predictions.

    dt = time.time() - t0
    git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                 capture_output=True, text=True).stdout.strip()
    meta = {
        'fd': fd, 'seed': seed, 'split_hash': split_hash,
        'git_commit': git_commit, 'stage1_best_rmse': best_rmse,
        'stage2_best_nll': best_val, 'elapsed_sec': round(dt, 1),
        'n_train': len(train_X), 'n_cal': len(cal_X), 'n_test': len(test_X),
    }
    with open(OUT / f'meta_{tag}.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[{tag}] done in {dt:.0f}s  split_hash={split_hash}  '
          f'stage1_rmse={best_rmse:.2f} stage2_nll={best_val:.4f}')
    return meta


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    fds = ['FD001', 'FD002', 'FD003', 'FD004']
    seeds = [0, 1, 2, 3, 4]
    all_meta = []
    for fd in fds:
        for seed in seeds:
            tag = f'{fd}_seed{seed}'
            if (OUT / f'meta_{tag}.json').exists():
                print(f'[{tag}] already done, skipping')
                continue
            meta = train_one(fd, seed, device)
            all_meta.append(meta)
    print('ALL DONE')
