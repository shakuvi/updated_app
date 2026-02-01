import os
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# ───────── CONFIG ─────────
CSV_PATH      = 'geom_features.csv'
RUN_DIR       = 'run_mlp'
TEST_SUBJ_FRAC= 0.2       # fraction of subjects held out for final test
VAL_FRAC      = 0.2       # fraction of remaining train used for validation
BATCH_SIZE    = 64
EPOCHS        = 30
LR            = 1e-3
RANDOM_SEED   = 42
# ────────────────────────────


FEATURES = ['yaw', 'pitch', 'l_h_ratio', 'l_v_ratio', 'r_h_ratio', 'r_v_ratio']


def _seed_everything(RANDOM_SEED):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def _load_frame(CSV_PATH):
    return pd.read_csv(CSV_PATH)


def _subject_split(df, TEST_SUBJ_FRAC, RANDOM_SEED):
    subjects = df['subject'].unique().tolist()
    random.seed(RANDOM_SEED)
    random.shuffle(subjects)

    n_test = int(len(subjects) * TEST_SUBJ_FRAC)
    test_subj = set(subjects[:n_test])
    train_subj = set(subjects[n_test:])

    train_df = df[df.subject.isin(train_subj)].reset_index(drop=True)
    test_df = df[df.subject.isin(test_subj)].reset_index(drop=True)
    return train_df, test_df, train_subj, test_subj


def _train_val_split(train_df, VAL_FRAC, RANDOM_SEED):
    val_df = train_df.sample(frac=VAL_FRAC, random_state=RANDOM_SEED)
    trn_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return trn_df, val_df


def _xy_from_frames(trn_df, val_df, test_df):
    X_train = trn_df[FEATURES].values.astype(np.float32)
    y_train = trn_df['label'].values.astype(np.float32)

    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df['label'].values.astype(np.float32)

    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _scale_and_persist(X_train, X_val, X_test, RUN_DIR):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    os.makedirs(RUN_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(RUN_DIR, 'scaler.joblib'))
    return X_train, X_val, X_test, scaler


def _make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


class GeoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _criterion_and_opt(model, y_train, device, LR):
    pos_ratio = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_ratio], device=device)
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    return criterion, optimizer, pos_ratio


def _val_auc(model, val_loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(yb.numpy())

    preds = 1 / (1 + np.exp(-np.concatenate(all_logits)))
    labels = np.concatenate(all_labels)
    return roc_auc_score(labels, preds)


def _fit(model, train_loader, val_loader, criterion, optimizer, device, EPOCHS, RUN_DIR):
    best_auc = 0.0
    best_path = os.path.join(RUN_DIR, 'best_geom_mlp.pth')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        auc = _val_auc(model, val_loader, device)
        print(f"Epoch {epoch:02d}  Val ROC AUC = {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)

    print(f"\n✅ Best validation AUC: {best_auc:.4f}")
    return best_auc


def _test_metrics(model, test_loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(yb.numpy())

    probs = 1 / (1 + np.exp(-np.concatenate(all_logits)))
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, (probs > 0.5).astype(int))
    auc_t = roc_auc_score(labels, probs)
    return acc, auc_t


def _save_metrics(RUN_DIR, best_auc, acc, auc_t, pos_ratio):
    with open(os.path.join(RUN_DIR, 'metrics.json'), 'w') as f:
        import json
        json.dump(
            {
                'best_val_auc': best_auc,
                'test_accuracy': acc,
                'test_roc_auc': auc_t,
                'pos_ratio': pos_ratio,
            },
            f,
            indent=2
        )


def main():
    _seed_everything(RANDOM_SEED)

    df = _load_frame(CSV_PATH)
    train_df, test_df, train_subj, test_subj = _subject_split(df, TEST_SUBJ_FRAC, RANDOM_SEED)
    trn_df, val_df = _train_val_split(train_df, VAL_FRAC, RANDOM_SEED)

    X_train, y_train, X_val, y_val, X_test, y_test = _xy_from_frames(trn_df, val_df, test_df)

    print(f"Subjects → train:{len(train_subj)} val_split:{len(val_df)} test:{len(test_subj)}")
    print(f"Samples  → train:{len(X_train)} val:{len(X_val)} test:{len(X_test)}")

    X_train, X_val, X_test, scaler = _scale_and_persist(X_train, X_val, X_test, RUN_DIR)
    train_loader, val_loader, test_loader = _make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE)

    device = _device()
    model = GeoMLP().to(device)

    criterion, optimizer, pos_ratio = _criterion_and_opt(model, y_train, device, LR)
    best_auc = _fit(model, train_loader, val_loader, criterion, optimizer, device, EPOCHS, RUN_DIR)

    model.load_state_dict(torch.load(os.path.join(RUN_DIR, 'best_geom_mlp.pth'), map_location=device))
    acc, auc_t = _test_metrics(model, test_loader, device)

    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test ROC AUC : {auc_t:.3f}")

    _save_metrics(RUN_DIR, best_auc, acc, auc_t, pos_ratio)

    print(f"All artifacts saved under '{RUN_DIR}/'")


if __name__ == '__main__':
    main()
