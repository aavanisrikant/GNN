import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import time

DATA_DIR   = "data/processed"
SAVE_DIR   = "models/baselines"
HORIZON    = 1

BATCH_SIZE = 128
EPOCHS     = 50
LR         = 0.001
PATIENCE   = 10

MAX_TRAIN_SAMPLES = 5000
MAX_VAL_SAMPLES   = 2000
MAX_TEST_SAMPLES  = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

X_train = np.load(f"{DATA_DIR}/X_train_h{HORIZON}.npy")
y_train = np.load(f"{DATA_DIR}/y_train_h{HORIZON}.npy")
X_val   = np.load(f"{DATA_DIR}/X_val_h{HORIZON}.npy")
y_val   = np.load(f"{DATA_DIR}/y_val_h{HORIZON}.npy")
X_test  = np.load(f"{DATA_DIR}/X_test_h{HORIZON}.npy")
y_test  = np.load(f"{DATA_DIR}/y_test_h{HORIZON}.npy")
time_index_test = np.load(
    f"{DATA_DIR}/time_index_h{HORIZON}.npy",
    allow_pickle=True
)

min_vals = np.load(f"{DATA_DIR}/min_vals.npy")
max_vals = np.load(f"{DATA_DIR}/max_vals.npy")

X_train, y_train = X_train[:MAX_TRAIN_SAMPLES], y_train[:MAX_TRAIN_SAMPLES]
X_val,   y_val   = X_val[:MAX_VAL_SAMPLES],     y_val[:MAX_VAL_SAMPLES]
X_test,  y_test  = X_test[:MAX_TEST_SAMPLES],   y_test[:MAX_TEST_SAMPLES]

N = X_train.shape[1]
T = X_train.shape[2]

print(f"Stations: {N} | Time steps: {T}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

def reshape_for_lstm(X, y):
    S, N, T = X.shape
    return X.reshape(S * N, T, 1), y.reshape(S * N, 1)

X_train_r, y_train_r = reshape_for_lstm(X_train, y_train)
X_val_r,   y_val_r   = reshape_for_lstm(X_val,   y_val)
X_test_r,  y_test_r  = reshape_for_lstm(X_test,  y_test)

class AQIDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(AQIDataset(X_train_r, y_train_r), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(AQIDataset(X_val_r, y_val_r), batch_size=BATCH_SIZE)
test_loader  = torch.utils.data.DataLoader(AQIDataset(X_test_r, y_test_r), batch_size=BATCH_SIZE)

class LSTMBaseline(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMBaseline().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

n_parameters = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_parameters:,}")

def inverse_transform(preds, targets):
    preds = preds.squeeze()
    targets = targets.squeeze()

    reps = len(preds) // N
    min_tiled = np.tile(min_vals, reps)
    max_tiled = np.tile(max_vals, reps)

    preds_real   = preds * (max_tiled - min_tiled) + min_tiled
    targets_real = targets * (max_tiled - min_tiled) + min_tiled

    return preds_real, targets_real

def compute_metrics(preds, targets):
    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    mask = targets > 1.0
    mape = np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100 if mask.sum() > 0 else np.nan

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return mae, rmse, mape, r2

def train_epoch():
    model.train()
    total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(train_loader)

def evaluate(loader):
    model.eval()
    total_loss = 0
    preds_all, targets_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            preds = model(xb)
            total_loss += loss_fn(preds, yb).item()

            preds_all.append(preds.cpu().numpy())
            targets_all.append(yb.cpu().numpy())

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)

    preds_real, targets_real = inverse_transform(preds, targets)
    return total_loss / len(loader), *compute_metrics(preds_real, targets_real)

print("\nTraining LSTM baseline...\n")

start_time = time.time()

best_val = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch()
    val_loss, val_mae, _, _, _ = evaluate(val_loader)

    print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.2f}")

    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"{SAVE_DIR}/lstm.pt")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

training_time = time.time() - start_time

model.load_state_dict(torch.load(f"{SAVE_DIR}/lstm.pt"))

val_loss, val_mae, val_rmse, val_mape, val_r2 = evaluate(val_loader)

# ── FINAL TEST EVALUATION ─────────────────────────────────────────
print("\nFINAL TEST RESULTS")

test_loss, test_mae, test_rmse, test_mape, test_r2 = evaluate(test_loader)
# ── SAVE PREDICTIONS FOR ERROR ANALYSIS ─────────────────────
# Collect predictions + targets from test_loader

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        out = model(x)

        all_preds.append(out.cpu().numpy())
        all_targets.append(y.cpu().numpy())

preds = np.concatenate(all_preds, axis=0)   # (samples, N)
targets = np.concatenate(all_targets, axis=0)

# Inverse transform
preds_real = preds * (max_vals - min_vals) + min_vals
targets_real = targets * (max_vals - min_vals) + min_vals

# Save
np.save(f"{SAVE_DIR}/lstm_preds_h{HORIZON}.npy", preds_real)
np.save(f"{SAVE_DIR}/lstm_targets_h{HORIZON}.npy", targets_real)

print("Saved predictions for error analysis.")
print(f"MAE  : {test_mae:.2f}")
print(f"RMSE : {test_rmse:.2f}")
print(f"MAPE : {test_mape:.1f}%")
print(f"R²   : {test_r2:.3f}")

results = {
    "model": "LSTM_baseline",
    "horizon": HORIZON,
    "MAE": float(test_mae),
    "RMSE": float(test_rmse),
    "MAPE": float(test_mape),
    "R2": float(test_r2),
    "training_time": round(training_time, 2),
    "n_parameters": int(n_parameters)
}

out_path = f"{SAVE_DIR}/lstm_results_h{HORIZON}.json"

with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved → {out_path}")
