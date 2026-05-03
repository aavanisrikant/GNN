import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json

DATA_DIR   = "data/processed"
SAVE_DIR   = "models/baselines"
HORIZON    = 1
BATCH_SIZE = 64
EPOCHS     = 100
LR         = 0.001
PATIENCE   = 15        
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

X_train = np.load(f"{DATA_DIR}/X_train_h{HORIZON}.npy")  # (S, N, 24)
y_train = np.load(f"{DATA_DIR}/y_train_h{HORIZON}.npy")  # (S, N)
X_val   = np.load(f"{DATA_DIR}/X_val_h{HORIZON}.npy")
y_val   = np.load(f"{DATA_DIR}/y_val_h{HORIZON}.npy")
X_test  = np.load(f"{DATA_DIR}/X_test_h{HORIZON}.npy")
y_test  = np.load(f"{DATA_DIR}/y_test_h{HORIZON}.npy")

min_vals = np.load(f"{DATA_DIR}/min_vals.npy")  # (N,)
max_vals = np.load(f"{DATA_DIR}/max_vals.npy")  # (N,)

N = X_train.shape[1]   # stations
T = X_train.shape[2]   # 24

print(f"Stations: {N} | Time steps: {T}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

def reshape_for_lstm(X, y):
    S, N, T = X.shape
    X_r = X.reshape(S * N, T, 1) 
    y_r = y.reshape(S * N, 1)
    return X_r, y_r

X_train_r, y_train_r = reshape_for_lstm(X_train, y_train)
X_val_r,   y_val_r   = reshape_for_lstm(X_val,   y_val)
X_test_r,  y_test_r  = reshape_for_lstm(X_test,  y_test)
print(f"Reshaped train X: {X_train_r.shape}")


class AQIDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(
    AQIDataset(X_train_r, y_train_r), batch_size=BATCH_SIZE, shuffle=True
)
val_loader  = torch.utils.data.DataLoader(
    AQIDataset(X_val_r,   y_val_r),   batch_size=BATCH_SIZE
)
test_loader = torch.utils.data.DataLoader(
    AQIDataset(X_test_r,  y_test_r),  batch_size=BATCH_SIZE
)

#model
class LSTMBaseline(nn.Module):
    """Per-station LSTM. No spatial awareness — purely temporal baseline."""
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = 1,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, 24, hidden)
        out = out[:, -1, :]         # last timestep: (batch, hidden)
        return self.fc(out)         # (batch, 1)

model     = LSTMBaseline().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=7, verbose=True
)
loss_fn = nn.MSELoss()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# inverse transform
def inverse_transform(preds_norm: np.ndarray, targets_norm: np.ndarray):
    """
    Converts normalised predictions and targets back to real AQI.
    Handles variable batch sizes safely by tiling based on actual array length.

    preds_norm, targets_norm: shape (S*N,) or (S*N, 1)
    Returns: preds_real, targets_real both shape (S*N,)
    """
    p = preds_norm.squeeze()
    t = targets_norm.squeeze()

    total = len(p)    
    reps  = total // N
    remainder = total - reps * N

    min_tiled = np.concatenate([np.tile(min_vals, reps),
                                 min_vals[:remainder]]) if remainder else np.tile(min_vals, reps)
    max_tiled = np.concatenate([np.tile(max_vals, reps),
                                 max_vals[:remainder]]) if remainder else np.tile(max_vals, reps)

    p_real = p * (max_tiled - min_tiled) + min_tiled
    t_real = t * (max_tiled - min_tiled) + min_tiled
    return p_real, t_real

def compute_metrics(preds_real: np.ndarray, targets_real: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(preds_real - targets_real)))
    rmse = float(np.sqrt(np.mean((preds_real - targets_real) ** 2)))

    mask = targets_real > 1.0   # avoid near-zero division in MAPE
    mape = float(np.mean(
        np.abs((preds_real[mask] - targets_real[mask]) / targets_real[mask])
    ) * 100) if mask.sum() > 0 else float("nan")

    ss_res = np.sum((targets_real - preds_real) ** 2)
    ss_tot = np.sum((targets_real - np.mean(targets_real)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

#train
def train_epoch() -> float:
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(train_loader)

def evaluate(loader) -> tuple:
    """Returns (mse_loss, metrics_dict, preds_real, targets_real)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            total_loss += loss_fn(preds, yb).item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    preds_norm   = np.concatenate(all_preds)
    targets_norm = np.concatenate(all_targets)

    preds_real, targets_real = inverse_transform(preds_norm, targets_norm)
    metrics = compute_metrics(preds_real, targets_real)

    return total_loss / len(loader), metrics, preds_real, targets_real

print(f"\nTraining LSTM baseline | Horizon={HORIZON}h | Patience={PATIENCE}\n")

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    train_loss          = train_epoch()
    val_loss, val_m, _, _ = evaluate(val_loader)

    scheduler.step(val_loss)

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"{SAVE_DIR}/lstm_h{HORIZON}.pt")
        tag = " ✓ saved"
    else:
        patience_counter += 1
        tag = f" (patience {patience_counter}/{PATIENCE})"

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | "
        f"MAE: {val_m['MAE']:.2f} | RMSE: {val_m['RMSE']:.2f} | "
        f"MAPE: {val_m['MAPE']:.1f}% | R²: {val_m['R2']:.3f}"
        + tag
    )

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} — no improvement for {PATIENCE} epochs.")
        break


#test evaluation
model.load_state_dict(torch.load(f"{SAVE_DIR}/lstm_h{HORIZON}.pt"))
_, test_m, preds_real, targets_real = evaluate(test_loader)

print("\n── LSTM BASELINE — FINAL TEST RESULTS ──────────────────")
print(f"  MAE  : {test_m['MAE']:.2f} AQI units")
print(f"  RMSE : {test_m['RMSE']:.2f} AQI units")
print(f"  MAPE : {test_m['MAPE']:.1f}%")
print(f"  R²   : {test_m['R2']:.3f}")
print("─────────────────────────────────────────────────────────")
results = {
    "model":   "LSTM_baseline",
    "horizon": HORIZON,
    "MAE":     round(test_m["MAE"],  3),
    "RMSE":    round(test_m["RMSE"], 3),
    "MAPE":    round(test_m["MAPE"], 3),
    "R2":      round(test_m["R2"],   3),
}
out_path = f"{SAVE_DIR}/lstm_results_h{HORIZON}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_path}")
