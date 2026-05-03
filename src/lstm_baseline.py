import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json

DATA_DIR   = "data/processed"
SAVE_DIR   = "models/baselines"
HORIZON    = 1

BATCH_SIZE = 128          
EPOCHS     = 50           
LR         = 0.001
PATIENCE   = 10          
MAX_TRAIN_SAMPLES = 50000   
MAX_VAL_SAMPLES   = 10000
MAX_TEST_SAMPLES  = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

X_train = np.load(f"{DATA_DIR}/X_train_h{HORIZON}.npy")
y_train = np.load(f"{DATA_DIR}/y_train_h{HORIZON}.npy")
X_val   = np.load(f"{DATA_DIR}/X_val_h{HORIZON}.npy")
y_val   = np.load(f"{DATA_DIR}/y_val_h{HORIZON}.npy")
X_test  = np.load(f"{DATA_DIR}/X_test_h{HORIZON}.npy")
y_test  = np.load(f"{DATA_DIR}/y_test_h{HORIZON}.npy")

min_vals = np.load(f"{DATA_DIR}/min_vals.npy")
max_vals = np.load(f"{DATA_DIR}/max_vals.npy")
X_train = X_train[:MAX_TRAIN_SAMPLES]
y_train = y_train[:MAX_TRAIN_SAMPLES]

X_val = X_val[:MAX_VAL_SAMPLES]
y_val = y_val[:MAX_VAL_SAMPLES]

X_test = X_test[:MAX_TEST_SAMPLES]
y_test = y_test[:MAX_TEST_SAMPLES]

N = X_train.shape[1]
T = X_train.shape[2]

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
val_loader = torch.utils.data.DataLoader(
    AQIDataset(X_val_r, y_val_r), batch_size=BATCH_SIZE
)
test_loader = torch.utils.data.DataLoader(
    AQIDataset(X_test_r, y_test_r), batch_size=BATCH_SIZE
)

class LSTMBaseline(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMBaseline().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

def inverse_transform(preds, targets):
    preds = preds.squeeze()
    targets = targets.squeeze()

    total = len(preds)
    reps = total // N

    min_tiled = np.tile(min_vals, reps)
    max_tiled = np.tile(max_vals, reps)

    preds_real = preds * (max_tiled - min_tiled) + min_tiled
    targets_real = targets * (max_tiled - min_tiled) + min_tiled

    return preds_real, targets_real

def compute_metrics(preds, targets):
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return mae, rmse

def train_epoch():
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(loader):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            preds = model(xb)
            total_loss += loss_fn(preds, yb).item()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    preds_real, targets_real = inverse_transform(preds, targets)
    mae, rmse = compute_metrics(preds_real, targets_real)

    return total_loss / len(loader), mae, rmse

print("\nTraining LSTM baseline...\n")

best_val = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch()
    val_loss, val_mae, val_rmse = evaluate(val_loader)

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

model.load_state_dict(torch.load(f"{SAVE_DIR}/lstm.pt"))

_, test_mae, test_rmse = evaluate(test_loader)

print("\nFINAL TEST RESULTS")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")

results = {
    "MAE": float(test_mae),
    "RMSE": float(test_rmse)
}

with open(f"{SAVE_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved results.")
