import time
import copy
import pathlib

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

# MODEL
from tgcn_model import TGCN

# REAL DATA
from data_loader import REAL_DATA


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    *,
    lr=0.001,
    max_epochs=50,
    patience=10,
    max_norm=5.0,
    checkpoint_path="best_real_tgcn.pt",
    device=None,
):

    # ────────────────────────────────────────────────────────────
    # DEVICE
    # ────────────────────────────────────────────────────────────
    if device is None:

        if torch.cuda.is_available():

            device = torch.device("cuda")

        elif torch.backends.mps.is_available():

            device = torch.device("mps")

        else:

            device = torch.device("cpu")

    print(f"Using device: {device}")

    model = model.to(device)

    # ────────────────────────────────────────────────────────────
    # OPTIMIZER
    # ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # ────────────────────────────────────────────────────────────
    # LOSSES
    # ────────────────────────────────────────────────────────────
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    # ────────────────────────────────────────────────────────────
    # LR SCHEDULER
    # ────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # ────────────────────────────────────────────────────────────
    # HISTORY
    # ────────────────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "lr": [],
    }

    # ────────────────────────────────────────────────────────────
    # EARLY STOPPING
    # ────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None

    checkpoint_path = pathlib.Path(
        checkpoint_path
    )

    # ────────────────────────────────────────────────────────────
    # TRAINING HEADER
    # ────────────────────────────────────────────────────────────
    print()

    print(
        f"{'Epoch':>6}  "
        f"{'Train MSE':>10}  "
        f"{'Val MSE':>10}  "
        f"{'Val MAE':>10}  "
        f"{'LR':>10}  "
        f"{'Time':>7}"
    )

    print("─" * 75)

    # ────────────────────────────────────────────────────────────
    # EPOCH LOOP
    # ────────────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):

        t0 = time.time()

        # ========================================================
        # TRAIN
        # ========================================================
        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} | "
                     f"Batch {batch_idx}/{len(train_loader)}"
                )

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size = x_batch.size(0)

            optimizer.zero_grad()

            preds = model(x_batch)

            loss = mse_loss(
                preds,
                y_batch,
            )

            loss.backward()

            # ────────────────────────────────────────────────
            # GRADIENT CLIPPING
            # ────────────────────────────────────────────────
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
            )

            optimizer.step()

            train_loss_sum += (
                loss.item()
                * batch_size
            )

            train_samples += batch_size

        avg_train_loss = (
            train_loss_sum
            / train_samples
        )

        # ========================================================
        # VALIDATION
        # ========================================================
        model.eval()

        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_samples = 0

        with torch.no_grad():

            for x_batch, y_batch in val_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                batch_size = x_batch.size(0)

                preds = model(x_batch)

                mse = mse_loss(
                    preds,
                    y_batch,
                )

                mae = mae_loss(
                    preds,
                    y_batch,
                )

                val_loss_sum += (
                    mse.item()
                    * batch_size
                )

                val_mae_sum += (
                    mae.item()
                    * batch_size
                )

                val_samples += batch_size

        avg_val_loss = (
            val_loss_sum
            / val_samples
        )

        avg_val_mae = (
            val_mae_sum
            / val_samples
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        elapsed = (
            time.time() - t0
        )

        # ========================================================
        # SAVE HISTORY
        # ========================================================
        history["train_loss"].append(
            avg_train_loss
        )

        history["val_loss"].append(
            avg_val_loss
        )

        history["val_mae"].append(
            avg_val_mae
        )

        history["lr"].append(
            current_lr
        )

        # ========================================================
        # CHECKPOINT
        # ========================================================
        improved = (
            avg_val_loss
            < best_val_loss
        )

        if improved:

            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        best_weights,
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_loss":
                        avg_val_loss,
                    "val_mae":
                        avg_val_mae,
                },
                checkpoint_path,
            )

        else:

            epochs_no_improve += 1

        # ========================================================
        # PRINT LOG
        # ========================================================
        marker = " ✓" if improved else ""

        print(
            f"{epoch:>6}  "
            f"{avg_train_loss:>10.5f}  "
            f"{avg_val_loss:>10.5f}  "
            f"{avg_val_mae:>10.5f}  "
            f"{current_lr:>10.6f}  "
            f"{elapsed:>6.1f}s"
            f"{marker}"
        )

        # ========================================================
        # LR SCHEDULER
        # ========================================================
        scheduler.step(avg_val_loss)

        # ========================================================
        # EARLY STOPPING
        # ========================================================
        if epochs_no_improve >= patience:

            print()

            print(
                f"Early stopping triggered "
                f"at epoch {epoch}"
            )

            break

    # ────────────────────────────────────────────────────────────
    # RESTORE BEST MODEL
    # ────────────────────────────────────────────────────────────
    if best_weights is not None:

        model.load_state_dict(
            best_weights
        )

    # ────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ────────────────────────────────────────────────────────────
    print()

    print("─" * 75)

    print(
        f"Best validation MSE : "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best epoch          : "
        f"{best_epoch}"
    )

    print(
        f"Checkpoint saved to : "
        f"{checkpoint_path.resolve()}"
    )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def real_training_run():

    print("=" * 75)
    print("REAL DATA T-GCN TRAINING")
    print("=" * 75)

    # ────────────────────────────────────────────────────────
    # LOAD REAL DATA
    # ────────────────────────────────────────────────────────
    X_train = REAL_DATA["X_train_full"]
    y_train = REAL_DATA["y_train"]

    X_val = REAL_DATA["X_val_full"]
    y_val = REAL_DATA["y_val"]

    edge_index = REAL_DATA["edge_index"]
    edge_weight = REAL_DATA["edge_weight"]

    # ────────────────────────────────────────────────────────
    # SHAPES
    # ────────────────────────────────────────────────────────
    print("\nDataset shapes:")

    print(f"X_train : {tuple(X_train.shape)}")
    print(f"y_train : {tuple(y_train.shape)}")

    print(f"X_val   : {tuple(X_val.shape)}")
    print(f"y_val   : {tuple(y_val.shape)}")

    # ────────────────────────────────────────────────────────
    # CONFIG
    # ────────────────────────────────────────────────────────
    BATCH_SIZE = 64

    T = X_train.shape[1]
    N = X_train.shape[2]
    F_IN = X_train.shape[3]

    OUT = y_train.shape[-1]

    # Smaller hidden size for debugging speed
    HIDDEN = 64

    # ────────────────────────────────────────────────────────
    # DATALOADERS
    # ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            X_val,
            y_val,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ────────────────────────────────────────────────────────
    # MODEL
    # ────────────────────────────────────────────────────────
    model = TGCN(
        in_channels=F_IN,
        hidden_dim=HIDDEN,
        output_steps=OUT,
        edge_index=edge_index,
        edge_weight=edge_weight,
        n_nodes=N,
        num_layers=2,
        dropout=0.2,
    )

    print("\nModel:")
    print(model)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nParameters: "
        f"{total_params:,}"
    )

    # ────────────────────────────────────────────────────────
    # TRAIN
    # ────────────────────────────────────────────────────────
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        max_epochs=50,
        patience=10,
        max_norm=5.0,
        checkpoint_path="best_real_tgcn.pt",
    )

    print()

    print("=" * 75)
    print("REAL TRAINING COMPLETE")
    print("=" * 75)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    real_training_run()
