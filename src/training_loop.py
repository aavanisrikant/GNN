import time
import copy
import pathlib

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    *,
    lr=0.001,
    max_epochs=100,
    patience=15,
    max_norm=5.0,
    checkpoint_path="best_model.pt",
    device=None,
):

    """
    Train T-GCN model.

    Args:
        model:
            TGCN instance.

        train_loader:
            DataLoader yielding:
                x: (B, T, N, F)
                y: (B, N, output_steps)

        val_loader:
            Validation DataLoader.

        lr:
            Adam learning rate.

        max_epochs:
            Maximum epochs.

        patience:
            Early stopping patience.

        max_norm:
            Gradient clipping threshold.

        checkpoint_path:
            Best model save path.

        device:
            torch.device.
            Auto-detects if None.

    Returns:
        history dictionary.
    """

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
    # EARLY STOPPING STATE
    # ────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None

    checkpoint_path = pathlib.Path(
        checkpoint_path
    )

    # ────────────────────────────────────────────────────────────
    # LOG HEADER
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

    print("─" * 70)

    # ────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ────────────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):

        t0 = time.time()

        # ========================================================
        # TRAIN
        # ========================================================
        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size = x_batch.size(0)

            optimizer.zero_grad()

            preds = model(x_batch)

            loss = mse_loss(
                preds,
                y_batch
            )

            loss.backward()

            # ────────────────────────────────────────────────
            # GRADIENT CLIPPING
            # IMPORTANT FOR RNN / GRU STABILITY
            # ────────────────────────────────────────────────
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
            )

            optimizer.step()

            # Weighted accumulation
            train_loss_sum += (
                loss.item()
                * batch_size
            )

            train_samples += batch_size

        # Empty loader guard
        if train_samples == 0:

            raise ValueError(
                "train_loader is empty"
            )

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
                    y_batch
                )

                mae = mae_loss(
                    preds,
                    y_batch
                )

                # Weighted accumulation
                val_loss_sum += (
                    mse.item()
                    * batch_size
                )

                val_mae_sum += (
                    mae.item()
                    * batch_size
                )

                val_samples += batch_size

        # Empty loader guard
        if val_samples == 0:

            raise ValueError(
                "val_loader is empty"
            )

        avg_val_loss = (
            val_loss_sum
            / val_samples
        )

        avg_val_mae = (
            val_mae_sum
            / val_samples
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        elapsed = time.time() - t0

        # ========================================================
        # LOG HISTORY
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
        # CHECKPOINTING
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
                    "model_state_dict": best_weights,
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_mae": avg_val_mae,
                    "lr": current_lr,
                },
                checkpoint_path,
            )

        else:

            epochs_no_improve += 1

        # ========================================================
        # PRINT EPOCH LOG
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
                f"at epoch {epoch} "
                f"(no improvement for "
                f"{patience} epochs)."
            )

            break

    # ────────────────────────────────────────────────────────────
    # RESTORE BEST WEIGHTS
    # ────────────────────────────────────────────────────────────
    if best_weights is not None:

        model.load_state_dict(
            best_weights
        )

    # ────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ────────────────────────────────────────────────────────────
    print()

    print("─" * 70)

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
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():

    from torch.utils.data import (
        TensorDataset,
        DataLoader,
    )

    # IMPORTANT:
    # use YOUR actual filename
    from gru_component import TGCNCell
    from tgcn_model import (
        TGCN,
        build_toy_graph,
    )

    print("=" * 70)
    print("TRAIN LOOP SMOKE TEST")
    print("=" * 70)

    # ────────────────────────────────────────────────────────
    # CONFIG
    # ────────────────────────────────────────────────────────
    BATCH_TRAIN = 32
    BATCH_VAL = 8

    T = 24
    N = 15
    F_IN = 8
    HIDDEN = 32
    OUT = 1

    N_TRAIN = 128
    N_VAL = 32

    torch.manual_seed(42)

    # ────────────────────────────────────────────────────────
    # GRAPH
    # ────────────────────────────────────────────────────────
    edge_index, edge_weight = build_toy_graph(
        n_nodes=N
    )

    # ────────────────────────────────────────────────────────
    # MODEL
    # ────────────────────────────────────────────────────────
    model = TGCN(
        in_channels=F_IN,
        hidden_dim=HIDDEN,
        output_steps=OUT,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
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
        f"\nTotal params: "
        f"{total_params:,}"
    )

    # ────────────────────────────────────────────────────────
    # SYNTHETIC DATA
    # ────────────────────────────────────────────────────────
    X_train = torch.rand(
        N_TRAIN,
        T,
        N,
        F_IN,
    )

    y_train = torch.rand(
        N_TRAIN,
        N,
        OUT,
    )

    X_val = torch.rand(
        N_VAL,
        T,
        N,
        F_IN,
    )

    y_val = torch.rand(
        N_VAL,
        N,
        OUT,
    )

    # ────────────────────────────────────────────────────────
    # DATALOADERS
    # ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train,
        ),
        batch_size=BATCH_TRAIN,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            X_val,
            y_val,
        ),
        batch_size=BATCH_VAL,
        shuffle=False,
    )

    # ────────────────────────────────────────────────────────
    # TRAIN
    # ────────────────────────────────────────────────────────
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        max_epochs=10,
        patience=15,
        max_norm=5.0,
        checkpoint_path="best_model_smoke.pt",
    )

    # ────────────────────────────────────────────────────────
    # HISTORY CHECKS
    # ────────────────────────────────────────────────────────
    assert (
        len(history["train_loss"]) > 0
    )

    assert (
        len(history["train_loss"])
        ==
        len(history["val_loss"])
    )

    assert (
        len(history["train_loss"])
        ==
        len(history["val_mae"])
    )

    print()

    print(
        "[PASS ✓] "
        "History lengths consistent"
    )

    print(
        f"[PASS ✓] "
        f"Completed "
        f"{len(history['train_loss'])} epochs"
    )

    print()

    print("=" * 70)

    print("SMOKE TEST PASSED")

    print("=" * 70)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    smoke_test()
