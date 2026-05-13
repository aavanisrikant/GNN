import argparse
import pathlib
import textwrap
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT YOUR MODULES
# ─────────────────────────────────────────────────────────────────────────────

# IMPORTANT:
# make sure these filenames MATCH your actual files

from gru_component import TGCNCell
from tgcn_model import TGCN, build_toy_graph
from training_loop import train


# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_nan_inf(tensor, name, fix=True):

    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()

    if n_nan == 0 and n_inf == 0:

        print(f"[PASS ✓] {name}: no NaN / Inf")

        return tensor

    print(f"[WARN ⚠] {name}: {n_nan} NaN, {n_inf} Inf")

    if not fix:

        return tensor

    tensor = tensor.clone()

    tensor[torch.isinf(tensor)] = float("nan")

    for f in range(tensor.shape[-1]):

        mask_nan = torch.isnan(tensor[..., f])

        tensor[..., f][mask_nan] = 0.0

    print(f"           → replaced NaN/Inf with 0")

    return tensor


def check_feature_scale(tensor, name, warn_max=10.0):

    fmax = tensor.abs().max().item()
    fmean = tensor.float().mean().item()

    if fmax > warn_max:

        print(
            f"[WARN ⚠] {name}: "
            f"max abs value = {fmax:.3f}"
        )

        print(
            "          values appear un-normalised"
        )

    else:

        print(
            f"[PASS ✓] {name}: "
            f"max={fmax:.3f} "
            f"mean={fmean:.3f}"
        )


def normalise(X_train, X_val):

    flat = X_train.reshape(
        -1,
        X_train.shape[-1]
    )

    f_min = flat.min(dim=0).values
    f_max = flat.max(dim=0).values

    rng = f_max - f_min

    rng[rng == 0] = 1.0

    X_train_norm = (
        (X_train - f_min)
        / rng
    )

    X_val_norm = (
        (X_val - f_min)
        / rng
    )

    X_val_norm = X_val_norm.clamp(
        0.0,
        1.0
    )

    stats = {
        "min": f_min,
        "max": f_max,
        "range": rng,
    }

    return (
        X_train_norm,
        X_val_norm,
        stats,
    )


def check_gradient_flow(
    model,
    x_sample,
    y_sample,
    device,
):

    model = model.to(device)

    x_sample = x_sample.to(device)
    y_sample = y_sample.to(device)

    criterion = nn.MSELoss()

    model.train()

    preds = model(x_sample)

    loss = criterion(
        preds,
        y_sample,
    )

    loss.backward()

    print("\nGradient norms:")

    dead = False
    exploding = False

    for name, param in model.named_parameters():

        if param.grad is None:

            print(
                f"{name:<45} NO GRAD"
            )

            continue

        norm = (
            param.grad.norm().item()
        )

        flag = ""

        if norm == 0.0:

            dead = True

            flag = " ← DEAD"

        elif norm > 10.0:

            exploding = True

            flag = (
                f" ← LARGE ({norm:.2f})"
            )

        print(
            f"{name:<45} "
            f"{norm:.6f}"
            f"{flag}"
        )

    if not dead and not exploding:

        print(
            "\n[PASS ✓] "
            "Gradient flow healthy"
        )

    for p in model.parameters():

        if p.grad is not None:

            p.grad.zero_()


def check_output_range(
    model,
    x_sample,
    device,
    y_mean,
    y_std,
):

    model.eval()

    with torch.no_grad():

        preds = model(
            x_sample.to(device)
        ).cpu()

    pmin = preds.min().item()
    pmax = preds.max().item()
    pstd = preds.std().item()

    print(
        f"\nPrediction range: "
        f"[{pmin:.3f}, {pmax:.3f}]"
    )

    print(
        f"Prediction std : "
        f"{pstd:.6f}"
    )

    print(
        f"Target mean/std: "
        f"{y_mean:.3f} / {y_std:.3f}"
    )

    if pstd < 1e-4:

        print(
            "[WARN ⚠] collapsed predictions"
        )

    else:

        print(
            "[PASS ✓] output variance healthy"
        )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets):

    mae = np.mean(
        np.abs(preds - targets)
    )

    rmse = np.sqrt(
        np.mean((preds - targets) ** 2)
    )

    mask = targets > 1.0

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (preds[mask] - targets[mask])
                / targets[mask]
            )
        ) * 100

    else:

        mape = np.nan

    ss_res = np.sum(
        (targets - preds) ** 2
    )

    ss_tot = np.sum(
        (targets - np.mean(targets)) ** 2
    )

    if ss_tot > 0:

        r2 = 1 - ss_res / ss_tot

    else:

        r2 = 0.0

    return mae, rmse, mape, r2


def evaluate_model(
    model,
    loader,
    device,
):

    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            preds = model(xb)

            preds_all.append(
                preds.cpu().numpy()
            )

            targets_all.append(
                yb.numpy()
            )

    preds = np.concatenate(
        preds_all,
        axis=0,
    )

    targets = np.concatenate(
        targets_all,
        axis=0,
    )

    return compute_metrics(
        preds,
        targets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_to_baseline(
    val_maes,
    lstm_mae,
    target_epoch=50,
):

    best_mae = min(val_maes)

    best_epoch = (
        val_maes.index(best_mae)
        + 1
    )

    print()

    print("═" * 66)
    print("BASELINE COMPARISON")
    print("═" * 66)

    print(
        f"LSTM baseline MAE : "
        f"{lstm_mae:.4f}"
    )

    print(
        f"T-GCN best MAE    : "
        f"{best_mae:.4f}"
    )

    print(
        f"Best epoch         : "
        f"{best_epoch}"
    )

    if (
        best_mae < lstm_mae
        and best_epoch <= target_epoch
    ):

        improvement = (
            lstm_mae - best_mae
        )

        print()

        print(
            f"[PASS ✓] "
            f"Beat LSTM baseline "
            f"by {improvement:.4f}"
        )

    else:

        print()

        print(
            "[FAIL ✗] "
            "Did not beat baseline"
        )

        print()

        print(
            "Debug suggestions:"
        )

        print(
            "  • lower lr → 0.0005"
        )

        print(
            "  • increase dropout"
        )

        print(
            "  • verify input shapes"
        )

        print(
            "  • check graph connectivity"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):

    print("=" * 70)
    print("DAY 27 — T-GCN TRAINING")
    print("=" * 70)

    # ────────────────────────────────────────────────────────
    # DEVICE
    # ────────────────────────────────────────────────────────
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Device: {device}")

    # ────────────────────────────────────────────────────────
    # LOAD DATA
    # ────────────────────────────────────────────────────────
    if args.X_train is not None:

        print("\nLoading dataset...")

        X_train = torch.tensor(
            np.load(args.X_train),
            dtype=torch.float32,
        )

        y_train = torch.tensor(
            np.load(args.y_train),
            dtype=torch.float32,
        )

        X_val = torch.tensor(
            np.load(args.X_val),
            dtype=torch.float32,
        )

        y_val = torch.tensor(
            np.load(args.y_val),
            dtype=torch.float32,
        )

    else:

        print(
            "\nNo dataset provided."
        )

        print(
            "Using synthetic data."
        )

        torch.manual_seed(42)

        B_train = 512
        B_val = 128

        T = 24
        N = 15
        F = 8

        X_train = torch.rand(
            B_train,
            T,
            N,
            F,
        )

        y_train = torch.rand(
            B_train,
            N,
            1,
        )

        X_val = torch.rand(
            B_val,
            T,
            N,
            F,
        )

        y_val = torch.rand(
            B_val,
            N,
            1,
        )

    # ────────────────────────────────────────────────────────
    # SHAPE VERIFICATION
    # ────────────────────────────────────────────────────────
    print("\nDataset shapes:")

    print(
        f"X_train: "
        f"{tuple(X_train.shape)}"
    )

    print(
        f"y_train: "
        f"{tuple(y_train.shape)}"
    )

    print(
        f"X_val  : "
        f"{tuple(X_val.shape)}"
    )

    print(
        f"y_val  : "
        f"{tuple(y_val.shape)}"
    )

    if X_train.ndim != 4:

        raise ValueError(
            "Expected X_train shape "
            "(B,T,N,F)"
        )

    if y_train.ndim != 3:

        raise ValueError(
            "Expected y_train shape "
            "(B,N,out)"
        )

    B, T, N, F = X_train.shape

    OUT = y_train.shape[-1]

    # ────────────────────────────────────────────────────────
    # PRE-FLIGHT CHECKS
    # ────────────────────────────────────────────────────────
    print("\n── NaN / Inf Check ──")

    X_train = check_nan_inf(
        X_train,
        "X_train",
    )

    X_val = check_nan_inf(
        X_val,
        "X_val",
    )

    y_train = check_nan_inf(
        y_train,
        "y_train",
        fix=False,
    )

    y_val = check_nan_inf(
        y_val,
        "y_val",
        fix=False,
    )

    print("\n── Feature Scale Check ──")

    check_feature_scale(
        X_train,
        "X_train",
    )

    # ────────────────────────────────────────────────────────
    # AUTO NORMALISATION
    # ────────────────────────────────────────────────────────
    max_val = (
        X_train.abs().max().item()
    )

    if (
        args.auto_normalise
        or max_val > 10.0
    ):

        print(
            "\nApplying min-max normalisation..."
        )

        (
            X_train,
            X_val,
            norm_stats,
        ) = normalise(
            X_train,
            X_val,
        )

        print(
            "[PASS ✓] "
            "normalisation complete"
        )

    # ────────────────────────────────────────────────────────
    # GRAPH
    # ────────────────────────────────────────────────────────
    if args.edge_index is not None:

        edge_index = torch.tensor(
            np.load(args.edge_index),
            dtype=torch.long,
        )

        edge_weight = torch.tensor(
            np.load(args.edge_weight),
            dtype=torch.float32,
        )

        print(
            f"\nLoaded graph: "
            f"{edge_index.shape[1]} edges"
        )

    else:

        edge_index, edge_weight = (
            build_toy_graph(
                n_nodes=N
            )
        )

        print(
            "\nUsing toy graph"
        )

    # ────────────────────────────────────────────────────────
    # MODEL
    # ────────────────────────────────────────────────────────
    model = TGCN(
        in_channels=F,
        hidden_dim=args.hidden,
        output_steps=OUT,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
        num_layers=2,
        dropout=args.dropout,
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nModel:")
    print(model)

    print(
        f"\nParameters: "
        f"{total_params:,}"
    )

    # ────────────────────────────────────────────────────────
    # DEBUG CHECKS
    # ────────────────────────────────────────────────────────
    print("\n── Gradient Check ──")

    check_gradient_flow(
        model,
        X_train[:2],
        y_train[:2],
        device,
    )

    print("\n── Output Range Check ──")

    check_output_range(
        model,
        X_train[:2],
        device,
        y_mean=y_train.mean().item(),
        y_std=y_train.std().item(),
    )

    # ────────────────────────────────────────────────────────
    # DATALOADERS
    # ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            X_val,
            y_val,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ────────────────────────────────────────────────────────
    # TRAIN
    # ────────────────────────────────────────────────────────
    print("\n── Training ──")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        max_epochs=50,
        patience=15,
        max_norm=5.0,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # ────────────────────────────────────────────────────────
    # EVALUATE
    # ────────────────────────────────────────────────────────
    print("\n── Final Validation Metrics ──")

    val_mae, val_rmse, val_mape, val_r2 = (
        evaluate_model(
            model,
            val_loader,
            device,
        )
    )

    print(f"MAE  : {val_mae:.4f}")
    print(f"RMSE : {val_rmse:.4f}")
    print(f"MAPE : {val_mape:.2f}%")
    print(f"R²   : {val_r2:.4f}")

    # ────────────────────────────────────────────────────────
    # BASELINE COMPARISON
    # ────────────────────────────────────────────────────────
    compare_to_baseline(
        val_maes=history["val_mae"],
        lstm_mae=args.lstm_mae,
        target_epoch=50,
    )

    # ────────────────────────────────────────────────────────
    # SAVE HISTORY
    # ────────────────────────────────────────────────────────
    csv_path = (
        pathlib.Path(
            args.checkpoint
        ).stem
        + "_history.csv"
    )

    with open(csv_path, "w") as f:

        f.write(
            "epoch,train_loss,val_loss,val_mae,lr\n"
        )

        for i, (
            tl,
            vl,
            vm,
            lr_,
        ) in enumerate(
            zip(
                history["train_loss"],
                history["val_loss"],
                history["val_mae"],
                history["lr"],
            ),
            start=1,
        ):

            f.write(
                f"{i},"
                f"{tl:.6f},"
                f"{vl:.6f},"
                f"{vm:.6f},"
                f"{lr_:.8f}\n"
            )

    print(
        f"\nSaved history → {csv_path}"
    )

    print("\nDone.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Day 27 T-GCN training"
    )

    # DATA
    parser.add_argument(
        "--X_train",
        default=None,
    )

    parser.add_argument(
        "--y_train",
        default=None,
    )

    parser.add_argument(
        "--X_val",
        default=None,
    )

    parser.add_argument(
        "--y_val",
        default=None,
    )

    # GRAPH
    parser.add_argument(
        "--edge_index",
        default=None,
    )

    parser.add_argument(
        "--edge_weight",
        default=None,
    )

    # MODEL
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    # BASELINE
    parser.add_argument(
        "--lstm_mae",
        type=float,
        default=0.5,
    )

    # MISC
    parser.add_argument(
        "--checkpoint",
        default="best_tgcn_day27.pt",
    )

    parser.add_argument(
        "--auto_normalise",
        action="store_true",
    )

    args = parser.parse_args()

    # REQUIRE COMPLETE DATASET
    if args.X_train is not None:

        required = [
            args.y_train,
            args.X_val,
            args.y_val,
        ]

        if not all(required):

            parser.error(
                "Provide all dataset files."
            )

    main(args)
