import argparse
import csv
import json
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gru_component import TGCNCell
from tgcn_model import TGCN, build_toy_graph
from training_loop import train

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

FEATURE_NAMES = [
    "aqi",
    "road_density",
    "arterial_fraction",
    "industrial_fraction",
    "residential_fraction",
    "green_fraction",
    "building_density",
    "coastline_distance",
]

AQI_COLS = [0]

ROAD_COLS = [
    0,  # AQI
    1,  # road_density
    2,  # arterial_fraction
]

ALL_COLS = list(range(len(FEATURE_NAMES)))

EXPERIMENTS = {
    "A_aqi_only": {
        "label": "AQI only (no morphology)",
        "cols": AQI_COLS,
        "marker": "○",
    },

    "B_road": {
        "label": "AQI + road_density + arterial_fraction",
        "cols": ROAD_COLS,
        "marker": "◑",
    },

    "C_all": {
        "label": "AQI + all morphology features",
        "cols": ALL_COLS,
        "marker": "●",
    },
}

def select_features(X, cols):

    return X[..., cols]
def make_loaders(
    X,
    y,
    batch_size,
    shuffle=False,
):

    ds = TensorDataset(X, y)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
    )

def build_model(
    n_features,
    n_nodes,
    edge_index,
    edge_weight,
    hidden,
    dropout,
    out,
):

    return TGCN(
        in_channels=n_features,
        hidden_dim=hidden,
        output_steps=out,
        edge_index=edge_index,
        n_nodes=n_nodes,
        edge_weight=edge_weight,
        num_layers=2,
        dropout=dropout,
    )


def check_nan_inf(name, tensor):

    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()

    if n_nan > 0 or n_inf > 0:

        raise ValueError(
            f"{name} contains "
            f"{n_nan} NaN and "
            f"{n_inf} Inf values"
        )

    print(
        f"[PASS ✓] {name}: "
        f"no NaN / Inf"
    )


def evaluate_metrics(
    model,
    loader,
    device,
):

    model.eval()

    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    mae_total = 0.0
    mse_total = 0.0

    count = 0

    preds_all = []
    targets_all = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            preds = model(x)

            mae = mae_criterion(preds, y)
            mse = mse_criterion(preds, y)

            batch_size = x.size(0)

            mae_total += mae.item() * batch_size
            mse_total += mse.item() * batch_size

            count += batch_size

            preds_all.append(
                preds.cpu().numpy()
            )

            targets_all.append(
                y.cpu().numpy()
            )

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    rmse = np.sqrt(
        np.mean(
            (preds_all - targets_all) ** 2
        )
    )

    mask = targets_all > 1.0

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    preds_all[mask]
                    - targets_all[mask]
                )
                / targets_all[mask]
            )
        ) * 100

    else:

        mape = np.nan

    ss_res = np.sum(
        (targets_all - preds_all) ** 2
    )

    ss_tot = np.sum(
        (
            targets_all
            - np.mean(targets_all)
        ) ** 2
    )

    if ss_tot > 0:

        r2 = 1 - ss_res / ss_tot

    else:

        r2 = 0.0

    return {
        "mae": mae_total / count,
        "mse": mse_total / count,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }

def print_results_table(
    results,
    lstm_mae,
):

    print()
    print("═" * 78)
    print("DAY 28 — MORPHOLOGY ABLATION RESULTS")
    print("═" * 78)

    print(
        f"{'Experiment':<45}"
        f"{'Features':>10}"
        f"{'Val MAE':>12}"
        f"{'Δ vs A':>12}"
    )

    print("─" * 78)

    mae_A = results["A_aqi_only"]["mae"]

    for key, exp in EXPERIMENTS.items():

        r = results[key]

        mae = r["mae"]

        if key == "A_aqi_only":

            delta_str = "baseline"

        else:

            delta = mae - mae_A
            delta_str = f"{delta:+.4f}"

        print(
            f"{exp['marker']} "
            f"{exp['label']:<43}"
            f"{len(exp['cols']):>10}"
            f"{mae:>12.4f}"
            f"{delta_str:>12}"
        )

    print("─" * 78)

    print(
        f"{'LSTM baseline':<55}"
        f"{lstm_mae:>12.4f}"
    )

    print("═" * 78)


def interpret_results(
    results,
    lstm_mae,
):

    mae_A = results["A_aqi_only"]["mae"]
    mae_B = results["B_road"]["mae"]
    mae_C = results["C_all"]["mae"]

    print()
    print("INTERPRETATION")
    print("─" * 78)

    # Graph contribution
    if mae_A < lstm_mae:

        print(
            "✓ Graph structure alone "
            "beats LSTM baseline."
        )

    else:

        print(
            "✗ Graph structure alone "
            "does not beat LSTM."
        )

    # Road features
    road_gain = mae_A - mae_B

    if road_gain > 0.001:

        print(
            f"✓ Road morphology improves "
            f"MAE by {road_gain:.4f}"
        )

    elif road_gain < -0.001:

        print(
            f"✗ Road morphology hurts "
            f"MAE by {abs(road_gain):.4f}"
        )

    else:

        print(
            "~ Road morphology has "
            "minimal impact"
        )

    # Full morphology
    full_gain = mae_A - mae_C

    if full_gain > 0.001:

        pct = (
            full_gain / mae_A
        ) * 100

        print(
            f"✓ Full morphology improves "
            f"MAE by {full_gain:.4f} "
            f"({pct:.1f}% reduction)"
        )

    elif full_gain < -0.001:

        print(
            f"✗ Full morphology hurts "
            f"MAE by {abs(full_gain):.4f}"
        )

    else:

        print(
            "~ Full morphology has "
            "minimal impact"
        )

    print("─" * 78)

def save_results(
    results,
    out_dir,
    lstm_mae,
):

    out_dir = pathlib.Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CSV SUMMARY
    csv_path = (
        out_dir
        / "day28_ablation_results.csv"
    )

    rows = []

    mae_A = results["A_aqi_only"]["mae"]

    for key, exp in EXPERIMENTS.items():

        r = results[key]

        rows.append({
            "experiment": key,
            "label": exp["label"],
            "feature_cols": str(exp["cols"]),
            "n_features": len(exp["cols"]),
            "val_mae": round(r["mae"], 6),
            "val_mse": round(r["mse"], 6),
            "val_rmse": round(r["rmse"], 6),
            "val_r2": round(r["r2"], 6),
            "best_epoch": r["best_epoch"],
            "training_sec": round(
                r["training_sec"],
                1,
            ),
            "n_params": r["n_params"],
            "delta_vs_A": round(
                r["mae"] - mae_A,
                6,
            ),
            "beats_lstm": (
                r["mae"] < lstm_mae
            ),
        })

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    # JSON SUMMARY
    json_path = (
        out_dir
        / "day28_summary.json"
    )

    summary = {
        "lstm_mae": lstm_mae,
        "experiments": results,
    }

    with open(json_path, "w") as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved results → {out_dir}"
    )

def main(args):

    print("=" * 78)
    print("DAY 28 — MORPHOLOGY ABLATION")
    print("=" * 78)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Device: {device}")

    
    if args.X_train:

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

        T = 24
        N = 15
        F = 8

        X_train = torch.rand(
            512,
            T,
            N,
            F,
        )

        y_train = torch.rand(
            512,
            N,
            1,
        )

        X_val = torch.rand(
            128,
            T,
            N,
            F,
        )

        y_val = torch.rand(
            128,
            N,
            1,
        )

    # ────────────────────────────────────────────────────────
    # VERIFY DATA
    # ────────────────────────────────────────────────────────
    print("\nDataset shapes:")

    print(
        f"X_train: {tuple(X_train.shape)}"
    )

    print(
        f"y_train: {tuple(y_train.shape)}"
    )

    print(
        f"X_val  : {tuple(X_val.shape)}"
    )

    print(
        f"y_val  : {tuple(y_val.shape)}"
    )

    print("\nFeature mapping:")

    for i, name in enumerate(FEATURE_NAMES):

        print(f"  {i}: {name}")

    # NaN checks
    print("\nNaN / Inf checks:")

    check_nan_inf("X_train", X_train)
    check_nan_inf("y_train", y_train)
    check_nan_inf("X_val", X_val)
    check_nan_inf("y_val", y_val)

    B, T, N, F = X_train.shape

    OUT = y_train.shape[-1]

    # ────────────────────────────────────────────────────────
    # GRAPH
    # ────────────────────────────────────────────────────────
    if args.edge_index:

        edge_index = torch.tensor(
            np.load(args.edge_index),
            dtype=torch.long,
        )

        edge_weight = torch.tensor(
            np.load(args.edge_weight),
            dtype=torch.float32,
        )

    else:

        edge_index, edge_weight = (
            build_toy_graph(
                n_nodes=N
            )
        )

    print(
        f"\nGraph: {N} nodes, "
        f"{edge_index.shape[1]} edges"
    )

    # ────────────────────────────────────────────────────────
    # EXPERIMENTS
    # ────────────────────────────────────────────────────────
    results = {}

    for key, exp in EXPERIMENTS.items():

        print()
        print("─" * 78)

        print(
            f"{exp['marker']} "
            f"{exp['label']}"
        )

        print(
            f"Feature cols: "
            f"{exp['cols']}"
        )

        print("─" * 78)

        cols = exp["cols"]

        X_train_exp = select_features(
            X_train,
            cols,
        )

        X_val_exp = select_features(
            X_val,
            cols,
        )

        n_features = (
            X_train_exp.shape[-1]
        )

        train_loader = make_loaders(
            X_train_exp,
            y_train,
            batch_size=args.batch_size,
            shuffle=True,
        )

        val_loader = make_loaders(
            X_val_exp,
            y_val,
            batch_size=args.batch_size,
            shuffle=False,
        )

        model = build_model(
            n_features=n_features,
            n_nodes=N,
            edge_index=edge_index,
            edge_weight=edge_weight,
            hidden=args.hidden,
            dropout=args.dropout,
            out=OUT,
        )

        n_params = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            f"Parameters: {n_params:,}"
        )

        ckpt_path = (
            f"day28_{key}.pt"
        )

        t0 = time.time()

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            max_norm=5.0,
            checkpoint_path=ckpt_path,
            device=device,
        )

        elapsed = time.time() - t0

        # Load best checkpoint
        ckpt = torch.load(
            ckpt_path,
            map_location=device,
        )

        model.load_state_dict(
            ckpt["model_state_dict"]
        )

        metrics = evaluate_metrics(
            model,
            val_loader,
            device,
        )

        best_epoch = (
            history["val_mae"].index(
                min(history["val_mae"])
            )
            + 1
        )

        results[key] = {
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "r2": metrics["r2"],
            "best_epoch": best_epoch,
            "training_sec": elapsed,
            "n_params": n_params,
        }

        print()
        print(
            f"Final MAE : "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"Final RMSE: "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

    # ────────────────────────────────────────────────────────
    # RESULTS
    # ────────────────────────────────────────────────────────
    print_results_table(
        results,
        args.lstm_mae,
    )

    interpret_results(
        results,
        args.lstm_mae,
    )

    save_results(
        results,
        args.out_dir,
        args.lstm_mae,
    )

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Day 28 morphology ablation"
        )
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

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    # BASELINE
    parser.add_argument(
        "--lstm_mae",
        type=float,
        default=0.5,
    )

    # OUTPUT
    parser.add_argument(
        "--out_dir",
        default="day28_results",
    )

    args = parser.parse_args()

    # Validate dataset args
    if args.X_train:

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
