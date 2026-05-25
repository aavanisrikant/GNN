import argparse
import csv
import json
import pathlib
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from gru_component import (
    TGCNCell,
    TGCNGraphConv,
    build_normalised_adjacency,
)

from tgcn_model import (
    TGCN,
    build_toy_graph,
)

from training_loop import train


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAFFIC REGIMES
# ─────────────────────────────────────────────────────────────────────────────

RUSH_HOURS = (
    set(range(7, 10))
    | set(range(17, 20))
)

NIGHT_HOURS = (
    set(range(23, 24))
    | set(range(0, 6))
)

REGIME_RUSH = 0
REGIME_NORMAL = 1
REGIME_NIGHT = 2

REGIME_NAMES = {

    REGIME_RUSH:
        "rush hour",

    REGIME_NORMAL:
        "normal",

    REGIME_NIGHT:
        "night",
}

MULTIPLIERS = {

    REGIME_RUSH:
        1.5,

    REGIME_NORMAL:
        1.0,

    REGIME_NIGHT:
        0.7,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAFFIC MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

def traffic_multiplier(hour):

    """
    Return multiplier based on hour.
    """

    if hour in RUSH_HOURS:

        return MULTIPLIERS[REGIME_RUSH]

    if hour in NIGHT_HOURS:

        return MULTIPLIERS[REGIME_NIGHT]

    return MULTIPLIERS[REGIME_NORMAL]


def hour_to_regime(hour):

    """
    Convert hour -> regime id.
    """

    if hour in RUSH_HOURS:

        return REGIME_RUSH

    if hour in NIGHT_HOURS:

        return REGIME_NIGHT

    return REGIME_NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD HOURS TENSOR
# ─────────────────────────────────────────────────────────────────────────────

def make_hours_tensor(
    start_hours,
    T,
):

    """
    start_hours:
        (N_samples,)

    Returns:
        (N_samples, T)
    """

    offsets = torch.arange(
        T,
        dtype=torch.long,
    )

    hours = (
        start_hours.unsqueeze(1)
        + offsets.unsqueeze(0)
    ) % 24

    return hours


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD TEMPORAL ADJACENCIES
# IMPORTANT FIX:
# SCALE AFTER NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_adjacencies(
    edge_index,
    base_weight,
    n_nodes,
):

    """
    Build one normalised adjacency
    per traffic regime.

    IMPORTANT:
    We scale AFTER normalisation.

    This avoids the multiplier being
    cancelled by degree normalisation.
    """

    regime_adj = {}

    # Build base normalised adjacency
    base_A = build_normalised_adjacency(
        edge_index=edge_index,
        edge_weight=base_weight,
        n_nodes=n_nodes,
    )

    for regime, multiplier in MULTIPLIERS.items():

        # IMPORTANT FIX
        # scale AFTER normalisation
        scaled_A = base_A.coalesce()

        scaled_values = (
            scaled_A.values()
            * multiplier
        )

        A_scaled = torch.sparse_coo_tensor(
            scaled_A.indices(),
            scaled_values,
            scaled_A.size(),
        ).coalesce()

        regime_adj[regime] = A_scaled

    return regime_adj


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEMPORAL GRAPH CONV
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGraphConv(nn.Module):

    """
    Graph convolution with
    time-varying edge weights.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        regime_adj,
        bias=True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Register sparse adjacency
        for regime, A in regime_adj.items():

            self.register_buffer(
                f"A_norm_{regime}",
                A,
            )

        self.W = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
            )
        )

        if bias:

            self.b = nn.Parameter(
                torch.zeros(out_channels)
            )

        else:

            self.register_parameter(
                "b",
                None,
            )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.W)

        if self.b is not None:

            nn.init.zeros_(self.b)

    def forward(
        self,
        x,
        regime,
    ):

        A = getattr(
            self,
            f"A_norm_{regime}",
        )

        aggr = torch.sparse.mm(A, x)

        out = aggr @ self.W

        if self.b is not None:

            out = out + self.b

        return out

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"in={self.in_channels}, "
            f"out={self.out_channels})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. TEMPORAL TGCN CELL
# ─────────────────────────────────────────────────────────────────────────────

class TGCNTemporalCell(nn.Module):

    """
    T-GCN GRU cell with
    temporal edge weights.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        regime_adj,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        combined_dim = (
            in_channels
            + hidden_dim
        )

        self.gconv_z = TemporalGraphConv(
            combined_dim,
            hidden_dim,
            regime_adj,
        )

        self.gconv_r = TemporalGraphConv(
            combined_dim,
            hidden_dim,
            regime_adj,
        )

        self.gconv_h = TemporalGraphConv(
            combined_dim,
            hidden_dim,
            regime_adj,
        )

    def forward(
        self,
        x_t,
        h_prev,
        regime,
    ):

        combined = torch.cat(
            [x_t, h_prev],
            dim=-1,
        )

        z_t = torch.sigmoid(
            self.gconv_z(
                combined,
                regime,
            )
        )

        r_t = torch.sigmoid(
            self.gconv_r(
                combined,
                regime,
            )
        )

        reset_hidden = (
            r_t * h_prev
        )

        candidate_input = torch.cat(
            [x_t, reset_hidden],
            dim=-1,
        )

        h_tilde = torch.tanh(
            self.gconv_h(
                candidate_input,
                regime,
            )
        )

        h_t = (
            z_t * h_prev
            + (1 - z_t) * h_tilde
        )

        return h_t


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL TEMPORAL TGCN
# ─────────────────────────────────────────────────────────────────────────────

class TGCNTemporal(nn.Module):

    """
    Full T-GCN with
    time-varying edge weights.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        output_steps,
        edge_index,
        n_nodes,
        base_weight=None,
        num_layers=2,
        dropout=0.2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_steps = output_steps
        self.n_nodes = n_nodes
        self.num_layers = num_layers

        if base_weight is None:

            base_weight = torch.ones(
                edge_index.size(1),
                dtype=torch.float32,
            )

        regime_adj = build_regime_adjacencies(
            edge_index=edge_index,
            base_weight=base_weight,
            n_nodes=n_nodes,
        )

        cells = []

        for i in range(num_layers):

            layer_in = (
                in_channels
                if i == 0
                else hidden_dim
            )

            cells.append(
                TGCNTemporalCell(
                    in_channels=layer_in,
                    hidden_dim=hidden_dim,
                    regime_adj=regime_adj,
                )
            )

        self.cells = nn.ModuleList(cells)

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            output_steps,
        )

    def forward(
        self,
        x,
        hours,
    ):

        """
        x:
            (B,T,N,F)

        hours:
            (B,T)
        """

        B, T, N, F = x.shape

        outputs = []

        for b in range(B):

            x_b = x[b]

            hrs_b = hours[b]

            h = [

                torch.zeros(
                    N,
                    self.hidden_dim,
                    device=x.device,
                    dtype=x.dtype,
                )

                for _ in range(self.num_layers)
            ]

            for t in range(T):

                x_t = x_b[t]

                regime = hour_to_regime(
                    int(
                        hrs_b[t].item()
                    )
                )

                for layer_idx, cell in enumerate(self.cells):

                    h[layer_idx] = cell(
                        x_t,
                        h[layer_idx],
                        regime,
                    )

                    x_t = h[layer_idx]

            final_h = h[-1]

            out = self.dropout(
                final_h
            )

            out = self.output_layer(out)

            outputs.append(out)

        outputs = torch.stack(
            outputs,
            dim=0,
        )

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# 8. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEdgeDataset(Dataset):

    """
    Dataset wrapper for:
        X,
        hours,
        y
    """

    def __init__(
        self,
        X,
        hours,
        y,
    ):

        assert (
            X.shape[0]
            == hours.shape[0]
            == y.shape[0]
        )

        self.X = X
        self.hours = hours
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.hours[idx],
            self.y[idx],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. TEMPORAL TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_temporal(
    model,
    train_loader,
    val_loader,
    lr=0.001,
    max_epochs=50,
    patience=15,
    max_norm=5.0,
    checkpoint_path="best_temporal.pt",
    device=None,
):

    if device is None:

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    mse_fn = nn.MSELoss()

    mae_fn = nn.L1Loss()

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    )

    history = {

        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "lr": [],
    }

    best_val_loss = float("inf")

    best_weights = None

    epochs_no_improve = 0

    print()

    print(
        f"{'Epoch':>6}  "
        f"{'Train MSE':>10}  "
        f"{'Val MSE':>10}  "
        f"{'Val MAE':>10}  "
        f"{'LR':>10}"
    )

    print("─" * 64)

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        # TRAIN
        model.train()

        train_loss = 0.0

        train_n = 0

        for x, hrs, y in train_loader:

            x = x.to(device)

            hrs = hrs.to(device)

            y = y.to(device)

            optimizer.zero_grad()

            preds = model(x, hrs)

            loss = mse_fn(preds, y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm,
            )

            optimizer.step()

            train_loss += loss.item()

            train_n += 1

        # VALIDATE
        model.eval()

        val_loss = 0.0

        val_mae = 0.0

        val_n = 0

        with torch.no_grad():

            for x, hrs, y in val_loader:

                x = x.to(device)

                hrs = hrs.to(device)

                y = y.to(device)

                preds = model(x, hrs)

                val_loss += mse_fn(
                    preds,
                    y,
                ).item()

                val_mae += mae_fn(
                    preds,
                    y,
                ).item()

                val_n += 1

        avg_train = train_loss / train_n

        avg_val = val_loss / val_n

        avg_mae = val_mae / val_n

        lr_now = (
            optimizer.param_groups[0]["lr"]
        )

        history["train_loss"].append(
            avg_train
        )

        history["val_loss"].append(
            avg_val
        )

        history["val_mae"].append(
            avg_mae
        )

        history["lr"].append(
            lr_now
        )

        improved = (
            avg_val < best_val_loss
        )

        if improved:

            best_val_loss = avg_val

            epochs_no_improve = 0

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict":
                        best_weights,

                    "val_loss":
                        best_val_loss,
                },
                checkpoint_path,
            )

        else:

            epochs_no_improve += 1

        print(
            f"{epoch:>6}  "
            f"{avg_train:>10.5f}  "
            f"{avg_val:>10.5f}  "
            f"{avg_mae:>10.5f}  "
            f"{lr_now:>10.6f}"
            f"{' ✓' if improved else ''}"
        )

        scheduler.step(avg_val)

        if epochs_no_improve >= patience:

            print(
                f"\nEarly stopping "
                f"at epoch {epoch}"
            )

            break

    if best_weights is not None:

        model.load_state_dict(
            best_weights
        )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 10. SAFE JSON
# ─────────────────────────────────────────────────────────────────────────────

def convert_json_safe(obj):

    if isinstance(obj, dict):

        return {
            k: convert_json_safe(v)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):

        return [
            convert_json_safe(v)
            for v in obj
        ]

    elif isinstance(obj, tuple):

        return tuple(
            convert_json_safe(v)
            for v in obj
        )

    elif isinstance(obj, np.integer):

        return int(obj)

    elif isinstance(obj, np.floating):

        return float(obj)

    elif isinstance(obj, np.ndarray):

        return obj.tolist()

    elif isinstance(obj, torch.Tensor):

        return (
            obj.detach()
            .cpu()
            .tolist()
        )

    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 11. RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(
    static_mae,
    temporal_mae,
    lstm_mae,
):

    delta = (
        static_mae
        - temporal_mae
    )

    delta_pct = (
        (delta / static_mae) * 100
        if static_mae > 0
        else 0.0
    )

    print()

    print("═" * 70)
    print("DAY 29 — TEMPORAL EDGE WEIGHTS")
    print("═" * 70)

    print(
        f"{'Model':<40}"
        f"{'Val MAE':>12}"
    )

    print("─" * 70)

    print(
        f"{'LSTM baseline':<40}"
        f"{lstm_mae:>12.4f}"
    )

    print(
        f"{'Static edges':<40}"
        f"{static_mae:>12.4f}"
    )

    print(
        f"{'Temporal edges':<40}"
        f"{temporal_mae:>12.4f}"
    )

    print("─" * 70)

    if delta > 0:

        print(
            f"\n✓ Temporal edges improved "
            f"MAE by {delta:.4f} "
            f"({delta_pct:.1f}%)"
        )

    elif delta < -0.001:

        print(
            f"\n✗ Temporal edges hurt "
            f"MAE by {abs(delta):.4f}"
        )

    else:

        print(
            "\n~ No meaningful difference"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    static_mae,
    temporal_mae,
    lstm_mae,
    static_history,
    temporal_history,
    out_dir,
):

    out_dir = pathlib.Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # SUMMARY CSV
    csv_path = (
        out_dir
        / "day29_comparison.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "model",
            "val_mae",
        ])

        writer.writerow([
            "lstm_baseline",
            float(lstm_mae),
        ])

        writer.writerow([
            "static_edges",
            float(static_mae),
        ])

        writer.writerow([
            "temporal_edges",
            float(temporal_mae),
        ])

    # JSON
    summary = {

        "lstm_mae":
            float(lstm_mae),

        "static_mae":
            float(static_mae),

        "temporal_mae":
            float(temporal_mae),

        "delta_mae":
            float(
                round(
                    static_mae
                    - temporal_mae,
                    6,
                )
            ),
    }

    safe_summary = convert_json_safe(
        summary
    )

    json_path = (
        out_dir
        / "day29_summary.json"
    )

    with open(json_path, "w") as f:

        json.dump(
            safe_summary,
            f,
            indent=2,
        )

    print(
        f"\nSaved results → "
        f"{out_dir}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):

    print("=" * 70)
    print("DAY 29 — TEMPORAL EDGE WEIGHTS")
    print("=" * 70)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Device: {device}")

    # DATA
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

        if args.hours_train:

            sh_train = torch.tensor(
                np.load(args.hours_train),
                dtype=torch.long,
            )

            sh_val = torch.tensor(
                np.load(args.hours_val),
                dtype=torch.long,
            )

        else:

            print(
                "No hours file."
            )

            sh_train = (
                torch.arange(
                    X_train.shape[0]
                ) % 24
            )

            sh_val = (
                torch.arange(
                    X_val.shape[0]
                ) % 24
            )

    else:

        print(
            "\nUsing synthetic data."
        )

        T = 24
        N = 15
        F = 8

        torch.manual_seed(42)

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

        sh_train = (
            torch.arange(512) % 24
        )

        sh_val = (
            torch.arange(128) % 24
        )

    B, T, N, F = X_train.shape

    OUT = y_train.shape[-1]

    print(
        f"\nX_train: "
        f"{tuple(X_train.shape)}"
    )

    print(
        f"y_train: "
        f"{tuple(y_train.shape)}"
    )

    # GRAPH
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
        f"\nGraph: "
        f"{N} nodes, "
        f"{edge_index.shape[1]} edges"
    )

    # HOURS
    hours_train = make_hours_tensor(
        sh_train,
        T,
    )

    hours_val = make_hours_tensor(
        sh_val,
        T,
    )

    # STATIC MODEL
    print(
        "\nTraining static-edge model..."
    )

    static_model = TGCN(
        in_channels=F,
        hidden_dim=args.hidden,
        output_steps=OUT,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
        num_layers=2,
        dropout=args.dropout,
    )

    static_train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    static_val_loader = DataLoader(
        TensorDataset(
            X_val,
            y_val,
        ),
        batch_size=args.batch_size,
    )

    static_history = train(
        model=static_model,
        train_loader=static_train_loader,
        val_loader=static_val_loader,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_norm=5.0,
        checkpoint_path="day29_static.pt",
        device=device,
    )

    static_mae = min(
        static_history["val_mae"]
    )

    # TEMPORAL MODEL
    print(
        "\nTraining temporal-edge model..."
    )

    temporal_model = TGCNTemporal(
        in_channels=F,
        hidden_dim=args.hidden,
        output_steps=OUT,
        edge_index=edge_index,
        n_nodes=N,
        base_weight=edge_weight,
        num_layers=2,
        dropout=args.dropout,
    )

    temporal_train_ds = (
        TemporalEdgeDataset(
            X_train,
            hours_train,
            y_train,
        )
    )

    temporal_val_ds = (
        TemporalEdgeDataset(
            X_val,
            hours_val,
            y_val,
        )
    )

    temporal_train_loader = DataLoader(
        temporal_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    temporal_val_loader = DataLoader(
        temporal_val_ds,
        batch_size=args.batch_size,
    )

    temporal_history = train_temporal(
        model=temporal_model,
        train_loader=temporal_train_loader,
        val_loader=temporal_val_loader,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_norm=5.0,
        checkpoint_path="day29_temporal.pt",
        device=device,
    )

    temporal_mae = min(
        temporal_history["val_mae"]
    )

    # RESULTS
    print_comparison(
        static_mae,
        temporal_mae,
        args.lstm_mae,
    )

    save_results(
        static_mae,
        temporal_mae,
        args.lstm_mae,
        static_history,
        temporal_history,
        out_dir=args.out_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 14. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Day 29 — Temporal edge weights"
        )
    )

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

    parser.add_argument(
        "--hours_train",
        default=None,
    )

    parser.add_argument(
        "--hours_val",
        default=None,
    )

    parser.add_argument(
        "--edge_index",
        default=None,
    )

    parser.add_argument(
        "--edge_weight",
        default=None,
    )

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

    parser.add_argument(
        "--lstm_mae",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--out_dir",
        default="day29_results",
    )

    args = parser.parse_args()

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

    if args.hours_train and not args.hours_val:

        parser.error(
            "Provide both "
            "--hours_train and --hours_val"
        )

    main(args)
