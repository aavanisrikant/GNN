import argparse
import csv
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.nn import GATConv

from tgcn_model import TGCN, build_toy_graph
from training_loop import train

# JSON CONVERSION

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

# graph conversion

class GATGraphConv(nn.Module):

    """
    PyG GATConv wrapper matching
    TGCNGraphConv-style interface.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=4,
        dropout=0.0,
        concat=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            add_self_loops=True,
        )

    def forward(
        self,
        x,
        edge_index,
        return_attn=False,
    ):

        if return_attn:

            out, (ei, alpha) = self.gat(
                x,
                edge_index,
                return_attention_weights=True,
            )

            return out, (ei, alpha)

        out = self.gat(
            x,
            edge_index,
        )

        return out

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"in={self.in_channels}, "
            f"out={self.out_channels}, "
            f"heads={self.heads})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. TGAT CELL
# ─────────────────────────────────────────────────────────────────────────────

class TGATCell(nn.Module):

    """
    T-GCN GRU cell using GATConv.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        edge_index,
        heads=4,
        gat_dropout=0.0,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.register_buffer(
            "edge_index",
            edge_index,
        )

        combined_dim = (
            in_channels
            + hidden_dim
        )

        self.gat_z = GATGraphConv(
            combined_dim,
            hidden_dim,
            heads,
            gat_dropout,
        )

        self.gat_r = GATGraphConv(
            combined_dim,
            hidden_dim,
            heads,
            gat_dropout,
        )

        self.gat_h = GATGraphConv(
            combined_dim,
            hidden_dim,
            heads,
            gat_dropout,
        )

    def forward(
        self,
        x_t,
        h_prev,
        return_attn=False,
    ):

        ei = self.edge_index

        combined = torch.cat(
            [x_t, h_prev],
            dim=-1,
        )

        z_t = torch.sigmoid(
            self.gat_z(
                combined,
                ei,
            )
        )

        r_t = torch.sigmoid(
            self.gat_r(
                combined,
                ei,
            )
        )

        reset_hidden = (
            r_t * h_prev
        )

        cand_input = torch.cat(
            [x_t, reset_hidden],
            dim=-1,
        )

        if return_attn:

            h_raw, attn_info = self.gat_h(
                cand_input,
                ei,
                return_attn=True,
            )

            h_tilde = torch.tanh(
                h_raw
            )

            h_t = (
                z_t * h_prev
                + (1 - z_t) * h_tilde
            )

            return h_t, attn_info

        h_tilde = torch.tanh(
            self.gat_h(
                cand_input,
                ei,
            )
        )

        h_t = (
            z_t * h_prev
            + (1 - z_t) * h_tilde
        )

        return h_t


# ─────────────────────────────────────────────────────────────────────────────
# 4. FULL TGAT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TGATModel(nn.Module):

    """
    Full T-GAT model.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        output_steps,
        edge_index,
        n_nodes,
        heads=4,
        gat_dropout=0.1,
        num_layers=2,
        dropout=0.2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_steps = output_steps
        self.n_nodes = n_nodes
        self.num_layers = num_layers

        cells = []

        for i in range(num_layers):

            layer_in = (
                in_channels
                if i == 0
                else hidden_dim
            )

            cells.append(

                TGATCell(
                    in_channels=layer_in,
                    hidden_dim=hidden_dim,
                    edge_index=edge_index,
                    heads=heads,
                    gat_dropout=gat_dropout,
                )
            )

        self.cells = nn.ModuleList(
            cells
        )

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
        return_attn=False,
    ):

        B, T, N, F = x.shape

        outputs = []

        attn_list = []

        for b in range(B):

            x_b = x[b]

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

                for layer_idx, cell in enumerate(self.cells):

                    capture = (
                        return_attn
                        and layer_idx == self.num_layers - 1
                        and t == T - 1
                    )

                    if capture:

                        h[layer_idx], attn_info = cell(
                            x_t,
                            h[layer_idx],
                            return_attn=True,
                        )

                        attn_list.append(
                            attn_info[1].detach()
                        )

                    else:

                        h[layer_idx] = cell(
                            x_t,
                            h[layer_idx],
                        )

                    x_t = h[layer_idx]

            final_h = h[-1]

            out = self.dropout(
                final_h
            )

            out = self.output_layer(
                out
            )

            outputs.append(out)

        outputs = torch.stack(
            outputs,
            dim=0,
        )

        if return_attn and len(attn_list) > 0:

            alpha = torch.stack(
                attn_list
            ).mean(dim=0)

            ei = self.cells[-1].edge_index

            return outputs, (ei, alpha)

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXTRACT ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_attention_weights(
    model,
    X_sample,
    device,
):

    model.eval()

    model = model.to(device)

    with torch.no_grad():

        _, (ei, alpha) = model(
            X_sample.to(device),
            return_attn=True,
        )

    # SAFER MULTI-HEAD HANDLING
    if alpha.dim() == 1:

        alpha = alpha.unsqueeze(-1)

    alpha_mean = alpha.mean(dim=1)

    return (
        ei.cpu(),
        alpha_mean.cpu(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. TOP-K REPORT
# ─────────────────────────────────────────────────────────────────────────────

def report_top_attention(
    edge_index,
    alpha,
    node_names,
    top_k=10,
):

    alpha_np = alpha.numpy()

    sorted_idx = np.argsort(
        alpha_np
    )[::-1]

    print()
    print("TOP ATTENTION EDGES")
    print("─" * 60)

    print(
        f"{'Rank':<5} "
        f"{'Node i':<20} "
        f"{'Node j':<20} "
        f"{'α':>8}"
    )

    print("─" * 60)

    for rank, eidx in enumerate(
        sorted_idx[:top_k],
        start=1,
    ):

        i = edge_index[0, eidx].item()

        j = edge_index[1, eidx].item()

        ni = node_names[i]

        nj = node_names[j]

        a = alpha_np[eidx]

        print(
            f"{rank:<5} "
            f"{ni:<20} "
            f"{nj:<20} "
            f"{a:>8.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. SYNTHETIC MUMBAI COORDS
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_mumbai_coords(
    n_nodes,
):

    np.random.seed(0)

    lats = np.random.uniform(
        18.87,
        19.27,
        n_nodes,
    )

    lons = np.random.uniform(
        72.77,
        72.99,
        n_nodes,
    )

    return np.stack(
        [lats, lons],
        axis=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. ATTENTION MAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_map(
    edge_index,
    alpha,
    node_coords,
    node_names=None,
    top_k=10,
    out_path="day30_attention_map.png",
):

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

    except ImportError:

        print(
            "[SKIP] matplotlib unavailable"
        )

        return

    xs = node_coords[:, 1]

    ys = node_coords[:, 0]

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    alpha_np = alpha.numpy()

    norm = mcolors.Normalize(
        vmin=float(alpha_np.min()),
        vmax=float(alpha_np.max()),
    )

    cmap = cm.get_cmap("YlOrRd")

    E = edge_index.shape[1]

    for eidx in range(E):

        i = edge_index[0, eidx].item()

        j = edge_index[1, eidx].item()

        a = alpha_np[eidx]

        colour = cmap(norm(a))

        lw = 0.5 + 4 * norm(a)

        ax.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            color=colour,
            linewidth=lw,
            alpha=0.8,
        )

    ax.scatter(
        xs,
        ys,
        s=60,
    )

    if node_names is not None:

        for i, name in enumerate(node_names):

            ax.text(
                xs[i],
                ys[i],
                name,
                fontsize=7,
            )

    plt.title(
        "T-GAT Attention Map"
    )

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=180,
    )

    plt.close()

    print(
        f"Attention map saved → {out_path}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(
    tgcn_mae,
    tgat_mae,
    lstm_mae,
):

    delta = (
        tgcn_mae
        - tgat_mae
    )

    delta_pct = (
        (delta / tgcn_mae) * 100
        if tgcn_mae > 0
        else 0
    )

    print()
    print("═" * 66)
    print("DAY 30 — T-GAT vs FIXED-WEIGHT T-GCN")
    print("═" * 66)

    print(
        f"{'Model':<40}"
        f"{'Val MAE':>12}"
    )

    print("─" * 66)

    print(
        f"{'LSTM baseline':<40}"
        f"{lstm_mae:>12.4f}"
    )

    print(
        f"{'T-GCN fixed':<40}"
        f"{tgcn_mae:>12.4f}"
    )

    print(
        f"{'T-GAT attention':<40}"
        f"{tgat_mae:>12.4f}"
    )

    print("─" * 66)

    if delta > 0.001:

        print(
            f"\nAttention improved MAE "
            f"by {delta:.4f} "
            f"({delta_pct:.1f}%)"
        )

    elif delta < -0.001:

        print(
            f"\nFixed weights outperform "
            f"attention by "
            f"{abs(delta):.4f}"
        )

    else:

        print(
            "\nNegligible difference."
        )

    print("═" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    tgcn_mae,
    tgat_mae,
    lstm_mae,
    tgcn_history,
    tgat_history,
    edge_index,
    alpha,
    node_names,
    out_dir,
):

    out_dir = pathlib.Path(
        out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # SUMMARY CSV
    with open(
        out_dir / "day30_comparison.csv",
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
            "tgcn_fixed",
            float(tgcn_mae),
        ])

        writer.writerow([
            "tgat_attention",
            float(tgat_mae),
        ])

    # ATTENTION CSV
    alpha_np = alpha.numpy()

    with open(
        out_dir / "day30_attention_weights.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "edge_idx",
            "node_i",
            "node_j",
            "name_i",
            "name_j",
            "alpha",
        ])

        for eidx in range(
            edge_index.shape[1]
        ):

            i = edge_index[
                0,
                eidx,
            ].item()

            j = edge_index[
                1,
                eidx,
            ].item()

            writer.writerow([

                eidx,
                i,
                j,
                node_names[i],
                node_names[j],
                round(
                    float(alpha_np[eidx]),
                    6,
                ),
            ])

    # JSON SAFE
    summary = {

        "lstm_mae":
            float(lstm_mae),

        "tgcn_mae":
            float(tgcn_mae),

        "tgat_mae":
            float(tgat_mae),

        "delta_mae":
            float(
                round(
                    tgcn_mae
                    - tgat_mae,
                    6,
                )
            ),
    }

    safe_summary = convert_json_safe(
        summary
    )

    with open(
        out_dir / "day30_summary.json",
        "w",
    ) as f:

        json.dump(
            safe_summary,
            f,
            indent=2,
        )

    print(
        f"\nResults saved → {out_dir}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):

    print("=" * 66)
    print("DAY 30 — GRAPH ATTENTION NETWORK")
    print("=" * 66)

    device = (

        torch.device("cuda")

        if torch.cuda.is_available()

        else torch.device("cpu")
    )

    print(f"Device: {device}")

    # DATA
    if args.X_train:

        print("\nLoading data...")

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
            "\nUsing synthetic data."
        )

        T, N, F = 24, 15, 8

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

    B, T, N, F = X_train.shape

    OUT = y_train.shape[-1]

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

    # COORDS
    if args.node_coords:

        node_coords = np.load(
            args.node_coords
        )

    else:

        node_coords = synthetic_mumbai_coords(
            N
        )

    # NODE NAMES
    if args.node_names:

        with open(
            args.node_names
        ) as f:

            node_names = [

                line.strip()

                for line in f.readlines()

            ][:N]

    else:

        node_names = [
            f"Station_{i}"
            for i in range(N)
        ]

    # DATALOADERS
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
    )

    # TRAIN TGCN
    print(
        "\nTraining fixed-weight T-GCN..."
    )

    tgcn = TGCN(
        in_channels=F,
        hidden_dim=args.hidden,
        output_steps=OUT,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
        num_layers=2,
        dropout=args.dropout,
    )

    tgcn_history = train(
        model=tgcn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_norm=5.0,
        checkpoint_path="day30_tgcn.pt",
        device=device,
    )

    tgcn_mae = min(
        tgcn_history["val_mae"]
    )

    # TRAIN TGAT
    print(
        "\nTraining T-GAT..."
    )

    tgat = TGATModel(
        in_channels=F,
        hidden_dim=args.hidden,
        output_steps=OUT,
        edge_index=edge_index,
        n_nodes=N,
        heads=args.heads,
        gat_dropout=args.gat_dropout,
        num_layers=2,
        dropout=args.dropout,
    )

    tgat_history = train(
        model=tgat,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_norm=5.0,
        checkpoint_path="day30_tgat.pt",
        device=device,
    )

    tgat_mae = min(
        tgat_history["val_mae"]
    )

    # RESULTS
    print_comparison(
        tgcn_mae,
        tgat_mae,
        args.lstm_mae,
    )

    # LOAD BEST MODEL
    ckpt = torch.load(
        "day30_tgat.pt",
        map_location=device,
    )

    tgat.load_state_dict(
        ckpt["model_state_dict"]
    )

    # EXTRACT ATTENTION
    print(
        "\nExtracting attention..."
    )

    X_sample = X_val[
        :min(16, len(X_val))
    ]

    ei, alpha = extract_attention_weights(
        tgat,
        X_sample,
        device,
    )

    report_top_attention(
        ei,
        alpha,
        node_names,
        top_k=10,
    )

    # PLOT MAP
    out_dir = pathlib.Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_attention_map(
        ei,
        alpha,
        node_coords,
        node_names,
        top_k=10,
        out_path=str(
            out_dir
            / "day30_attention_map.png"
        ),
    )

    # SAVE
    save_results(
        tgcn_mae,
        tgat_mae,
        args.lstm_mae,
        tgcn_history,
        tgat_history,
        ei,
        alpha,
        node_names,
        args.out_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 12. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Day 30 — T-GAT"
    )

    p.add_argument(
        "--X_train",
        default=None,
    )

    p.add_argument(
        "--y_train",
        default=None,
    )

    p.add_argument(
        "--X_val",
        default=None,
    )

    p.add_argument(
        "--y_val",
        default=None,
    )

    p.add_argument(
        "--edge_index",
        default=None,
    )

    p.add_argument(
        "--edge_weight",
        default=None,
    )

    p.add_argument(
        "--node_coords",
        default=None,
    )

    p.add_argument(
        "--node_names",
        default=None,
    )

    p.add_argument(
        "--hidden",
        type=int,
        default=64,
    )

    p.add_argument(
        "--heads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--gat_dropout",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--max_epochs",
        type=int,
        default=50,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    p.add_argument(
        "--lstm_mae",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--out_dir",
        default="day30_results",
    )

    args = p.parse_args()

    main(args)
