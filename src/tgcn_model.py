import torch
import torch.nn as nn

from tgcn_cell import TGCNCell


class TGCN(nn.Module):

    """
    Full Temporal Graph Convolutional Network.

    Input:
        (B, T, N, F)

    Output:
        (B, N, output_steps)

    Pipeline:

        Sequence
            ↓
        TGCNCell recurrence
            ↓
        Final hidden state
            ↓
        Linear decoder
            ↓
        AQI prediction
    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        output_steps,
        edge_index,
        n_nodes,
        edge_weight=None,
        num_layers=2,
        dropout=0.2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_steps = output_steps
        self.n_nodes = n_nodes
        self.num_layers = num_layers

        # ────────────────────────────────────────────────────────────
        # STACKED T-GCN CELLS
        # ────────────────────────────────────────────────────────────
        cells = []

        for i in range(num_layers):

            layer_input_dim = (
                in_channels
                if i == 0
                else hidden_dim
            )

            cell = TGCNCell(
                in_channels=layer_input_dim,
                hidden_dim=hidden_dim,
                edge_index=edge_index,
                n_nodes=n_nodes,
                edge_weight=edge_weight,
            )

            cells.append(cell)

        self.cells = nn.ModuleList(cells)

        # ────────────────────────────────────────────────────────────
        # OUTPUT DECODER
        # ────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(
            p=dropout
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            output_steps,
        )

    # ────────────────────────────────────────────────────────────────
    # FORWARD
    # ────────────────────────────────────────────────────────────────
    def forward(self, x):

        """
        Args:
            x:
                Input sequence
                Shape: (B, T, N, F)

        Returns:
            predictions:
                Shape: (B, N, output_steps)
        """

        B, T, N, F_in = x.shape

        # Safety checks
        assert (
            N == self.n_nodes
        ), (
            f"Expected {self.n_nodes} nodes, "
            f"got {N}"
        )

        assert (
            F_in == self.in_channels
        ), (
            f"Expected {self.in_channels} input features, "
            f"got {F_in}"
        )

        batch_outputs = []

        # ────────────────────────────────────────────────────────────
        # PROCESS EACH BATCH ITEM
        # ────────────────────────────────────────────────────────────
        #
        # We process one graph sequence at a time because
        # adjacency matrix shape is fixed at (N, N).
        #
        # x_b:
        #     (T, N, F)
        #
        # h[layer]:
        #     (N, hidden_dim)
        #
        # ────────────────────────────────────────────────────────────

        for b in range(B):

            x_b = x[b]

            # Hidden state for each layer
            hidden_states = [

                torch.zeros(
                    N,
                    self.hidden_dim,
                    device=x.device,
                    dtype=x.dtype,
                )

                for _ in range(self.num_layers)
            ]

            # ────────────────────────────────────────────────────────
            # TEMPORAL UNROLLING
            # ────────────────────────────────────────────────────────
            for t in range(T):

                x_t = x_b[t]

                # Pass through stacked T-GCN layers
                for layer_idx, cell in enumerate(self.cells):

                    h_prev = hidden_states[layer_idx]

                    h_new = cell(
                        x_t=x_t,
                        h_prev=h_prev,
                    )

                    hidden_states[layer_idx] = h_new

                    # Output becomes input to next layer
                    x_t = h_new

            # ────────────────────────────────────────────────────────
            # FINAL HIDDEN STATE
            # ────────────────────────────────────────────────────────
            #
            # Shape:
            #   (N, hidden_dim)
            #
            # Sequence-to-one forecasting:
            #
            #   Input sequence
            #   → final hidden state
            #   → next-step prediction
            #
            # ────────────────────────────────────────────────────────
            final_hidden = hidden_states[-1]

            # ────────────────────────────────────────────────────────
            # OUTPUT HEAD
            # ────────────────────────────────────────────────────────
            out = self.dropout(
                final_hidden
            )

            out = self.output_layer(
                out
            )

            # Shape:
            #   (N, output_steps)

            batch_outputs.append(out)

        # Stack batch dimension
        predictions = torch.stack(
            batch_outputs,
            dim=0
        )

        # Final:
        #   (B, N, output_steps)

        return predictions

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"in={self.in_channels}, "
            f"hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, "
            f"output_steps={self.output_steps})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOY GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_toy_graph(n_nodes=15):

    """
    Random undirected graph for shape testing.
    """

    import random

    src = []
    dst = []
    wts = []

    for i in range(n_nodes):

        for j in range(i + 1, n_nodes):

            if random.random() < 0.4:

                w = random.uniform(
                    0.1,
                    1.0
                )

                src.extend([i, j])
                dst.extend([j, i])
                wts.extend([w, w])

    edge_index = torch.tensor(
        [src, dst],
        dtype=torch.long,
    )

    edge_weight = torch.tensor(
        wts,
        dtype=torch.float32,
    )

    return edge_index, edge_weight


# ─────────────────────────────────────────────────────────────────────────────
# SHAPE ASSERTION
# ─────────────────────────────────────────────────────────────────────────────

def assert_shape(
    tensor,
    expected,
    label,
):

    actual = tuple(tensor.shape)

    ok = actual == expected

    print(
        f"  [{'PASS ✓' if ok else 'FAIL ✗'}] "
        f"{label}: "
        f"got {actual} "
        f"expected {expected}"
    )

    assert ok, (
        f"{label} shape mismatch: "
        f"expected {expected}, "
        f"got {actual}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY MODEL
# ─────────────────────────────────────────────────────────────────────────────

def verify_tgcn_model():

    print("=" * 72)
    print("FULL T-GCN MODEL VERIFICATION")
    print("=" * 72)

    # ────────────────────────────────────────────────────────────
    # CONFIG
    # ────────────────────────────────────────────────────────────
    BATCH_SIZE = 1
    T = 24
    N = 15
    F_IN = 8
    HIDDEN = 64
    OUT_STEPS = 1
    NUM_LAYERS = 2

    torch.manual_seed(42)

    # ────────────────────────────────────────────────────────────
    # GRAPH
    # ────────────────────────────────────────────────────────────
    edge_index, edge_weight = build_toy_graph(
        n_nodes=N
    )

    print("\nGraph:")

    print(
        f"  Nodes         : {N}"
    )

    print(
        f"  Directed edges: "
        f"{edge_index.size(1)}"
    )

    # ────────────────────────────────────────────────────────────
    # MODEL
    # ────────────────────────────────────────────────────────────
    model = TGCN(
        in_channels=F_IN,
        hidden_dim=HIDDEN,
        output_steps=OUT_STEPS,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
        num_layers=NUM_LAYERS,
        dropout=0.2,
    )

    print("\nModel:")

    print(model)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nTotal parameters: "
        f"{total_params:,}"
    )

    # ────────────────────────────────────────────────────────────
    # FORWARD PASS TEST
    # ────────────────────────────────────────────────────────────
    print("\nForward pass test:")

    print(
        f"  Input shape : "
        f"(B={BATCH_SIZE}, "
        f"T={T}, "
        f"N={N}, "
        f"F={F_IN})"
    )

    x = torch.rand(
        BATCH_SIZE,
        T,
        N,
        F_IN,
    )

    model.eval()

    with torch.no_grad():

        preds = model(x)

    assert_shape(
        preds,
        expected=(
            BATCH_SIZE,
            N,
            OUT_STEPS,
        ),
        label="output shape",
    )

    # ────────────────────────────────────────────────────────────
    # MULTI-STEP FORECAST TEST
    # ────────────────────────────────────────────────────────────
    print("\nMulti-step forecast test:")

    model_multi = TGCN(
        in_channels=F_IN,
        hidden_dim=HIDDEN,
        output_steps=3,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
        num_layers=NUM_LAYERS,
    )

    with torch.no_grad():

        preds_multi = model_multi(x)

    assert_shape(
        preds_multi,
        expected=(
            BATCH_SIZE,
            N,
            3,
        ),
        label="3-step output",
    )

    # ────────────────────────────────────────────────────────────
    # BATCH TEST
    # ────────────────────────────────────────────────────────────
    print("\nBatch test (B=4):")

    x_batch = torch.rand(
        4,
        T,
        N,
        F_IN,
    )

    with torch.no_grad():

        preds_batch = model(x_batch)

    assert_shape(
        preds_batch,
        expected=(
            4,
            N,
            OUT_STEPS,
        ),
        label="batched output",
    )

    # ────────────────────────────────────────────────────────────
    # BACKPROP TEST
    # ────────────────────────────────────────────────────────────
    print("\nBackpropagation test:")

    model_train = TGCN(
        in_channels=F_IN,
        hidden_dim=HIDDEN,
        output_steps=OUT_STEPS,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
        num_layers=NUM_LAYERS,
    )

    model_train.train()

    optimizer = torch.optim.Adam(
        model_train.parameters(),
        lr=1e-3,
    )

    criterion = nn.MSELoss()

    target = torch.rand(
        BATCH_SIZE,
        N,
        OUT_STEPS,
    )

    optimizer.zero_grad()

    output = model_train(x)

    loss = criterion(
        output,
        target,
    )

    loss.backward()

    optimizer.step()

    print(
        f"  Loss : "
        f"{loss.item():.6f}"
    )

    grad_norm = (
        model_train
        .cells[0]
        .gconv_z
        .W
        .grad
        .norm()
        .item()
    )

    print(
        f"  Gradient norm : "
        f"{grad_norm:.6f}"
    )

    print(
        "  [PASS ✓] Backpropagation"
    )

    # ────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)

    print(
        "ALL CHECKS PASSED — "
        "FULL T-GCN MODEL READY"
    )

    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    verify_tgcn_model()
