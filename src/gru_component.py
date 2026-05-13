import torch
import torch.nn as nn

from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD NORMALISED ADJACENCY
# ─────────────────────────────────────────────────────────────────────────────

def build_normalised_adjacency(
    edge_index,
    edge_weight,
    n_nodes,
):

    """
    Computes:

        A_norm = D^{-1/2}(A+I)D^{-1/2}
    """

    edge_index_sl, edge_weight_sl = add_self_loops(
        edge_index=edge_index,
        edge_attr=edge_weight,
        fill_value=1.0,
        num_nodes=n_nodes,
    )

    row, col = edge_index_sl

    # Weighted degree
    deg = torch.zeros(
        n_nodes,
        dtype=edge_weight_sl.dtype,
        device=edge_weight_sl.device,
    )

    deg.scatter_add_(
        0,
        col,
        edge_weight_sl
    )

    deg_inv_sqrt = deg.pow(-0.5)

    deg_inv_sqrt[
        deg_inv_sqrt == float("inf")
    ] = 0.0

    norm = (
        edge_weight_sl
        * deg_inv_sqrt[row]
        * deg_inv_sqrt[col]
    )

    A_norm = torch.sparse_coo_tensor(
        edge_index_sl,
        norm,
        (n_nodes, n_nodes),
    ).coalesce()

    return A_norm


# ─────────────────────────────────────────────────────────────────────────────
# 2. TGCN GRAPH CONVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

class TGCNGraphConv(nn.Module):

    """
    Graph convolution:

        H = A_norm X W + b
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_index,
        n_nodes,
        edge_weight=None,
        bias=True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Default edge weights
        if edge_weight is None:

            edge_weight = torch.ones(
                edge_index.size(1),
                dtype=torch.float32,
            )

        # Precompute adjacency
        A_norm = build_normalised_adjacency(
            edge_index=edge_index,
            edge_weight=edge_weight,
            n_nodes=n_nodes,
        )

        self.register_buffer(
            "A_norm",
            A_norm
        )

        # Learnable weights
        self.W = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels
            )
        )

        if bias:

            self.b = nn.Parameter(
                torch.zeros(out_channels)
            )

        else:

            self.register_parameter(
                "b",
                None
            )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.W)

        if self.b is not None:

            nn.init.zeros_(self.b)

    def forward(self, x):

        # Aggregate neighbors
        aggr = torch.sparse.mm(
            self.A_norm,
            x
        )

        # Linear transform
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
# 3. T-GCN CELL
# ─────────────────────────────────────────────────────────────────────────────

class TGCNCell(nn.Module):

    """
    Temporal Graph Convolutional GRU Cell.

    T-GCN replaces standard GRU linear transforms
    with graph convolutions.

    Standard GRU:

        z_t = σ(W_z [x_t, h_{t-1}] + b_z)

    T-GCN:

        z_t = σ(GCN([x_t, h_{t-1}]))

    Same for:
        reset gate
        candidate hidden state
    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        edge_index,
        n_nodes,
        edge_weight=None,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes

        # Combined input:
        #
        # [x_t , h_{t-1}]
        #
        combined_dim = (
            in_channels
            + hidden_dim
        )

        # ────────────────────────────────────────────────────────────
        # UPDATE GATE
        # z_t
        # ────────────────────────────────────────────────────────────
        self.gconv_z = TGCNGraphConv(
            in_channels=combined_dim,
            out_channels=hidden_dim,
            edge_index=edge_index,
            n_nodes=n_nodes,
            edge_weight=edge_weight,
        )

        # ────────────────────────────────────────────────────────────
        # RESET GATE
        # r_t
        # ────────────────────────────────────────────────────────────
        self.gconv_r = TGCNGraphConv(
            in_channels=combined_dim,
            out_channels=hidden_dim,
            edge_index=edge_index,
            n_nodes=n_nodes,
            edge_weight=edge_weight,
        )

        # ────────────────────────────────────────────────────────────
        # CANDIDATE HIDDEN STATE
        # h~_t
        # ────────────────────────────────────────────────────────────
        self.gconv_h = TGCNGraphConv(
            in_channels=combined_dim,
            out_channels=hidden_dim,
            edge_index=edge_index,
            n_nodes=n_nodes,
            edge_weight=edge_weight,
        )

    # ────────────────────────────────────────────────────────────────
    # FORWARD
    # ────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_t,
        h_prev,
    ):

        """
        Args:
            x_t:
                Current node features
                Shape: (N, in_channels)

            h_prev:
                Previous hidden state
                Shape: (N, hidden_dim)

        Returns:
            h_t:
                Updated hidden state
                Shape: (N, hidden_dim)
        """

        # ────────────────────────────────────────────────────────────
        # CONCATENATE INPUT + PREVIOUS HIDDEN
        # ────────────────────────────────────────────────────────────
        combined = torch.cat(
            [x_t, h_prev],
            dim=-1
        )

        # ────────────────────────────────────────────────────────────
        # UPDATE GATE
        # Eq. 4
        # ────────────────────────────────────────────────────────────
        z_t = torch.sigmoid(
            self.gconv_z(combined)
        )

        # ────────────────────────────────────────────────────────────
        # RESET GATE
        # Eq. 5
        # ────────────────────────────────────────────────────────────
        r_t = torch.sigmoid(
            self.gconv_r(combined)
        )

        # ────────────────────────────────────────────────────────────
        # RESETTED HIDDEN STATE
        # ────────────────────────────────────────────────────────────
        reset_hidden = (
            r_t * h_prev
        )

        # ────────────────────────────────────────────────────────────
        # CANDIDATE STATE INPUT
        # ────────────────────────────────────────────────────────────
        candidate_input = torch.cat(
            [x_t, reset_hidden],
            dim=-1
        )

        # ────────────────────────────────────────────────────────────
        # CANDIDATE HIDDEN STATE
        # Eq. 6
        # ────────────────────────────────────────────────────────────
        h_tilde = torch.tanh(
            self.gconv_h(candidate_input)
        )

        # ────────────────────────────────────────────────────────────
        # FINAL HIDDEN UPDATE
        # ────────────────────────────────────────────────────────────
        h_t = (
            z_t * h_prev
            + (1 - z_t) * h_tilde
        )

        return h_t

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"in={self.in_channels}, "
            f"hidden={self.hidden_dim})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOY GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_toy_graph():

    """
    Graph:

        0 -- 1 -- 2
        |         |
        3 ------- 4
    """

    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 3, 2, 4, 3, 4],
        [1, 0, 2, 1, 3, 0, 4, 2, 4, 3],
    ], dtype=torch.long)

    edge_weight = torch.tensor(
        [
            0.5, 0.5,
            0.3, 0.3,
            0.8, 0.8,
            0.4, 0.4,
            0.6, 0.6,
        ],
        dtype=torch.float32,
    )

    return edge_index, edge_weight


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAPE ASSERTION
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

    assert ok


# ─────────────────────────────────────────────────────────────────────────────
# 6. VERIFY T-GCN CELL
# ─────────────────────────────────────────────────────────────────────────────

def verify_tgcn():

    print("=" * 70)
    print("T-GCN CELL VERIFICATION")
    print("=" * 70)

    # Config
    N = 5
    F_IN = 8
    HIDDEN = 32
    T = 24

    # Graph
    edge_index, edge_weight = build_toy_graph()

    # T-GCN Cell
    tgcn = TGCNCell(
        in_channels=F_IN,
        hidden_dim=HIDDEN,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
    )

    print("\nTGCNCell:")
    print(tgcn)

    # ────────────────────────────────────────────────────────────
    # SINGLE TIMESTEP TEST
    # ────────────────────────────────────────────────────────────
    print("\nSingle timestep test:")

    x_t = torch.rand(
        N,
        F_IN
    )

    h_prev = torch.zeros(
        N,
        HIDDEN
    )

    with torch.no_grad():

        h_t = tgcn(
            x_t,
            h_prev
        )

    print(
        f"  x_t shape     : "
        f"{tuple(x_t.shape)}"
    )

    print(
        f"  h_prev shape  : "
        f"{tuple(h_prev.shape)}"
    )

    print(
        f"  h_t shape     : "
        f"{tuple(h_t.shape)}"
    )

    assert_shape(
        h_t,
        expected=(N, HIDDEN),
        label="single timestep output"
    )

    # ────────────────────────────────────────────────────────────
    # TEMPORAL SEQUENCE TEST
    # ────────────────────────────────────────────────────────────
    print(f"\nTemporal sequence test (T={T})")

    x_sequence = torch.rand(
        T,
        N,
        F_IN
    )

    h = torch.zeros(
        N,
        HIDDEN
    )

    outputs = []

    with torch.no_grad():

        for t in range(T):

            h = tgcn(
                x_sequence[t],
                h
            )

            outputs.append(h)

    outputs = torch.stack(
        outputs,
        dim=0
    )

    print(
        f"  Sequence shape: "
        f"{tuple(outputs.shape)}"
    )

    assert_shape(
        outputs,
        expected=(T, N, HIDDEN),
        label="temporal outputs"
    )

    # ────────────────────────────────────────────────────────────
    # BACKPROP TEST
    # ────────────────────────────────────────────────────────────
    print("\nBackpropagation test:")

    tgcn_train = TGCNCell(
        in_channels=F_IN,
        hidden_dim=HIDDEN,
        edge_index=edge_index,
        n_nodes=N,
        edge_weight=edge_weight,
    )

    optimizer = torch.optim.Adam(
        tgcn_train.parameters(),
        lr=1e-3
    )

    criterion = nn.MSELoss()

    target = torch.rand(
        N,
        HIDDEN
    )

    optimizer.zero_grad()

    pred = tgcn_train(
        x_t,
        h_prev
    )

    loss = criterion(
        pred,
        target
    )

    loss.backward()

    optimizer.step()

    print(
        f"  Prediction shape : "
        f"{tuple(pred.shape)}"
    )

    print(
        f"  Loss             : "
        f"{loss.item():.6f}"
    )

    # Gradient check
    grad_norm = (
        tgcn_train
        .gconv_z
        .W
        .grad
        .norm()
        .item()
    )

    print(
        f"  Gradient norm    : "
        f"{grad_norm:.6f}"
    )

    print("  [PASS ✓] Backpropagation")

    # ────────────────────────────────────────────────────────────
    # PARAMETER COUNT
    # ────────────────────────────────────────────────────────────
    total_params = sum(
        p.numel()
        for p in tgcn.parameters()
    )

    print("\nParameter count:")

    print(
        f"  Total params : "
        f"{total_params:,}"
    )

    print("\n" + "=" * 70)

    print("ALL CHECKS PASSED — T-GCN CELL READY")

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    verify_tgcn()
