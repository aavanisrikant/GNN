import torch
import torch.nn as nn

from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# 1. HELPER — BUILD NORMALISED SPARSE ADJACENCY
# ─────────────────────────────────────────────────────────────────────────────

def build_normalised_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
) -> torch.sparse.Tensor:

    """
    Constructs symmetrically normalised adjacency:

        A_norm = D^{-1/2} (A + I) D^{-1/2}

    Returns:
        sparse adjacency tensor of shape (N, N)
    """

    # ────────────────────────────────────────────────────────────────
    # STEP 1 — ADD SELF LOOPS
    # ────────────────────────────────────────────────────────────────
    edge_index_sl, edge_weight_sl = add_self_loops(
        edge_index=edge_index,
        edge_attr=edge_weight,
        fill_value=1.0,
        num_nodes=n_nodes,
    )

    row, col = edge_index_sl

    # ────────────────────────────────────────────────────────────────
    # STEP 2 — WEIGHTED DEGREE
    # IMPORTANT FIX:
    # degree() ignores edge weights
    # so we compute weighted degree manually
    # ────────────────────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────────
    # STEP 3 — D^{-1/2}
    # ────────────────────────────────────────────────────────────────
    deg_inv_sqrt = deg.pow(-0.5)

    deg_inv_sqrt[
        deg_inv_sqrt == float("inf")
    ] = 0.0

    # ────────────────────────────────────────────────────────────────
    # STEP 4 — NORMALISATION
    # norm[e] = w_ij / sqrt(d_i * d_j)
    # ────────────────────────────────────────────────────────────────
    norm = (
        edge_weight_sl
        * deg_inv_sqrt[row]
        * deg_inv_sqrt[col]
    )

    # ────────────────────────────────────────────────────────────────
    # STEP 5 — BUILD SPARSE ADJACENCY
    # ────────────────────────────────────────────────────────────────
    A_norm = torch.sparse_coo_tensor(
        indices=edge_index_sl,
        values=norm,
        size=(n_nodes, n_nodes),
    ).coalesce()

    return A_norm


# ─────────────────────────────────────────────────────────────────────────────
# 2. TGCN GRAPH CONVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

class TGCNGraphConv(nn.Module):

    """
    Graph convolution module for T-GCN.

    Formula:

        H_out = A_norm X W + b

    where:
        A_norm = normalised adjacency
        X      = node features
        W      = learnable weight matrix
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_index: torch.Tensor,
        n_nodes: int,
        edge_weight: torch.Tensor = None,
        bias: bool = True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_nodes = n_nodes

        # ────────────────────────────────────────────────────────────
        # DEFAULT EDGE WEIGHTS
        # ────────────────────────────────────────────────────────────
        if edge_weight is None:

            edge_weight = torch.ones(
                edge_index.size(1),
                dtype=torch.float32,
            )

        # ────────────────────────────────────────────────────────────
        # PRECOMPUTE NORMALISED ADJACENCY
        # ────────────────────────────────────────────────────────────
        A_norm = build_normalised_adjacency(
            edge_index=edge_index,
            edge_weight=edge_weight,
            n_nodes=n_nodes,
        )

        # Stored as buffer (not trainable)
        self.register_buffer(
            "A_norm",
            A_norm
        )

        # ────────────────────────────────────────────────────────────
        # LEARNABLE WEIGHT MATRIX
        # ────────────────────────────────────────────────────────────
        self.W = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels
            )
        )

        # ────────────────────────────────────────────────────────────
        # OPTIONAL BIAS
        # ────────────────────────────────────────────────────────────
        if bias:

            self.b = nn.Parameter(
                torch.zeros(out_channels)
            )

        else:

            self.register_parameter(
                "b",
                None
            )

        self._init_weights()

    # ────────────────────────────────────────────────────────────────
    # INITIALISATION
    # ────────────────────────────────────────────────────────────────
    def _init_weights(self):

        nn.init.xavier_uniform_(self.W)

        if self.b is not None:

            nn.init.zeros_(self.b)

    # ────────────────────────────────────────────────────────────────
    # STEP 1 — NORMALISE ADJACENCY
    # Already precomputed during __init__
    # ────────────────────────────────────────────────────────────────
    def normalise_adjacency(self):

        return self.A_norm

    # ────────────────────────────────────────────────────────────────
    # STEP 2 — AGGREGATE NEIGHBOUR FEATURES
    # ────────────────────────────────────────────────────────────────
    def aggregate(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """
        Aggregation:

            A_norm @ X
        """

        return torch.sparse.mm(
            self.A_norm,
            x
        )

    # ────────────────────────────────────────────────────────────────
    # STEP 3 — LINEAR TRANSFORMATION
    # ────────────────────────────────────────────────────────────────
    def linear_transform(
        self,
        aggr: torch.Tensor
    ) -> torch.Tensor:

        """
        Linear transform:

            aggr @ W + b
        """

        out = aggr @ self.W

        if self.b is not None:

            out = out + self.b

        return out

    # ────────────────────────────────────────────────────────────────
    # FORWARD
    # ────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # Step 1
        _ = self.normalise_adjacency()

        # Step 2
        aggr = self.aggregate(x)

        # Step 3
        out = self.linear_transform(aggr)

        return out

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"in={self.in_channels}, "
            f"out={self.out_channels}, "
            f"n_nodes={self.n_nodes})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOY GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_toy_graph(
    in_channels: int = 8,
    n_nodes: int = 5,
) -> Data:

    """
    Toy graph:

        0 -- 1 -- 2
        |         |
        3 ------- 4
    """

    torch.manual_seed(42)

    x = torch.rand(
        n_nodes,
        in_channels
    )

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

    y = torch.rand(
        n_nodes,
        1
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y,
        num_nodes=n_nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAPE ASSERTION
# ─────────────────────────────────────────────────────────────────────────────

def assert_shape(
    tensor: torch.Tensor,
    expected: tuple,
    label: str,
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
# 5. VERIFY SHAPES
# ─────────────────────────────────────────────────────────────────────────────

def verify_shapes():

    print("=" * 65)
    print("TGCNGraphConv SHAPE VERIFICATION")
    print("=" * 65)

    # ────────────────────────────────────────────────────────────
    # CONFIG
    # ────────────────────────────────────────────────────────────
    N = 5
    F_IN = 8
    HIDDEN = 64
    F_OUT = 1
    T = 24

    # ────────────────────────────────────────────────────────────
    # TOY GRAPH
    # ────────────────────────────────────────────────────────────
    data = build_toy_graph(
        in_channels=F_IN,
        n_nodes=N,
    )

    print(
        f"\nToy graph: "
        f"{N} nodes, "
        f"{data.num_edges} directed edges"
    )

    # ────────────────────────────────────────────────────────────
    # INSPECT NORMALISED ADJACENCY
    # ────────────────────────────────────────────────────────────
    conv = TGCNGraphConv(
        in_channels=F_IN,
        out_channels=HIDDEN,
        edge_index=data.edge_index,
        n_nodes=N,
        edge_weight=data.edge_attr,
    )

    A = conv.normalise_adjacency()

    A_dense = A.to_dense()

    print("\nNormalised adjacency:")

    print(
        f"  Shape       : "
        f"{tuple(A.shape)}"
    )

    print(
        f"  Non-zeros   : "
        f"{A._nnz()}"
    )

    print(
        f"  Dense matrix:\n"
        f"{A_dense.numpy().round(3)}"
    )

    # ────────────────────────────────────────────────────────────
    # AGGREGATION TEST
    # ────────────────────────────────────────────────────────────
    print("\nAggregation step:")

    aggr = conv.aggregate(data.x)

    print(
        f"  x shape     : "
        f"{tuple(data.x.shape)}"
    )

    print(
        f"  aggr shape  : "
        f"{tuple(aggr.shape)}"
    )

    assert_shape(
        aggr,
        expected=(N, F_IN),
        label="aggregate output"
    )

    # ────────────────────────────────────────────────────────────
    # LINEAR TRANSFORM TEST
    # ────────────────────────────────────────────────────────────
    print("\nLinear transform step:")

    out_linear = conv.linear_transform(
        aggr
    )

    print(
        f"  aggr shape  : "
        f"{tuple(aggr.shape)}"
    )

    print(
        f"  W shape     : "
        f"{tuple(conv.W.shape)}"
    )

    print(
        f"  out shape   : "
        f"{tuple(out_linear.shape)}"
    )

    assert_shape(
        out_linear,
        expected=(N, HIDDEN),
        label="linear_transform output"
    )

    # ────────────────────────────────────────────────────────────
    # FULL FORWARD PASS
    # ────────────────────────────────────────────────────────────
    print("\nFull forward pass:")

    with torch.no_grad():

        out_hidden = conv(data.x)

    assert_shape(
        out_hidden,
        expected=(N, HIDDEN),
        label="forward output"
    )

    # ────────────────────────────────────────────────────────────
    # SECOND LAYER
    # ────────────────────────────────────────────────────────────
    conv2 = TGCNGraphConv(
        in_channels=HIDDEN,
        out_channels=F_OUT,
        edge_index=data.edge_index,
        n_nodes=N,
        edge_weight=data.edge_attr,
    )

    with torch.no_grad():

        out_final = conv2(out_hidden)

    print("\nSecond layer output:")

    assert_shape(
        out_final,
        expected=(N, F_OUT),
        label="hidden → output"
    )

    # ────────────────────────────────────────────────────────────
    # NO EDGE WEIGHTS
    # ────────────────────────────────────────────────────────────
    print("\nNo edge weights:")

    conv_no_ew = TGCNGraphConv(
        in_channels=F_IN,
        out_channels=HIDDEN,
        edge_index=data.edge_index,
        n_nodes=N,
        edge_weight=None,
    )

    with torch.no_grad():

        out_no_ew = conv_no_ew(data.x)

    assert_shape(
        out_no_ew,
        expected=(N, HIDDEN),
        label="uniform edge weights"
    )

    # ────────────────────────────────────────────────────────────
    # BACKPROP TEST
    # ────────────────────────────────────────────────────────────
    print("\nBackpropagation test:")

    conv_train = TGCNGraphConv(
        in_channels=F_IN,
        out_channels=F_OUT,
        edge_index=data.edge_index,
        n_nodes=N,
        edge_weight=data.edge_attr,
    )

    optimizer = torch.optim.Adam(
        conv_train.parameters(),
        lr=1e-3
    )

    criterion = nn.MSELoss()

    optimizer.zero_grad()

    pred = conv_train(data.x)

    loss = criterion(
        pred,
        data.y
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

    print(
        f"  W.grad norm      : "
        f"{conv_train.W.grad.norm().item():.6f}"
    )

    print("  [PASS ✓] Backpropagation")

    # ────────────────────────────────────────────────────────────
    # T-GCN SEQUENCE TEST
    # ────────────────────────────────────────────────────────────
    print(f"\nT-GCN sequence test (T={T})")

    conv_seq = TGCNGraphConv(
        in_channels=F_IN,
        out_channels=HIDDEN,
        edge_index=data.edge_index,
        n_nodes=N,
        edge_weight=data.edge_attr,
    )

    x_sequence = torch.rand(
        T,
        N,
        F_IN
    )

    outputs = []

    with torch.no_grad():

        for t in range(T):

            x_t = x_sequence[t]

            h_t = conv_seq(x_t)

            outputs.append(h_t)

    gcn_sequence = torch.stack(
        outputs,
        dim=0
    )

    print(
        f"  Sequence shape : "
        f"{tuple(gcn_sequence.shape)}"
    )

    assert_shape(
        gcn_sequence,
        expected=(T, N, HIDDEN),
        label="GCN sequence"
    )

    # ────────────────────────────────────────────────────────────
    # PARAMETER COUNT
    # ────────────────────────────────────────────────────────────
    total_params = sum(
        p.numel()
        for p in conv.parameters()
    )

    print("\nParameter count:")

    print(
        f"  Total params : "
        f"{total_params:,}"
    )

    # ────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)

    print(
        "ALL CHECKS PASSED — "
        "TGCNGraphConv READY"
    )

    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    verify_shapes()
