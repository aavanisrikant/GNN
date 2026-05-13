import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────
# CUSTOM GCN LAYER
# ─────────────────────────────────────────────────────────────

class TGCNGraphConv(MessagePassing):

    """
    Custom GCN layer implemented from scratch using
    PyTorch Geometric MessagePassing.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True
    ):

        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(
            in_channels,
            out_channels,
            bias=False
        )

        if bias:

            self.bias = nn.Parameter(
                torch.zeros(out_channels)
            )

        else:

            self.register_parameter(
                "bias",
                None
            )

        self.reset_parameters()

    # ─────────────────────────────────────────────────────────
    # PARAMETER INITIALIZATION
    # ─────────────────────────────────────────────────────────
    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self.linear.weight
        )

        if self.bias is not None:

            nn.init.zeros_(
                self.bias
            )

    # ─────────────────────────────────────────────────────────
    # FORWARD
    # ─────────────────────────────────────────────────────────
    def forward(
        self,
        x,
        edge_index,
        edge_weight=None
    ):

        n_nodes = x.size(0)

        # Add self loops
        edge_index, edge_weight = add_self_loops(
            edge_index=edge_index,
            edge_attr=edge_weight,
            fill_value=1.0,
            num_nodes=n_nodes
        )

        # Default edge weights
        if edge_weight is None:

            edge_weight = torch.ones(
                edge_index.size(1),
                dtype=x.dtype,
                device=x.device
            )

        row, col = edge_index

        # Weighted degree
        deg = torch.zeros(
            n_nodes,
            dtype=x.dtype,
            device=x.device
        )

        deg.scatter_add_(
            0,
            col,
            edge_weight
        )

        # D^(-1/2)
        deg_inv_sqrt = deg.pow(-0.5)

        deg_inv_sqrt[
            deg_inv_sqrt == float("inf")
        ] = 0.0

        # Normalization
        norm = (
            edge_weight
            * deg_inv_sqrt[row]
            * deg_inv_sqrt[col]
        )

        # Linear transform
        x = self.linear(x)

        # Message passing
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            norm=norm
        )

        # Bias
        if self.bias is not None:

            out = out + self.bias

        return out

    # ─────────────────────────────────────────────────────────
    # MESSAGE
    # ─────────────────────────────────────────────────────────
    def message(
        self,
        x_j,
        norm
    ):

        return norm.unsqueeze(-1) * x_j

    # ─────────────────────────────────────────────────────────
    # UPDATE
    # ─────────────────────────────────────────────────────────
    def update(
        self,
        aggr_out
    ):

        return aggr_out

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(in={self.in_channels}, "
            f"out={self.out_channels})"
        )


# ─────────────────────────────────────────────────────────────
# TWO-LAYER GCN MODEL
# ─────────────────────────────────────────────────────────────

class GCN(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels,
        dropout=0.2
    ):

        super().__init__()

        self.conv1 = TGCNGraphConv(
            in_channels,
            hidden_dim
        )

        self.conv2 = TGCNGraphConv(
            hidden_dim,
            out_channels
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None
    ):

        # First layer
        x = self.conv1(
            x,
            edge_index,
            edge_weight
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Second layer
        x = self.conv2(
            x,
            edge_index,
            edge_weight
        )

        return x


# ─────────────────────────────────────────────────────────────
# TOY GRAPH
# ─────────────────────────────────────────────────────────────

def build_toy_graph(
    in_channels=8
):

    """
    Toy graph topology:

        0 -- 1 -- 2
        |         |
        3 ------- 4
    """

    torch.manual_seed(42)

    # Node features
    x = torch.rand(
        5,
        in_channels
    )

    # Undirected edges
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 3, 2, 4, 3, 4],
            [1, 0, 2, 1, 3, 0, 4, 2, 4, 3],
        ],
        dtype=torch.long
    )

    # Example edge weights
    edge_weight = torch.tensor(
        [
            0.5, 0.5,
            0.3, 0.3,
            0.8, 0.8,
            0.4, 0.4,
            0.6, 0.6,
        ],
        dtype=torch.float
    )

    # Dummy targets
    y = torch.rand(5, 1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y
    )

    return data


# ─────────────────────────────────────────────────────────────
# SHAPE CHECKER
# ─────────────────────────────────────────────────────────────

def assert_shape(
    tensor,
    expected,
    label
):

    actual = tuple(tensor.shape)

    if actual == expected:

        print(
            f"[PASS ✓] {label}: {actual}"
        )

    else:

        print(
            f"[FAIL ✗] {label}: "
            f"got {actual}, "
            f"expected {expected}"
        )

        raise AssertionError(
            f"{label} shape mismatch"
        )


# ─────────────────────────────────────────────────────────────
# PRINT ADJACENCY
# ─────────────────────────────────────────────────────────────

def print_adjacency(data):

    n = data.num_nodes

    A = torch.zeros(n, n)

    src, dst = data.edge_index

    for s, d, w in zip(
        src,
        dst,
        data.edge_attr
    ):

        A[s, d] = w.item()

    print("\nWeighted Adjacency Matrix:\n")

    print(
        "      "
        + " ".join(
            [f"{i:>6}" for i in range(n)]
        )
    )

    for i in range(n):

        row = " ".join(
            [f"{A[i, j]:6.2f}" for j in range(n)]
        )

        print(f"{i:>3} | {row}")

    print("\nTopology:")
    print("  0 -- 1 -- 2")
    print("  |         |")
    print("  3 ------- 4")


# ─────────────────────────────────────────────────────────────
# VERIFY SHAPES
# ─────────────────────────────────────────────────────────────

def verify_shapes():

    print("=" * 60)
    print("GCN SHAPE VERIFICATION")
    print("=" * 60)

    # Config
    IN_CHANNELS = 8
    HIDDEN_DIM = 64
    OUT_CHANNELS = 1
    N_NODES = 5

    # Build graph
    data = build_toy_graph(
        in_channels=IN_CHANNELS
    )

    print("\nToy Graph:")
    print(f"  Nodes       : {data.num_nodes}")
    print(f"  Edges       : {data.num_edges}")
    print(f"  x shape     : {data.x.shape}")
    print(f"  edge_index  : {data.edge_index.shape}")
    print(f"  edge_weight : {data.edge_attr.shape}")
    print(f"  target y    : {data.y.shape}")

    # ─────────────────────────────────────────────────────────
    # SINGLE GCN LAYER
    # ─────────────────────────────────────────────────────────

    single_conv = TGCNGraphConv(
        IN_CHANNELS,
        HIDDEN_DIM
    )

    single_conv.eval()

    with torch.no_grad():

        out_single = single_conv(
            data.x,
            data.edge_index,
            data.edge_attr
        )

    print("\nSingle GCN Layer:")

    print(
        f"  Output shape: "
        f"{out_single.shape}"
    )

    assert_shape(
        out_single,
        expected=(N_NODES, HIDDEN_DIM),
        label="Single GCN output"
    )

    # ─────────────────────────────────────────────────────────
    # FULL MODEL
    # ─────────────────────────────────────────────────────────

    model = GCN(
        in_channels=IN_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        out_channels=OUT_CHANNELS,
        dropout=0.2
    )

    model.eval()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nFull GCN:")
    print(f"  {model.conv1}")
    print(f"  {model.conv2}")
    print(f"  Parameters: {total_params:,}")

    with torch.no_grad():

        out_full = model(
            data.x,
            data.edge_index,
            data.edge_attr
        )

    print("\nFull Model Output:")

    print(
        f"  Output shape: "
        f"{out_full.shape}"
    )

    assert_shape(
        out_full,
        expected=(N_NODES, OUT_CHANNELS),
        label="Full GCN output"
    )

    # ─────────────────────────────────────────────────────────
    # TEST WITHOUT EDGE WEIGHTS
    # ─────────────────────────────────────────────────────────

    with torch.no_grad():

        out_no_weight = model(
            data.x,
            data.edge_index,
            edge_weight=None
        )

    print("\nWithout Edge Weights:")

    print(
        f"  Output shape: "
        f"{out_no_weight.shape}"
    )

    assert_shape(
        out_no_weight,
        expected=(N_NODES, OUT_CHANNELS),
        label="No-edge-weight output"
    )

    # ─────────────────────────────────────────────────────────
    # LOSS TEST
    # ─────────────────────────────────────────────────────────

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    optimizer.zero_grad()

    pred = model(
        data.x,
        data.edge_index,
        data.edge_attr
    )

    loss = criterion(
        pred,
        data.y
    )

    loss.backward()

    optimizer.step()

    print("\nLoss Computation:")

    print(
        f"  Prediction shape: "
        f"{pred.shape}"
    )

    print(
        f"  Target shape    : "
        f"{data.y.shape}"
    )

    print(
        f"  Loss            : "
        f"{loss.item():.6f}"
    )

    print(
        "  Backpropagation : PASS ✓"
    )

    # ─────────────────────────────────────────────────────────
    # T-GCN PREPARATION
    # ─────────────────────────────────────────────────────────

    T = 24

    print(
        f"\nT-GCN sequence test "
        f"(T={T})"
    )

    gcn_outputs = []

    model.eval()

    with torch.no_grad():

        for t in range(T):

            x_t = torch.rand(
                N_NODES,
                IN_CHANNELS
            )

            h_t = model.conv1(
                x_t,
                data.edge_index,
                data.edge_attr
            )

            h_t = F.relu(h_t)

            gcn_outputs.append(h_t)

    gcn_sequence = torch.stack(
        gcn_outputs,
        dim=0
    )

    print(
        f"  Sequence shape: "
        f"{gcn_sequence.shape}"
    )

    assert_shape(
        gcn_sequence,
        expected=(T, N_NODES, HIDDEN_DIM),
        label="GCN sequence"
    )

    print("\n" + "=" * 60)

    print("ALL CHECKS PASSED ✓")

    print(
        "GCN READY FOR T-GCN INTEGRATION"
    )

    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    toy = build_toy_graph()

    print_adjacency(toy)

    print()

    verify_shapes()
