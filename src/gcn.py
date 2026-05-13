import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# 1. CUSTOM GCN LAYER (FROM SCRATCH USING MessagePassing)
# ─────────────────────────────────────────────────────────────────────────────

class TGCNGraphConv(MessagePassing):
    """
    Custom Graph Convolution layer implemented from scratch using
    PyTorch Geometric's MessagePassing API.

    Implements the GCN update rule from Kipf & Welling (2017):

        H' = D^(-1/2) Â D^(-1/2) H W

    where:
        Â = A + I   (self-loops added)
        D = degree matrix of Â

    MessagePassing flow:
        message()   → compute message from neighbor j
        aggregate() → sum incoming messages
        update()    → final node update
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):

        # GCN uses SUM aggregation
        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transformation W
        self.linear = nn.Linear(
            in_channels,
            out_channels,
            bias=False
        )

        # Optional bias
        if bias:

            self.bias = nn.Parameter(
                torch.empty(out_channels)
            )

        else:

            self.register_parameter(
                "bias",
                None
            )

        self.reset_parameters()

    # ── INITIALIZATION ────────────────────────────────────────────────────
    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self.linear.weight
        )

        if self.bias is not None:

            nn.init.zeros_(
                self.bias
            )

    # ── FORWARD PASS ──────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:

        """
        Args:
            x:
                Node feature matrix
                Shape: (N, in_channels)

            edge_index:
                COO graph connectivity
                Shape: (2, E)

            edge_weight:
                Optional edge weights
                Shape: (E,)

        Returns:
            Updated node features
            Shape: (N, out_channels)
        """

        n_nodes = x.size(0)

        # ────────────────────────────────────────────────────────────────
        # STEP 1 — ADD SELF LOOPS
        # ────────────────────────────────────────────────────────────────
        edge_index, edge_weight = add_self_loops(
            edge_index=edge_index,
            edge_attr=edge_weight,
            fill_value=1.0,
            num_nodes=n_nodes,
        )

        # If no edge weights supplied,
        # assume all edges weight = 1
        if edge_weight is None:

            edge_weight = torch.ones(
                edge_index.size(1),
                dtype=x.dtype,
                device=x.device,
            )

        # ────────────────────────────────────────────────────────────────
        # STEP 2 — COMPUTE NORMALIZATION
        # ────────────────────────────────────────────────────────────────
        row, col = edge_index

        # Weighted node degree
        deg = torch.zeros(
            n_nodes,
            dtype=x.dtype,
            device=x.device,
        )

        deg.scatter_add_(
            0,
            col,
            edge_weight
        )

        # D^(-1/2)
        deg_inv_sqrt = deg.pow(-0.5)

        # Guard isolated nodes
        deg_inv_sqrt[
            deg_inv_sqrt == float("inf")
        ] = 0.0

        # Final normalization coefficient:
        #
        # edge_weight / sqrt(deg_i * deg_j)
        #
        norm = (
            edge_weight
            * deg_inv_sqrt[row]
            * deg_inv_sqrt[col]
        )

        # ────────────────────────────────────────────────────────────────
        # STEP 3 — LINEAR TRANSFORMATION
        # ────────────────────────────────────────────────────────────────
        x = self.linear(x)

        # ────────────────────────────────────────────────────────────────
        # STEP 4 — MESSAGE PASSING
        # ────────────────────────────────────────────────────────────────
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            norm=norm,
        )

        # ────────────────────────────────────────────────────────────────
        # STEP 5 — BIAS
        # ────────────────────────────────────────────────────────────────
        if self.bias is not None:

            out = out + self.bias

        return out

    # ── MESSAGE FUNCTION ─────────────────────────────────────────────────
    def message(
        self,
        x_j: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:

        """
        x_j:
            Neighbor node features
            Shape: (E, out_channels)

        norm:
            Edge normalization coefficients
            Shape: (E,)
        """

        return norm.unsqueeze(-1) * x_j

    # ── UPDATE FUNCTION ──────────────────────────────────────────────────
    def update(
        self,
        aggr_out: torch.Tensor
    ) -> torch.Tensor:

        """
        aggr_out:
            Aggregated messages
            Shape: (N, out_channels)
        """

        return aggr_out

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(in={self.in_channels}, "
            f"out={self.out_channels})"
        )

class GCN(nn.Module):

    """
    Simple 2-layer GCN baseline.

    Architecture:
        GCN → ReLU → Dropout → GCN
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        dropout: float = 0.2,
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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:

        # Layer 1
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

        # Layer 2
        x = self.conv2(
            x,
            edge_index,
            edge_weight
        )

        return x


def build_toy_graph(
    in_channels: int = 8
) -> Data:

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
    edge_index = torch.tensor([
        [0, 1,  1, 2,  0, 3,  2, 4,  3, 4],
        [1, 0,  2, 1,  3, 0,  4, 2,  4, 3],
    ], dtype=torch.long)

    # Simulated inverse-distance weights
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

    # AQI target
    y = torch.rand(5, 1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y
    )

def assert_shape(
    tensor: torch.Tensor,
    expected: tuple,
    label: str,
):

    actual = tuple(tensor.shape)

    if actual == expected:

        print(
            f"[PASS ✓] {label}: "
            f"{actual}"
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


def print_adjacency(
    data: Data
):

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

    print("      " + " ".join(
        [f"{i:>6}" for i in range(n)]
    ))

    for i in range(n):

        row = " ".join(
            [f"{A[i,j]:6.2f}" for j in range(n)]
        )

        print(f"{i:>3} | {row}")

    print("\nTopology:")
    print("  0 -- 1 -- 2")
    print("  |         |")
    print("  3 ------- 4")


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

       model = GCN(
        in_channels=IN_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        out_channels=OUT_CHANNELS,
        dropout=0.2,
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
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Target shape    : {data.y.shape}")
    print(f"  Loss            : {loss.item():.6f}")
    print("  Backpropagation : PASS ✓")

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
    print("GCN READY FOR T-GCN INTEGRATION")
    print("=" * 60)

if __name__ == "__main__":

    toy = build_toy_graph()

    print_adjacency(toy)

    print()

    verify_shapes()
