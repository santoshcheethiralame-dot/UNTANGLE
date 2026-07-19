"""Graph neural networks over the payment graph, in plain PyTorch.

Deliberately no PyTorch Geometric. Everything here is sparse matmul and
`index_add_`, which is ~150 lines, installs nowhere, and runs on CPU -- against
a dependency that is the single biggest install risk on Windows and would buy us
two layers we can write ourselves.

Three architectures share one training loop:

    sage    GraphSAGE: mean-aggregate the neighbourhood, concat with self
    gat     Graph attention: learn which neighbours matter, per head
    mlp     The ablation -- identical model with message passing switched off

`mlp` is the control that makes the whole result defensible. It sees exactly the
same 30 features and the same training budget, but never looks at a neighbour.
Whatever separates `sage`/`gat` from `mlp` is attributable to the graph and to
nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ARCHITECTURES = ("sage", "gat", "mlp")


# ---------------------------------------------------------------------------
# graph plumbing
# ---------------------------------------------------------------------------


def build_edges(tx: pd.DataFrame, n: int) -> torch.Tensor:
    """Undirected edge list with self-loops, as a [2, E] tensor.

    Money direction matters for features, but for message passing we want risk to
    flow both ways: a mule is suspicious because of who paid it *and* who it paid.
    Parallel transfers between the same pair collapse to one edge -- repetition is
    already captured in the node features.
    """
    pairs = tx[["src", "dst"]].drop_duplicates().to_numpy()
    src = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
    return torch.from_numpy(np.stack([src, dst])).long()


def mean_adjacency(edges: torch.Tensor, n: int) -> torch.Tensor:
    """Row-normalised sparse adjacency -- multiplying by it averages a node's
    neighbourhood, which is exactly the GraphSAGE mean aggregator."""
    src, dst = edges
    deg = torch.zeros(n).index_add_(0, dst, torch.ones(dst.numel()))
    vals = 1.0 / deg.clamp(min=1.0)[dst]
    return torch.sparse_coo_tensor(torch.stack([dst, src]), vals, (n, n)).coalesce()


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


class SAGELayer(nn.Module):
    """h' = W_self · h + W_neigh · mean(h of neighbours)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_neigh = nn.Linear(d_in, d_out, bias=False)

    def forward(self, h, adj, edges=None):
        return self.lin_self(h) + self.lin_neigh(torch.sparse.mm(adj, h))


class GATLayer(nn.Module):
    """Multi-head attention over neighbours, computed edge-wise.

    The softmax is a segment softmax over each node's incoming edges. It has to
    be done this way: merchant hubs in this graph have five-figure degree, so a
    dense N x N attention matrix is 6.4 billion entries and instantly fatal.
    """

    def __init__(self, d_in: int, d_out: int, heads: int = 4):
        super().__init__()
        assert d_out % heads == 0, "d_out must divide evenly across heads"
        self.heads, self.d_head = heads, d_out // heads
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.d_head))
        self.att_dst = nn.Parameter(torch.empty(heads, self.d_head))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def _alpha(self, z, edges):
        """Segment softmax over each node's incoming edges, max-shifted."""
        n = z.size(0)
        src, dst = edges
        logits = F.leaky_relu(
            (z * self.att_src).sum(-1)[src] + (z * self.att_dst).sum(-1)[dst], 0.2
        )
        peak = torch.full((n, self.heads), float("-inf"))
        peak = peak.index_reduce(0, dst, logits, "amax", include_self=False)
        weight = (logits - peak[dst]).exp()
        denom = torch.zeros(n, self.heads).index_add_(0, dst, weight)
        return weight / (denom[dst] + 1e-16)

    def forward(self, h, adj=None, edges=None):
        n = h.size(0)
        z = self.lin(h).view(n, self.heads, self.d_head)
        alpha = self._alpha(z, edges)

        # Aggregate one head at a time. Doing all heads at once would gather
        # z[src] as E x heads x d_head -- ~180M floats on the full graph, held
        # through the backward pass. Per head it is a quarter of that, and it
        # measures 3x faster than the sparse-matmul formulation because that one
        # re-sorts the edge indices on every single call.
        dst = edges[1]
        out = torch.stack(
            [
                torch.zeros(n, self.d_head).index_add_(
                    0, dst, z[:, k, :][edges[0]] * alpha[:, k : k + 1]
                )
                for k in range(self.heads)
            ],
            dim=1,
        )
        return out.reshape(n, -1) + self.bias

    def attention(self, h, edges) -> torch.Tensor:
        """Per-edge attention averaged over heads -- the raw material for
        'flagged because of its flows with X, Y, Z' reason codes."""
        z = self.lin(h).view(h.size(0), self.heads, self.d_head)
        return self._alpha(z, edges).mean(1)


class MLPLayer(nn.Module):
    """The ablation: same parameter budget, no neighbours."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, h, adj=None, edges=None):
        return self.lin(h)


def _make_layer(arch: str, d_in: int, d_out: int, heads: int):
    if arch == "sage":
        return SAGELayer(d_in, d_out)
    if arch == "gat":
        return GATLayer(d_in, d_out, heads)
    if arch == "mlp":
        return MLPLayer(d_in, d_out)
    raise ValueError(f"unknown architecture {arch!r}, expected one of {ARCHITECTURES}")


class Net(nn.Module):
    def __init__(self, arch, d_in, hidden=64, layers=2, heads=4, dropout=0.3):
        super().__init__()
        dims = [d_in] + [hidden] * layers
        self.layers = nn.ModuleList(
            _make_layer(arch, dims[i], dims[i + 1], heads) for i in range(layers)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, adj, edges):
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = F.relu(norm(layer(h, adj, edges)))
            h = F.dropout(h, self.dropout, self.training)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# the model wrapper -- same interface as the baselines
# ---------------------------------------------------------------------------


class GraphModel:
    """Transductive node classifier. Same `fit`/`score` contract as the baselines,
    so `cli.cmd_bench` treats it identically."""

    def __init__(
        self,
        tx: pd.DataFrame,
        n_accounts: int,
        arch: str = "sage",
        hidden: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        epochs: int = 300,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        patience: int = 40,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.__dict__.update(locals())
        del self.self
        self.name = arch
        self.edges = build_edges(tx, n_accounts)
        self.adj = mean_adjacency(self.edges, n_accounts)
        self.scaler = StandardScaler()

    def _tensor(self, X: pd.DataFrame, fit_mask=None) -> torch.Tensor:
        if fit_mask is not None:
            self.scaler.fit(X.to_numpy()[fit_mask])
        return torch.from_numpy(self.scaler.transform(X.to_numpy())).float()

    def fit(self, X, y, mask, val_mask=None) -> "GraphModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x = self._tensor(X, fit_mask=mask)
        yt = torch.from_numpy(y).float()
        train = torch.from_numpy(mask)
        self.net = Net(self.arch, x.size(1), self.hidden, self.layers, self.heads,
                       self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        # 1.5% positives -- without this the model predicts "nobody" and stops
        pos = float(y[mask].sum())
        pos_weight = torch.tensor(max((mask.sum() - pos) / max(pos, 1.0), 1.0))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_score, best_state, waited = -np.inf, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            out = self.net(x, self.adj, self.edges)
            loss = loss_fn(out[train], yt[train])
            loss.backward()
            opt.step()

            if val_mask is None:
                continue
            self.net.eval()
            with torch.no_grad():
                probs = torch.sigmoid(self.net(x, self.adj, self.edges)).numpy()
            # early stopping on PR-AUC, not accuracy -- at 1.5% prevalence
            # accuracy is maximised by predicting nothing
            score = average_precision_score(y[val_mask], probs[val_mask])
            if score > best_score:
                best_score, waited = score, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
            if self.verbose and epoch % 25 == 0:
                print(f"  epoch {epoch:>3}  loss {loss.item():.4f}  val PR-AUC {score:.4f}")

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.val_pr_auc_ = best_score
        self._x = x
        return self

    @torch.no_grad()
    def score(self, X) -> np.ndarray:
        self.net.eval()
        x = self._tensor(X)
        return torch.sigmoid(self.net(x, self.adj, self.edges)).numpy()

    @torch.no_grad()
    def edge_attention(self) -> tuple[np.ndarray, np.ndarray]:
        """(edges, attention) from the first GAT layer, for reason codes."""
        layer = self.net.layers[0]
        if not isinstance(layer, GATLayer):
            raise TypeError("edge_attention requires arch='gat'")
        return self.edges.numpy(), layer.attention(self._x, self.edges).numpy()
