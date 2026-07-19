import numpy as np
import pytest
import torch

from untangle.config import GraphConfig, SplitConfig
from untangle.features import build_features
from untangle.generate import generate
from untangle.gnn import ARCHITECTURES, GATLayer, GraphModel, build_edges, mean_adjacency
from untangle.splits import make_splits

CFG = GraphConfig(n_accounts=1500, n_rings=4, n_remitter=10, n_settlement=6,
                  n_chitfund=5, n_payroll=4)


@pytest.fixture(scope="module")
def built():
    accounts, tx = generate(CFG)
    X = build_features(accounts, tx, CFG)
    masks = make_splits(accounts, SplitConfig(seed=3))
    return accounts, tx, X, masks


def test_edges_are_symmetric_with_self_loops(built):
    accounts, tx, _, _ = built
    n = len(accounts)
    edges = build_edges(tx, n)
    pairs = {(int(s), int(d)) for s, d in edges.t()}
    assert all((d, s) in pairs for s, d in pairs), "edge list must be undirected"
    assert all((i, i) in pairs for i in range(n)), "every node needs a self-loop"


def test_mean_adjacency_rows_sum_to_one(built):
    accounts, tx, _, _ = built
    n = len(accounts)
    adj = mean_adjacency(build_edges(tx, n), n)
    rows = torch.sparse.mm(adj, torch.ones(n, 1)).squeeze()
    assert torch.allclose(rows, torch.ones(n), atol=1e-5)


def test_gat_attention_is_a_distribution_over_incoming_edges(built):
    """Attention must sum to 1 per receiving node, or the reason codes built on
    top of it are meaningless."""
    accounts, tx, X, _ = built
    n = len(accounts)
    edges = build_edges(tx, n)
    layer = GATLayer(X.shape[1], 32, heads=4)
    h = torch.randn(n, X.shape[1])
    alpha = layer._alpha(layer.lin(h).view(n, 4, 8), edges)
    assert (alpha >= 0).all() and torch.isfinite(alpha).all()
    total = torch.zeros(n, 4).index_add_(0, edges[1], alpha)
    assert torch.allclose(total, torch.ones(n, 4), atol=1e-4)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_each_architecture_trains_and_scores(built, arch):
    accounts, tx, X, masks = built
    y = accounts["is_mule"].to_numpy()
    model = GraphModel(tx, len(accounts), arch=arch, epochs=5, seed=0)
    model.fit(X, y, masks["train"], val_mask=masks["val"])
    s = model.score(X)
    assert s.shape == (len(accounts),)
    assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()


def test_mlp_ablation_ignores_the_graph(built):
    """The control must be genuinely blind to structure. If shuffling the edges
    changes its output, it is not a valid ablation."""
    accounts, tx, X, masks = built
    y = accounts["is_mule"].to_numpy()
    n = len(accounts)

    a = GraphModel(tx, n, arch="mlp", epochs=3, seed=0)
    a.fit(X, y, masks["train"], val_mask=masks["val"])
    scores_real = a.score(X)

    shuffled = tx.copy()
    rng = np.random.default_rng(0)
    shuffled["dst"] = rng.permutation(shuffled["dst"].to_numpy())
    shuffled = shuffled[shuffled.src != shuffled.dst]
    b = GraphModel(shuffled, n, arch="mlp", epochs=3, seed=0)
    b.fit(X, y, masks["train"], val_mask=masks["val"])

    assert np.allclose(scores_real, b.score(X))


def test_sage_actually_uses_the_graph(built):
    """Mirror image of the ablation test: rewiring the graph must change what
    GraphSAGE predicts, otherwise message passing is silently a no-op."""
    accounts, tx, X, masks = built
    y = accounts["is_mule"].to_numpy()
    n = len(accounts)

    a = GraphModel(tx, n, arch="sage", epochs=3, seed=0)
    a.fit(X, y, masks["train"], val_mask=masks["val"])

    shuffled = tx.copy()
    rng = np.random.default_rng(0)
    shuffled["dst"] = rng.permutation(shuffled["dst"].to_numpy())
    shuffled = shuffled[shuffled.src != shuffled.dst]
    b = GraphModel(shuffled, n, arch="sage", epochs=3, seed=0)
    b.fit(X, y, masks["train"], val_mask=masks["val"])

    assert not np.allclose(a.score(X), b.score(X))


def test_training_is_reproducible(built):
    accounts, tx, X, masks = built
    y = accounts["is_mule"].to_numpy()
    runs = []
    for _ in range(2):
        m = GraphModel(tx, len(accounts), arch="sage", epochs=5, seed=1)
        m.fit(X, y, masks["train"], val_mask=masks["val"])
        runs.append(m.score(X))
    assert np.allclose(runs[0], runs[1])
