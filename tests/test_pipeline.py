import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from untangle.baseline import RuleScorecard, TabularModel, rule_signals
from untangle.config import GraphConfig, SplitConfig
from untangle.evaluate import best_threshold, evaluate
from untangle.features import FEATURE_NAMES, _core_number, build_features
from untangle.generate import generate
from untangle.splits import make_splits

CFG = GraphConfig(n_accounts=3000, n_rings=6, n_remitter=20, n_settlement=10,
                  n_chitfund=8, n_payroll=6)


@pytest.fixture(scope="module")
def built():
    accounts, tx = generate(CFG)
    return accounts, tx, build_features(accounts, tx, CFG)


def test_features_are_finite_and_complete(built):
    accounts, _, X = built
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == len(accounts)
    assert np.isfinite(X.to_numpy()).all()


def test_core_number_matches_reference():
    """Vectorised peeling must agree with the textbook algorithm exactly."""
    networkx = pytest.importorskip("networkx")
    g = networkx.gnm_random_graph(300, 1500, seed=3)
    U = networkx.to_scipy_sparse_array(g, format="csr", dtype=float)
    ref = np.zeros(300)
    for node, k in networkx.core_number(g).items():
        ref[node] = k
    assert np.array_equal(_core_number(sp.csr_matrix(U)), ref)


def test_splits_keep_rings_whole(built):
    accounts, _, _ = built
    masks = make_splits(accounts, SplitConfig(seed=3))
    assert sum(m.sum() for m in masks.values()) == len(accounts)
    assert not (masks["train"] & masks["test"]).any()

    where = np.full(len(accounts), "", dtype=object)
    for name, m in masks.items():
        where[m] = name
    ring = accounts["ring_id"].to_numpy()
    for r in np.unique(ring[ring >= 0]):
        assert len(set(where[ring == r])) == 1, f"ring {r} straddles a split boundary"


def test_rule_signals_are_binary(built):
    _, _, X = built
    sig = rule_signals(X, CFG)
    assert set(np.unique(sig.to_numpy())) <= {0.0, 1.0}


def test_best_threshold_beats_flagging_everything(built):
    accounts, _, X = built
    y = accounts["is_mule"].to_numpy()
    model = TabularModel(seed=0)
    masks = make_splits(accounts, SplitConfig(seed=3))
    model.fit(X, y, masks["train"])
    s = model.score(X)
    thr = best_threshold(y[masks["val"]], s[masks["val"]])
    assert thr > s.min()


@pytest.mark.parametrize("model_cls", [RuleScorecard, TabularModel])
def test_models_run_end_to_end(built, model_cls):
    accounts, _, X = built
    y = accounts["is_mule"].to_numpy()
    masks = make_splits(accounts, SplitConfig(seed=3))
    model = model_cls() if model_cls is TabularModel else model_cls(CFG)
    model.fit(X, y, masks["train"])
    scores = model.score(X)
    assert scores.shape == (len(accounts),)
    rep = evaluate(model.name, accounts, scores, masks["test"],
                   best_threshold(y[masks["val"]], scores[masks["val"]]))
    assert 0.0 <= rep.precision <= 1.0
    assert 0.0 <= rep.recall <= 1.0
    assert 0.0 <= rep.ring_detection_rate <= 1.0


def test_tabular_beats_rules_on_ranking(built):
    """Sanity floor for the benchmark: the learned model should rank better than
    the hand-written scorecard. If it does not, something upstream is broken."""
    accounts, _, X = built
    y = accounts["is_mule"].to_numpy()
    masks = make_splits(accounts, SplitConfig(seed=3))
    reports = []
    for model in (RuleScorecard(CFG), TabularModel(seed=0)):
        model.fit(X, y, masks["train"])
        s = model.score(X)
        reports.append(evaluate(model.name, accounts, s, masks["test"],
                                best_threshold(y[masks["val"]], s[masks["val"]])))
    assert reports[1].pr_auc > reports[0].pr_auc
