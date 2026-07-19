"""Per-account features derived from the transaction stream and the graph.

Six families, and the split matters for the story we tell about results:

    profile      what the bank knows before any money moves
    volume       how much and how often
    flow         the shape of money in vs money out
    velocity     how fast money is passed on
    structuring  whether amounts hug the reporting threshold
    topology     the account's position in the graph

The rule baseline gets exactly these features, so any lift the GNN shows comes
from message passing over neighbours -- not from a better feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import GraphConfig

FEATURE_FAMILIES = {
    "profile": ["age_days_log", "kyc_level", "device_share_count", "is_business", "is_merchant"],
    "volume": ["n_in_log", "n_out_log", "amt_in_log", "amt_out_log", "cp_in_log", "cp_out_log",
               "mean_amt_in_log", "mean_amt_out_log", "max_amt_in_log"],
    "flow": ["pass_through", "retained_ratio", "net_flow_signed_log"],
    "velocity": ["median_fwd_hours", "min_fwd_hours", "active_span_hours", "max_burst_1h",
                 "tx_per_active_day"],
    "structuring": ["frac_near_threshold_in", "frac_near_threshold_out"],
    "topology": ["pagerank_log", "core_number", "deg_total_log", "reciprocity", "mean_nbr_deg_log",
                 "in_out_deg_ratio"],
}

FEATURE_NAMES = [f for fam in FEATURE_FAMILIES.values() for f in fam]

_NO_EVENT = 999.0  # sentinel hours for "never forwarded"


def _log1p(x):
    return np.log1p(np.asarray(x, dtype=np.float64))


def _keys(account: np.ndarray, t: np.ndarray, stride: float) -> np.ndarray:
    """Pack (account, time) into one sortable float.

    Because every account's events occupy a disjoint interval of the key space,
    one global sort plus one searchsorted answers a per-account question for
    every account at once -- no grouping, no Python loop over 40k accounts.
    """
    return account.astype(np.float64) * stride + t


def _forward_latency(n: int, tx: pd.DataFrame, stride: float) -> tuple[np.ndarray, np.ndarray]:
    """For every credit, how long until the account's next debit.

    This is the single most discriminative non-graph signal: a real account sits
    on its money, a mule forwards it. Returns (median, min) per account.
    """
    debit_keys = np.sort(_keys(tx["src"].to_numpy(), tx["t_hours"].to_numpy(), stride))
    cred_acct = tx["dst"].to_numpy()
    cred_keys = _keys(cred_acct, tx["t_hours"].to_numpy(), stride)

    idx = np.searchsorted(debit_keys, cred_keys, side="left")
    ok = idx < len(debit_keys)
    nxt = debit_keys[np.minimum(idx, len(debit_keys) - 1)]
    # the next debit must belong to the same account, not to the next one along
    ok &= np.floor(nxt / stride) == cred_acct
    lat = nxt - cred_keys

    median = np.full(n, _NO_EVENT)
    fastest = np.full(n, _NO_EVENT)
    if ok.any():
        df = pd.DataFrame({"a": cred_acct[ok], "lat": lat[ok]})
        g = df.groupby("a")["lat"]
        median[g.median().index.to_numpy()] = g.median().to_numpy()
        fastest[g.min().index.to_numpy()] = g.min().to_numpy()
    return median, fastest


def _burstiness(n: int, tx: pd.DataFrame, stride: float) -> np.ndarray:
    """Largest share of an account's activity crammed into any one-hour window."""
    acct = np.concatenate([tx["src"].to_numpy(), tx["dst"].to_numpy()])
    t = np.concatenate([tx["t_hours"].to_numpy(), tx["t_hours"].to_numpy()])
    keys = np.sort(_keys(acct, t, stride))

    # events within the next hour, for the same account
    ahead = np.searchsorted(keys, keys + 1.0, side="right") - np.arange(1, len(keys) + 1)
    owner = np.floor(keys / stride).astype(np.int64)
    window = ahead + 1  # count the event itself

    best = np.zeros(n)
    np.maximum.at(best, owner, window)
    total = np.bincount(owner, minlength=n).astype(float)
    return np.where(total > 1, best / np.maximum(total, 1), 0.0)


def _core_number(U: sp.spmatrix) -> np.ndarray:
    """k-core number of every node, by vectorised peeling.

    Each round is one sparse mat-vec, so the whole thing costs a few hundred
    milliseconds on a million edges -- where handing the graph to networkx costs
    the better part of a minute just to build the dict-of-dicts.
    """
    B = U.copy().tocsr()
    B.data[:] = 1.0
    B.setdiag(0)
    B.eliminate_zeros()

    n = B.shape[0]
    deg = np.asarray(B.sum(axis=1)).ravel()
    alive = deg > 0
    core = np.zeros(n)

    k = 1
    while alive.any():
        while True:
            drop = alive & (deg < k)
            if not drop.any():
                break
            core[drop] = k - 1
            alive &= ~drop
            deg = deg - (B @ drop.astype(np.float64))
            deg[~alive] = 0.0
        k += 1
    return core


def _topology(n: int, tx: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sparse-matrix graph features. Deliberately avoids anything quadratic in
    degree -- merchant hubs here have five-figure degree and triangle counting
    on them would dominate the whole pipeline."""
    edges = tx.groupby(["src", "dst"], sort=False)["amount"].sum().reset_index()
    r = edges["src"].to_numpy()
    c = edges["dst"].to_numpy()
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))

    out_deg = np.asarray(A.sum(axis=1)).ravel()
    in_deg = np.asarray(A.sum(axis=0)).ravel()
    deg = out_deg + in_deg

    # PageRank by power iteration on the column-stochastic matrix -- 30 iterations
    # is plenty for a ranking feature and avoids a networkx pass over 1M edges.
    d = 0.85
    outsum = np.where(out_deg > 0, out_deg, 1.0)
    M = sp.csr_matrix((np.ones(len(r)) / outsum[r], (c, r)), shape=(n, n))
    dangling = out_deg == 0
    pr = np.full(n, 1.0 / n)
    for _ in range(30):
        pr = (1 - d) / n + d * (M @ pr + pr[dangling].sum() / n)

    # reciprocity: share of an account's counterparties it both pays and is paid by
    mutual = np.asarray(A.multiply(A.T).sum(axis=1)).ravel()
    reciprocity = mutual / np.maximum(deg, 1)

    # average degree of the accounts you touch -- separates hub-adjacent from
    # periphery-adjacent accounts without any 2-hop enumeration
    U = A + A.T
    nbr_deg_sum = np.asarray(U @ deg.reshape(-1, 1)).ravel()
    mean_nbr_deg = nbr_deg_sum / np.maximum(deg, 1)

    core = _core_number(U)

    return {
        "pagerank_log": -np.log10(np.maximum(pr, 1e-12)),
        "core_number": core,
        "deg_total_log": _log1p(deg),
        "reciprocity": reciprocity,
        "mean_nbr_deg_log": _log1p(mean_nbr_deg),
        "in_out_deg_ratio": in_deg / np.maximum(in_deg + out_deg, 1),
    }


def build_features(
    accounts: pd.DataFrame, tx: pd.DataFrame, cfg: GraphConfig | None = None
) -> pd.DataFrame:
    """Returns a frame indexed by account_id with exactly FEATURE_NAMES columns."""
    cfg = cfg or GraphConfig()
    n = len(accounts)
    thr = cfg.reporting_threshold

    g_out = tx.groupby("src")
    g_in = tx.groupby("dst")

    def col(grouped, agg_col, how, fill=0.0):
        s = getattr(grouped[agg_col], how)()
        return s.reindex(range(n), fill_value=fill).to_numpy()

    n_out = col(g_out, "amount", "count")
    n_in = col(g_in, "amount", "count")
    amt_out = col(g_out, "amount", "sum")
    amt_in = col(g_in, "amount", "sum")
    cp_out = g_out["dst"].nunique().reindex(range(n), fill_value=0).to_numpy()
    cp_in = g_in["src"].nunique().reindex(range(n), fill_value=0).to_numpy()
    max_in = col(g_in, "amount", "max")

    # money that arrives and leaves again, vs money that stays
    lo = np.minimum(amt_in, amt_out)
    hi = np.maximum(amt_in, amt_out)
    pass_through = lo / np.maximum(hi, 1.0)
    retained = (amt_in - amt_out) / np.maximum(hi, 1.0)

    # activity window
    t_min = np.minimum(
        col(g_out, "t_hours", "min", np.inf), col(g_in, "t_hours", "min", np.inf)
    )
    t_max = np.maximum(
        col(g_out, "t_hours", "max", -np.inf), col(g_in, "t_hours", "max", -np.inf)
    )
    span = np.where(np.isfinite(t_min) & np.isfinite(t_max), t_max - t_min, 0.0)
    n_tx = n_in + n_out

    stride = float(tx["t_hours"].max()) + 2.0
    med_fwd, min_fwd = _forward_latency(n, tx, stride)

    near = tx["amount"].between(0.75 * thr, thr)
    near_in = tx[near].groupby("dst").size().reindex(range(n), fill_value=0).to_numpy()
    near_out = tx[near].groupby("src").size().reindex(range(n), fill_value=0).to_numpy()

    feats = {
        "age_days_log": _log1p(accounts["age_days"].to_numpy()),
        "kyc_level": accounts["kyc_level"].to_numpy().astype(float),
        "device_share_count": accounts["device_share_count"].to_numpy().astype(float),
        "is_business": (accounts["kind"] == "business").to_numpy().astype(float),
        "is_merchant": (accounts["kind"] == "merchant").to_numpy().astype(float),
        "n_in_log": _log1p(n_in),
        "n_out_log": _log1p(n_out),
        "amt_in_log": _log1p(amt_in),
        "amt_out_log": _log1p(amt_out),
        "cp_in_log": _log1p(cp_in),
        "cp_out_log": _log1p(cp_out),
        "mean_amt_in_log": _log1p(amt_in / np.maximum(n_in, 1)),
        "mean_amt_out_log": _log1p(amt_out / np.maximum(n_out, 1)),
        "max_amt_in_log": _log1p(max_in),
        "pass_through": pass_through,
        "retained_ratio": retained,
        "net_flow_signed_log": np.sign(amt_in - amt_out) * _log1p(np.abs(amt_in - amt_out)),
        "median_fwd_hours": med_fwd,
        "min_fwd_hours": min_fwd,
        "active_span_hours": span,
        "max_burst_1h": _burstiness(n, tx, stride),
        "tx_per_active_day": n_tx / np.maximum(span / 24.0, 1.0),
        "frac_near_threshold_in": near_in / np.maximum(n_in, 1),
        "frac_near_threshold_out": near_out / np.maximum(n_out, 1),
    }
    feats.update(_topology(n, tx))

    X = pd.DataFrame(feats, index=accounts["account_id"].to_numpy())[FEATURE_NAMES]
    return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
