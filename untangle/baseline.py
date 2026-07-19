"""The two things a bank already has, implemented honestly.

`RuleScorecard` is the deployed reality: a handful of typology rules with
hand-set weights, summed into a score, alerting above a cut-off. The rules here
are the ones AML teams actually write -- pass-through, rapid forwarding,
threshold structuring, device reuse, dormant-account reactivation, fan-in bursts.

`TabularModel` is the upgrade path most fraud teams take next: gradient boosting
over the same per-account feature table. It is a genuinely strong detector and
it is the number worth beating -- a graph model that only outperforms the rule
engine has proved nothing that a weekend of feature engineering wouldn't.

Both consume exactly the feature frame from `features.build_features`, so any
gap against the graph model comes from message passing and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import GraphConfig


def _raw(X: pd.DataFrame, col: str) -> np.ndarray:
    """Undo the log1p that build_features applies, for human-readable thresholds."""
    return np.expm1(X[col].to_numpy())


# ---------------------------------------------------------------------------
# rule engine
# ---------------------------------------------------------------------------

# (name, weight) -- weights are the analyst's prior on how damning each typology
# is, not fitted parameters. Only the alert cut-off is tuned, on validation.
RULE_WEIGHTS = {
    "pass_through": 3.0,
    "rapid_forward": 2.5,
    "structuring": 3.0,
    "device_reuse": 2.0,
    "new_account_high_value": 2.0,
    "fan_in_burst": 1.5,
    "dormant_reactivation": 1.0,
}


def rule_signals(X: pd.DataFrame, cfg: GraphConfig | None = None) -> pd.DataFrame:
    """Each column is one typology rule firing (0/1) for each account."""
    cfg = cfg or GraphConfig()
    amt_in = _raw(X, "amt_in_log")
    n_in = _raw(X, "n_in_log")
    cp_in = _raw(X, "cp_in_log")
    age = _raw(X, "age_days_log")

    return pd.DataFrame(
        {
            # money arrives from several sources and leaves almost untouched
            "pass_through": (
                (X["pass_through"] > 0.85) & (amt_in > 50_000) & (cp_in >= 5)
            ).astype(float),
            # onward transfer within two hours of a credit
            "rapid_forward": ((X["min_fwd_hours"] < 2.0) & (n_in >= 3)).astype(float),
            # repeatedly parked under the reporting threshold -- one such credit is
            # a rent payment, so the rule needs several before it means anything
            "structuring": ((X["frac_near_threshold_in"] > 0.50) & (n_in >= 5)).astype(float),
            # one handset, several customers
            "device_reuse": (X["device_share_count"] >= 3).astype(float),
            # a months-old account handling serious money
            "new_account_high_value": ((age < 180) & (amt_in > 100_000)).astype(float),
            # most of the account's life happened in one hour
            "fan_in_burst": ((X["max_burst_1h"] > 0.40) & (cp_in >= 5)).astype(float),
            # long-quiet account wakes up and moves value
            "dormant_reactivation": (
                (age > 365) & (X["active_span_hours"] < 72) & (amt_in > 200_000)
            ).astype(float),
        },
        index=X.index,
    )


class RuleScorecard:
    """Weighted sum of typology rules; only the cut-off is learned."""

    name = "rules"

    def __init__(self, cfg: GraphConfig | None = None):
        self.cfg = cfg or GraphConfig()

    def fit(self, X: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
            val_mask: np.ndarray | None = None) -> "RuleScorecard":
        return self  # nothing to fit -- that is the point of a rule engine

    def score(self, X: pd.DataFrame) -> np.ndarray:
        sig = rule_signals(X, self.cfg)
        w = np.array([RULE_WEIGHTS[c] for c in sig.columns])
        return (sig.to_numpy() * w).sum(axis=1)

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        return rule_signals(X, self.cfg)


# ---------------------------------------------------------------------------
# tabular ML
# ---------------------------------------------------------------------------


class TabularModel:
    """Gradient boosting on the per-account feature table. No graph structure --
    it sees each account as an isolated row of numbers, which is exactly the
    limitation the graph model is meant to remove."""

    name = "gbdt"

    def __init__(self, seed: int = 0):
        self.clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
            val_mask: np.ndarray | None = None) -> "TabularModel":
        self.clf.fit(X.to_numpy()[mask], y[mask])
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X.to_numpy())[:, 1]
