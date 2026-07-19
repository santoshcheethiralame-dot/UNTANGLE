"""Scoring. Three questions, because a single F1 hides the thing that matters.

    node level   of the accounts we flagged, how many were mules?
    ring level   of the rings that existed, how many did we surface at all,
                 and how much of each did we recover?
    queue level  an investigator works a ranked list with a fixed daily budget;
                 what do the top 100 alerts actually contain?

The ring-level view is the one that justifies a graph model. A detector can post
a respectable node F1 by picking off the obvious fresh mules in every ring while
never revealing a single ring completely -- which leaves the money moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class Report:
    name: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    n_flagged: int
    threshold: float
    ring_detection_rate: float
    mean_member_recovery: float
    precision_at_100: float
    recall_by_tactic: dict = field(default_factory=dict)
    recall_by_role: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "model": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "ring_rate": self.ring_detection_rate,
            "member_recov": self.mean_member_recovery,
            "p@100": self.precision_at_100,
            "flagged": self.n_flagged,
        }

    def __str__(self) -> str:
        lines = [
            f"{self.name}",
            f"  node    P {self.precision:.3f}   R {self.recall:.3f}   F1 {self.f1:.3f}   "
            f"PR-AUC {self.pr_auc:.3f}   ROC-AUC {self.roc_auc:.3f}",
            f"  ring    detected {self.ring_detection_rate:.0%} of rings   "
            f"recovered {self.mean_member_recovery:.0%} of members on average",
            f"  queue   precision@100 {self.precision_at_100:.3f}   "
            f"({self.n_flagged:,} accounts flagged at threshold {self.threshold:.3f})",
        ]
        if self.recall_by_tactic:
            parts = "  ".join(f"{k} {v:.2f}" for k, v in sorted(self.recall_by_tactic.items()))
            lines.append(f"  tactic  {parts}")
        if self.recall_by_role:
            parts = "  ".join(f"{k} {v:.2f}" for k, v in sorted(self.recall_by_role.items()))
            lines.append(f"  source  {parts}")
        return "\n".join(lines)


def best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1. Chosen on validation, never on test."""
    order = np.argsort(-scores)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return float(scores[order][int(np.argmax(f1))])


def evaluate(
    name: str,
    accounts: pd.DataFrame,
    scores: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    queue_size: int = 100,
) -> Report:
    sub = accounts.loc[mask]
    y = sub["is_mule"].to_numpy()
    s = scores[mask]
    pred = s >= threshold

    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # ring level: a ring counts as detected if any of its mules was flagged
    rings = sub.loc[sub.is_mule == 1, "ring_id"]
    detected, recovery = [], []
    for ring_id, idx in rings.groupby(rings).groups.items():
        members = sub.index.get_indexer(idx)
        hit = pred[members]
        detected.append(hit.any())
        recovery.append(hit.mean())

    top = np.argsort(-s)[:queue_size]

    def breakdown(col):
        out = {}
        for key, idx in sub.loc[sub.is_mule == 1].groupby(col).groups.items():
            if not key:
                continue
            out[str(key)] = float(pred[sub.index.get_indexer(idx)].mean())
        return out

    return Report(
        name=name,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=float(average_precision_score(y, s)) if y.any() else float("nan"),
        roc_auc=float(roc_auc_score(y, s)) if y.any() else float("nan"),
        n_flagged=int(pred.sum()),
        threshold=float(threshold),
        ring_detection_rate=float(np.mean(detected)) if detected else float("nan"),
        mean_member_recovery=float(np.mean(recovery)) if recovery else float("nan"),
        precision_at_100=float(y[top].mean()),
        recall_by_tactic=breakdown("ring_tactic"),
        recall_by_role=breakdown("ring_role"),
    )


def comparison_table(reports: list[Report]) -> str:
    df = pd.DataFrame([r.row() for r in reports])
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}")
