"""Train / validation / test masks.

Split by *ring*, not by account. A ring is one coordinated event; if half its
mules land in train and half in test, the test score is measuring memorisation
of that specific ring rather than the ability to spot the next one. Everything
below keeps a ring whole.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SplitConfig


def make_splits(accounts: pd.DataFrame, cfg: SplitConfig | None = None) -> dict[str, np.ndarray]:
    """Returns boolean masks {'train','val','test'} over accounts, in row order."""
    cfg = cfg or SplitConfig()
    rng = np.random.default_rng(cfg.seed)
    n = len(accounts)
    ring = accounts["ring_id"].to_numpy()

    split = np.empty(n, dtype=object)

    # 1. deal whole rings out, so ring membership never crosses a boundary
    rings = np.unique(ring[ring >= 0])
    rng.shuffle(rings)
    n_test = int(round(len(rings) * cfg.test_frac))
    n_val = int(round(len(rings) * cfg.val_frac))
    ring_split = {}
    for i, r in enumerate(rings):
        ring_split[r] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    for r, s in ring_split.items():
        split[ring == r] = s

    # 2. everyone not attached to a ring is dealt out independently
    free = split == None  # noqa: E711 -- object dtype, `is None` won't vectorise
    draw = rng.random(free.sum())
    labels = np.where(draw < cfg.test_frac, "test",
                      np.where(draw < cfg.test_frac + cfg.val_frac, "val", "train"))
    split[free] = labels

    return {name: (split == name) for name in ("train", "val", "test")}


def describe(accounts: pd.DataFrame, masks: dict[str, np.ndarray]) -> str:
    y = accounts["is_mule"].to_numpy()
    rows = []
    for name, m in masks.items():
        rings = accounts.loc[m & (accounts.is_mule == 1), "ring_id"].nunique()
        rows.append(f"  {name:<6} {m.sum():>7,} accounts   {y[m].sum():>5,} mules "
                    f"({y[m].mean():.2%})   {rings:>3} rings")
    return "split (ring-disjoint):\n" + "\n".join(rows)
