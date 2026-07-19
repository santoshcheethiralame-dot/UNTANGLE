"""Command line: build a world, or benchmark detectors on one.

    python -m untangle generate            write data/accounts.csv, data/transactions.csv
    python -m untangle bench               generate (or load), then score every detector
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .baseline import RuleScorecard, TabularModel
from .config import GraphConfig, SplitConfig
from .evaluate import best_threshold, comparison_table, evaluate
from .features import build_features
from .generate import generate, summarize
from .splits import describe, make_splits

DATA = Path("data")


def _load_or_build(args) -> tuple[pd.DataFrame, pd.DataFrame, GraphConfig]:
    cfg = GraphConfig(n_accounts=args.accounts, n_rings=args.rings, seed=args.seed,
                      n_days=args.days)
    acc_path, tx_path = DATA / "accounts.csv", DATA / "transactions.csv"
    if args.reuse and acc_path.exists() and tx_path.exists():
        print(f"loading {acc_path} / {tx_path}")
        return pd.read_csv(acc_path, keep_default_na=False), pd.read_csv(tx_path), cfg

    t = time.time()
    accounts, tx = generate(cfg)
    print(f"generated in {time.time() - t:.1f}s\n")
    print(summarize(accounts, tx))
    DATA.mkdir(exist_ok=True)
    accounts.to_csv(acc_path, index=False)
    tx.to_csv(tx_path, index=False)
    print(f"\nwrote {acc_path} and {tx_path}")
    return accounts, tx, cfg


def cmd_generate(args) -> None:
    _load_or_build(args)


def cmd_bench(args) -> None:
    accounts, tx, cfg = _load_or_build(args)
    print()

    t = time.time()
    X = build_features(accounts, tx, cfg)
    print(f"built {X.shape[1]} features for {len(X):,} accounts in {time.time() - t:.1f}s")

    masks = make_splits(accounts, SplitConfig(seed=args.seed))
    print(describe(accounts, masks))
    print()

    y = accounts["is_mule"].to_numpy()
    reports = []
    for model in (RuleScorecard(cfg), TabularModel(seed=args.seed)):
        model.fit(X, y, masks["train"])
        scores = model.score(X)
        thr = best_threshold(y[masks["val"]], scores[masks["val"]])
        rep = evaluate(model.name, accounts, scores, masks["test"], thr)
        reports.append(rep)
        print(rep, "\n")

    print(comparison_table(reports))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="untangle")
    p.add_argument("--accounts", type=int, default=GraphConfig.n_accounts)
    p.add_argument("--rings", type=int, default=GraphConfig.n_rings)
    p.add_argument("--days", type=int, default=GraphConfig.n_days)
    p.add_argument("--seed", type=int, default=GraphConfig.seed)
    p.add_argument("--reuse", action="store_true", help="use data/ on disk if present")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=cmd_generate)
    sub.add_parser("bench").set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
