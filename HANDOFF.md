# Untangle — build handoff

Everything a teammate needs to pick this up cold. Read the "Don't break this"
section before you touch `generate.py`.

- **Built:** Day 1 (synthetic graph engine) and Day 2 (features, baselines, eval).
- **Left:** Day 3 (GNN), Day 4 (ring extraction, API, viz), Day 5 (polish, pitch).
- **Python 3.12**, CPU only. No paid services, no GPU, no external data.

```bash
git clone https://github.com/santoshcheethiralame-dot/UNTANGLE
cd UNTANGLE
pip install -r requirements.txt
python -m untangle bench      # ~15s end to end, prints the table below
pytest tests -q               # 16 tests, ~33s
```

---

## 1. What this project claims, and why the benchmark is built the way it is

The pitch is: *fraud is a network, so detection has to be a network model.*

That claim is falsifiable, and our first working version falsified it — a plain
gradient-boosted tree over 30 per-account columns scored **0.987 F1** with no
graph at all. If that had shipped, the GNN would have been decoration.

So the generator was rebuilt to be *adversarial to our own thesis*. The whole
benchmark is designed so that the only accounts still undetectable by a row-wise
model are the ones identifiable purely by **who they transact with**. That
residual is what the graph model has to claim.

If you make the dataset easier — even accidentally — the project stops proving
anything. Section 6 lists the invariants that protect this, and there are tests
enforcing them.

---

## 2. Repo map

```
untangle/
  __init__.py      exports GraphConfig, SplitConfig
  __main__.py      entry point for `python -m untangle`
  config.py        every generator + split knob, two frozen dataclasses
  generate.py      the synthetic world  (Day 1)
  features.py      30 per-account features (Day 2)
  splits.py        ring-disjoint train/val/test masks
  baseline.py      rule scorecard + gradient boosting
  evaluate.py      node / ring / investigator-queue metrics
  cli.py           `generate` and `bench` subcommands
tests/
  test_generate.py  8 tests — world invariants
  test_pipeline.py  8 tests — features, splits, models, k-core correctness
data/               gitignored; regenerate with `python -m untangle generate`
```

### `config.py`

Two frozen dataclasses. `GraphConfig` has ~30 fields in six groups: scale,
population mix, legit traffic volume, amounts, rings, hard negatives. `.evolve(**kw)`
returns a modified copy. `SplitConfig` holds val/test fractions and a seed.

Defaults: 40,000 accounts, 30 days, 26 rings, seed 7 → 40,210 accounts and
~1.4M transactions at 1.5% mule prevalence.

### `generate.py`

`generate(cfg) -> (accounts, transactions)`, deterministic given the seed.
Assembly order matters — legit traffic, then decoys, then rings (rings recruit
from accounts that already exist and already have history).

Legit flows: `_legit_salary`, `_legit_spend`, `_legit_p2p`, `_legit_rent`,
`_legit_settlement_sweep`.
Hard negatives: `_decoy_payroll`, `_decoy_settlement`, `_decoy_chitfund`,
`_decoy_remitter`.
Rings: `_inject_ring` → `_spawn_mules` + `_ring_amounts`.

`summarize(accounts, tx)` prints the world stats block.

### `features.py`

`build_features(accounts, tx, cfg) -> DataFrame` indexed by `account_id`, with
exactly the columns in `FEATURE_NAMES` (30). All finite, no NaNs.

`_core_number` is a hand-written vectorised k-core peeling on a sparse matrix.
It replaced `networkx.core_number`, which took **80 of the original 96 seconds**
just building a dict-of-dicts for 1.4M edges. It is verified equal to networkx's
output in `test_core_number_matches_reference`. Feature extraction is now ~2s.

`_forward_latency` and `_burstiness` pack `(account, time)` into a single sortable
float so one global sort + `searchsorted` answers a per-account question for all
40k accounts at once, with no Python loop.

### `splits.py`

`make_splits(accounts, cfg) -> {'train','val','test'}` boolean masks. **Splits by
ring, not by account.** A ring is one coordinated event; if half its mules are in
train and half in test, the test score measures memorisation of that ring rather
than generalisation to the next one. Enforced by `test_splits_keep_rings_whole`.

### `baseline.py`

`RuleScorecard` — 7 AML typology rules (pass-through, rapid forward, structuring,
device reuse, new-account-high-value, fan-in burst, dormant reactivation), fixed
analyst weights, only the alert cut-off is tuned on validation. Has `.explain()`
returning which rules fired per account — reuse this for the demo's reason codes.

`TabularModel` — `HistGradientBoostingClassifier`, class-balanced. This is the
real number to beat, not the rule engine.

Both expose the same interface, and **the GNN must implement it too**:

```python
class Model:
    name: str
    def fit(self, X: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> "Model": ...
    def score(self, X: pd.DataFrame) -> np.ndarray:  # one score per account, all N
```

### `evaluate.py`

`best_threshold(y, scores)` — F1-optimal cut-off, chosen on validation only.
`evaluate(name, accounts, scores, mask, threshold) -> Report`.

A `Report` carries node metrics (P/R/F1/PR-AUC/ROC-AUC), **ring metrics**
(detection rate, mean member recovery), **queue metrics** (precision@100), and
recall broken down by ring tactic and by mule sourcing route.

Read the ring and role breakdowns, not F1. A detector can post a decent F1 by
picking off the obvious mule in every ring while never dismantling one.

---

## 3. Data schemas

`data/accounts.csv` — one row per account:

| column | meaning |
|---|---|
| `account_id` | 0..N-1, equals the row index and the graph node id |
| `kind` | `personal` / `business` / `merchant` |
| `age_days`, `kyc_level` | profile at observation time (kyc 0–2) |
| `device_id` | shared handsets collide on this value |
| `community`, `region` | latent social/geographic grouping |
| `is_mule` | **the label**, 0/1 |
| `ring_id` | ring membership, -1 if none (the victim source also carries one) |
| `ring_role` | `mule_fresh` / `mule_rented` / `collector` / `source` / `""` |
| `ring_tactic` | `structuring` / `smurf_wide` / `slow_layer` |
| `decoy_type` | `payroll_hub` / `settlement_merchant` / `chitfund` / `remitter` |
| `device_share_count` | how many accounts share this device |

`data/transactions.csv` — one row per directed transfer:

| column | meaning |
|---|---|
| `tx_id`, `src`, `dst` | ids; sorted by time, `src != dst` guaranteed |
| `amount` | rupees |
| `t_hours` | hours from t=0, range [0, 24 × n_days] |
| `channel` | `UPI` / `IMPS` / `NEFT` |
| `ring_id` | -1 for legitimate and for mule cover traffic |

`ring_role` and `ring_tactic` are **evaluation metadata, not features.** Never
feed them to a model.

---

## 4. The 30 features

| family | features |
|---|---|
| profile (5) | `age_days_log`, `kyc_level`, `device_share_count`, `is_business`, `is_merchant` |
| volume (9) | `n_in_log`, `n_out_log`, `amt_in_log`, `amt_out_log`, `cp_in_log`, `cp_out_log`, `mean_amt_in_log`, `mean_amt_out_log`, `max_amt_in_log` |
| flow (3) | `pass_through`, `retained_ratio`, `net_flow_signed_log` |
| velocity (5) | `median_fwd_hours`, `min_fwd_hours`, `active_span_hours`, `max_burst_1h`, `tx_per_active_day` |
| structuring (2) | `frac_near_threshold_in`, `frac_near_threshold_out` |
| topology (6) | `pagerank_log`, `core_number`, `deg_total_log`, `reciprocity`, `mean_nbr_deg_log`, `in_out_deg_ratio` |

Money and count features are `log1p`'d. `_raw(X, col)` in `baseline.py` inverts
that when you need human-readable thresholds.

**The GNN gets this same table as node features.** That is deliberate: both
baselines see identical inputs, so any lift is attributable to message passing
and to nothing else. Do not add features for the GNN only — if a feature helps,
add it to `FEATURE_NAMES` and re-run the baselines so the comparison stays fair.

---

## 5. Current results — the bar to beat

40,210 accounts · 1.4M transactions · 26 rings · 1.5% prevalence.
Test set: 7,963 accounts, 119 mules, 5 rings never seen in training.

| model | P | R | F1 | PR-AUC | rings found | members recovered | P@100 | alerts |
|---|---|---|---|---|---|---|---|---|
| rules | 0.039 | 0.210 | 0.066 | 0.033 | 100% | 18% | 0.080 | 644 |
| gbdt | 0.918 | 0.471 | 0.622 | 0.660 | 100% | 45% | 0.680 | 61 |

Recall split by how the mule was recruited:

| | fresh mules | **rented mules** |
|---|---|---|
| rules | 0.05 | 0.26 |
| gbdt | **1.00** | **0.18** |

**This is the whole story.** Gradient boosting catches every disposable mule and
18% of the rented ones. It recovers 45% of the average ring, so most of every
ring keeps operating after the alert fires.

Targets for the GNN, in priority order:

1. **rented-mule recall** — the headline. 0.18 → 0.50+ would be a real result.
2. **mean member recovery** — 45% → 75%+. "We surface the whole ring."
3. **PR-AUC** — 0.660 → 0.75+.
4. **precision@100** — 0.680 → 0.85+. This is the investigator-workload story.

Ring detection rate is already 100% for everyone; it is not a differentiator at
this ring size. Don't put it on a slide as a win.

---

## 6. Don't break this (invariants)

These are what make the benchmark meaningful. Several have tests; all of them
have a reason.

1. **Half the ring must stay rented.** `mule_fresh_frac = 0.35`. Rented mules are
   real accounts with genuine history, statistically identical to ordinary
   customers on every profile feature. Raising this makes the problem easy and
   the GNN's lift fake. Guarded by `test_rented_mules_look_ordinary_on_profile`.
2. **Hard negatives are labelled legitimate and must stay populous.** 220
   remitters, 120 settlement merchants, 80 chit funds, 60 payroll hubs. They are
   the reason "high pass-through" isn't a solution. Guarded by
   `test_decoys_are_labelled_legitimate`.
3. **Keep the pot small.** `pot_median = ₹350k` over ~26 mules ≈ ₹13k each, inside
   the normal amount distribution. Raising it re-introduces a trivial amount rule.
4. **Keep rent payments.** 55% of people send 8–10k monthly. This is what stops
   `frac_near_threshold_in` from being a giveaway — and it is true of real UPI.
5. **Splits stay ring-disjoint.** Guarded by `test_splits_keep_rings_whole`.
6. **Tune on validation, report on test.** `best_threshold` must never see test.
7. **Never feed `ring_role`, `ring_tactic`, `ring_id`, or `decoy_type` to a model.**
   Label leakage; instant disqualification if a judge spots it.
8. **Both baselines and the GNN see the same feature table.**

If you deliberately change the difficulty, say so in the commit and re-run the
baselines in the same commit so the comparison never goes stale.

---

## 7. What's left

### Day 3 — the graph model  *(owner: ML)*

New file `untangle/gnn.py`. Hand-rolled in plain PyTorch — **we are deliberately
not using PyTorch Geometric.** It is the biggest install risk on Windows, and
sparse matmul gets us the same layers in ~80 lines.

- Build a sparse normalised adjacency from `transactions` (both directions, plus
  self-loops). Aggregate parallel edges; consider log-amount edge weights.
- `GraphSAGE` layer: `H' = act(W_self @ H + W_neigh @ (Â @ H))`. Two or three
  layers, hidden 64, dropout, `BatchNorm`.
- `GAT` layer: attention over neighbours. On a graph with five-figure-degree
  merchant hubs, compute attention edge-wise with a segment softmax
  (`scatter` via `index_add_`) — do **not** materialise a dense N×N matrix.
- Train transductively: full-graph forward, loss masked to `masks['train']`,
  early-stop on validation PR-AUC. Class imbalance is 1.5% — use `pos_weight`
  in `BCEWithLogitsLoss`.
- Wrap as `GNNModel` with the same `fit/score` interface as the baselines, and
  add it to the loop in `cli.py::cmd_bench`. That's the only wiring needed.
- Report **both** GraphSAGE and GAT separately in the table. Slide 5 promises
  both, and having two graph models makes the ablation credible.

Then run the ablation that makes the result defensible: same features, graph
model with message passing **switched off** (identity adjacency). If that
degrades to roughly GBDT, the lift is proven to come from the graph.

### Day 4 — ring extraction, API, visualisation

**Ring extraction** *(ML/backend)* — new `untangle/rings.py`. Scored nodes aren't
a ring yet. Take high-scoring accounts, induce the subgraph on them plus 1-hop
neighbours, run connected components (or Louvain), and rank clusters by
`sum(score) × value moved`. Output: cluster id, member list, total exposure,
confidence. Slide 6 promises "7-account ring flagged · ₹42L exposure · 0.94
confidence" — that string has to come from this module.

**API** *(backend)* — `untangle/server.py`, FastAPI:
- `GET /accounts/{id}` → score, top reason codes, ring id if any
- `GET /rings` → ranked detected rings with exposure
- `GET /graph?ring_id=` → nodes + edges for the visualiser
- `POST /score` → score a batch of transactions against the trained model

Load the trained model once at startup. Sub-second is easy — inference is one
sparse matmul; don't rebuild features per request, cache them.

**Reason codes** *(backend)* — `RuleScorecard.explain()` already returns which
typologies fired per account. For the GNN, the cheap defensible version is GAT
attention weights: "flagged because of its flows with accounts X, Y, Z". Avoid
promising full GNNExplainer in 5 days.

**Visualisation** *(frontend)* — React + D3 force-directed graph. The demo moment
on slide 6 is the ring igniting red while legitimate accounts stay calm, so build
for that: animate transactions along edges in time order, colour nodes by score,
click a node for its reason codes. Consume `/graph` and `/rings`.

### Day 5 — polish and pitch *(product)*

- Rerun the full benchmark, freeze the numbers, update README + deck.
- **Deck corrections needed now** (slides 5 and 7 currently promise things we are
  not doing): remove **PyTorch Geometric** and **Neo4j**. We use plain PyTorch
  and in-memory sparse matrices + NetworkX. Say so — a judge who asks "which PyG
  conv did you use?" will get a much better answer if the slide never claimed it.
- Slide 7's "+35 pts recall lift" is marked illustrative. Replace with the real
  measured number once Day 3 lands, or cut it.
- Rehearse the honest answer to "isn't your data synthetic?" — the answer is
  section 1 of this doc: we built it adversarially against our own thesis, here
  are the hard negatives, and here is the ablation.

### Not started, lower priority
- Streaming/incremental scoring (deck says "streaming ingest"; we score a static
  snapshot). Either build a simple windowed re-score or soften the claim.
- Cross-institution intel and the enterprise deployment story are business-model
  slides only — no code needed for the hackathon.

---

## 8. Suggested division of labour

| role | owns | first task |
|---|---|---|
| ML / Graph AI | `gnn.py`, `rings.py`, the ablation | GraphSAGE layer + wire into `bench` |
| Backend | `server.py`, caching, reason codes | FastAPI skeleton against the *existing* GBDT so the frontend is unblocked on day 3 |
| Frontend | React + D3 viz | Build against `/graph` returning a hardcoded ring first — don't wait for the GNN |
| Product | deck, metrics, demo script | Fix slides 5 and 7 today; own the "synthetic data" answer |

The unblock order matters: backend should stand up the API against the gradient
boosting model immediately, so the frontend has a real endpoint on day 3 and the
GNN drops in behind the same interface with no frontend rework.

---

## 9. Gotchas

- `data/` is gitignored. `python -m untangle generate` rebuilds it in ~9s; it is
  deterministic, so everyone with the same seed gets a byte-identical world.
- `python -m untangle --reuse bench` skips regeneration. Global flags go **before**
  the subcommand: `python -m untangle --accounts 8000 bench`, not after.
- `pd.read_csv` needs `keep_default_na=False` for `accounts.csv` — the empty
  string in `decoy_type` / `ring_role` is meaningful and otherwise becomes NaN.
  `cli.py` already does this; copy it in any new loader.
- Feature extraction is ~2s. If it takes 90s you've reintroduced a networkx call
  on the full graph — don't.
- Small configs are noisy: at `--rings 12` the test split holds only 2 rings.
  Use the defaults for any number that goes on a slide.
- Sorting: `transactions` is sorted by `t_hours` and `tx_id` is reassigned after
  sorting, so `tx_id` is a row index, not a creation order.
