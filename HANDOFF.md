# Untangle — build handoff

Everything a teammate needs to pick this up cold. Read the "Don't break this"
section before you touch `generate.py`.

- **Built:** Day 1 (graph engine), Day 2 (features, baselines, eval), Day 3 (GNN).
- **Left:** Day 4 (ring extraction, API, viz), Day 5 (polish, pitch).
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

That claim is falsifiable, and it has been falsified twice during this build.

**Failure 1 — the row-wise model solved it.** A plain gradient-boosted tree over
30 per-account columns scored **0.987 F1** with no graph at all. If that had
shipped, the GNN would have been decoration. Fixed by making half of every ring
rented accounts, shrinking the laundered pot, adding tactic variety, and salting
the graph with legitimate look-alikes.

**Failure 2 — the graph model solved it for the wrong reason.** GraphSAGE then
scored a *perfect* **1.000 PR-AUC**. The cause was in the population, not the
model: every account existed from t=0 with full history, so freshly-created mule
accounts were unique in the world — degree 9 against a population average of 37,
and **0.08%** of ordinary accounts had ever touched one. The model was detecting
recently-created nodes, i.e. our own injection code. Fixed by adding customer
onboarding, dormancy, heavy-tailed activity, and by giving mules ordinary lives
(including *incoming* peer transfers, without which a ring is an isolated island
that any clustering algorithm finds for free). Ordinary accounts touching a fresh
mule went 0.08% → 5.7%, and fresh-mule degree 9.4 → 27.0.

The lesson both times: **when a number looks great, attack it.** The benchmark is
the product here. A model that wins on a broken world proves nothing, and the
failure mode is always silent.

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
    def fit(self, X, y, mask, val_mask=None) -> "Model": ...
    def score(self, X) -> np.ndarray:  # one score per account, all N
```

`val_mask` is optional and ignored by both baselines; the GNN uses it for early
stopping. Any new model must accept it even if it does nothing with it.

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

## 5. Current results

40,230 accounts · 1.15M transactions · 26 rings · 1.6% prevalence.
Test set: 7,988 accounts, 157 mules, 5 rings never seen in training.

| model | P | R | F1 | PR-AUC | rings found | members recovered | P@100 | alerts |
|---|---|---|---|---|---|---|---|---|
| rules | 0.048 | 0.210 | 0.078 | 0.040 | 100% | 24% | 0.080 | 693 |
| gbdt | 0.763 | 0.452 | 0.568 | 0.637 | 100% | 49% | 0.730 | 93 |
| mlp *(ablation)* | 0.683 | 0.535 | 0.600 | 0.637 | 100% | 58% | 0.750 | 123 |
| **sage** | 0.966 | 0.917 | 0.941 | 0.987 | 100% | 91% | 1.000 | 149 |
| gat | 0.961 | 0.465 | 0.627 | 0.912 | 80% | 49% | 0.950 | 76 |

Recall split by how the mule was recruited:

| | fresh mules | **rented mules** | members recovered |
|---|---|---|---|
| rules | 0.11 | 0.26 | 24% |
| gbdt | 0.35 | 0.49 | 49% |
| mlp | 0.67 | 0.44 | 58% |
| **sage** | **0.98** | **0.88** | **91%** |

**This is the whole story.** Rented mules — indistinguishable from ordinary
customers on every profile feature — go from 0.44 recall without message passing
to 0.88 with it, on identical inputs. Ring member recovery goes 49% → 91%.

Ring *detection* rate is 100% for nearly everyone; it is not a differentiator at
this ring size. Don't put it on a slide as a win.

**Known problem: GAT's threshold does not transfer.** Its ranking is sound
(PR-AUC 0.912, ROC-AUC 0.998) but the F1-optimal cut-off picked on validation
leaves it firing on 76 accounts, dropping recall to 0.465 and ring detection to
80%. Almost certainly fixable by selecting the threshold at a fixed alert budget
(top-K) instead of by F1 — which is also how banks actually staff investigations,
so it is worth doing as an evaluation-protocol change rather than a GAT hack. See
Day 4 below.

---

## 6. Don't break this (invariants)

These are what make the benchmark meaningful. Several have tests; all of them
have a reason.

1. **Half the ring must stay rented.** `mule_fresh_frac = 0.35`. Rented mules are
   real accounts with genuine history, statistically identical to ordinary
   customers on every profile feature. Raising this makes the problem easy and
   the GNN's lift fake. Guarded by `test_rented_mules_look_ordinary_on_profile`.
1b. **Keep customer churn and mule cover traffic.** `newcomer_frac = 0.12`,
   `dormant_frac = 0.08`, `mule_cover_traffic = 0.85`, and `_ordinary_life` must
   keep generating *incoming* peer transfers for mules. Remove any of these and
   fresh mules become beacons again — this is exactly what produced a fake
   1.000 PR-AUC. Sanity check after any generator change:

   ```python
   # ordinary accounts with a fresh-mule neighbour should be ~5%, never ~0%
   # fresh-mule mean degree should be within ~1.5x of legit newcomers
   ```
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

### Day 3 — the graph model  *(done)*

`untangle/gnn.py`. Hand-rolled in plain PyTorch — **deliberately not PyTorch
Geometric**, which is the biggest install risk on Windows for two layers we can
write ourselves.

- `SAGELayer`: `h' = W_self·h + W_neigh·mean(neighbours)`, via a row-normalised
  sparse adjacency built once in `mean_adjacency`.
- `GATLayer`: multi-head attention with an edge-wise segment softmax. It
  aggregates **one head at a time** through `index_add_`. Doing all heads at once
  gathers `E × heads × d_head` (~180M floats, retained through backward); a
  sparse-matmul version avoids that but benchmarked *3x slower* because
  `torch.sparse.mm` re-sorts indices on every call. Don't "optimise" this back
  without measuring.
- `MLPLayer`: the ablation. Same training loop, never reads a neighbour.
- Training is transductive and full-batch, `pos_weight` for the 1.6% imbalance,
  early stopping on validation **PR-AUC** (not accuracy — at this prevalence
  accuracy is maximised by predicting nobody).
- `GraphModel.edge_attention()` returns per-edge attention from the first GAT
  layer. **This is the input to reason codes on Day 4.**

`--arch` is not implemented; `--no-gnn` skips all three graph models. If you want
to iterate without waiting ~19 min for SAGE + GAT, add an arch filter to
`cli.cmd_bench` — it's a two-line change.

### Day 4 — ring extraction, API, visualisation

**Fix GAT's threshold first** *(ML, ~1 hour, highest value-per-minute on the
board)*. Add a fixed-alert-budget threshold to `evaluate.py` alongside
`best_threshold` — pick the cut-off that flags the top K accounts on validation,
where K matches a plausible investigator capacity. Report both. This likely turns
GAT from 0.465 recall into something competitive with SAGE, and it is a more
honest evaluation protocol for every model, not a GAT-specific patch.

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

**Reason codes** *(backend)* — `RuleScorecard.explain()` returns which typologies
fired per account, and `GraphModel.edge_attention()` (arch='gat') returns
per-edge attention, already normalised to sum to 1 per receiving node. Combine
them: "flagged because of its flows with accounts X, Y, Z, and because it passed
through 94% of what it received within two hours". Avoid promising full
GNNExplainer in 5 days.

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
| ML / Graph AI | `gnn.py`, `rings.py`, the ablation | fixed-alert-budget threshold, then `rings.py` |
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
- `accounts.csv` has a `joined_at_hours` column. Any transaction generated before
  either party's join time is dropped in `generate()`, and an assert fires if a
  *ring* transfer is ever dropped — that assert has already caught two real bugs,
  so if it trips, something upstream is genuinely wrong. Don't relax it.
- Training a graph model takes ~7 min (SAGE) / ~12 min (GAT) on 40k accounts.
  Use `--no-gnn`, or a smaller `--accounts`, while working on anything upstream.
- Small configs are noisy: at `--rings 12` the test split holds only 2 rings.
  Use the defaults for any number that goes on a slide.
- Sorting: `transactions` is sorted by `t_hours` and `tx_id` is reassigned after
  sorting, so `tx_id` is a row index, not a creation order.
