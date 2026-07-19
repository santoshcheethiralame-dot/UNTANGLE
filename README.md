# Untangle

Graph-native detection of coordinated mule-account fraud rings in UPI-style
payment data.

Fraud isn't a transaction, it's a network. A payment engine that scores each
transfer in isolation cannot see a laundering ring, because no single hop in a
ring looks abnormal — that is the entire design of the crime. Untangle models the
payment system as one graph and asks a different question: *who is this account
moving money with, and does that pattern look coordinated?*

**Status: Day 1–2 complete.** Synthetic world, feature extraction, two baselines,
and the evaluation harness are built and benchmarked. The graph model is next.

---

## Why the benchmark is the hard part

It is easy to build a synthetic fraud dataset that a graph model wins on. It is
also worthless — if the mules are all brand-new accounts churning money at
10× normal velocity, a decision tree on five columns finds them, and the graph
never earned its place.

So the generator is built adversarially against its own thesis. Three choices do
that work:

**1. Half the ring is rented, not fresh.** Fresh mules are opened for the job:
days old, thin KYC, handsets shared across the ring. Rented mules are ordinary
accounts — years old, full KYC, real salary and shopping history — whose owner
sold access for a cut. Account rental is the dominant modality in real mule
networks, and a rented mule is *statistically identical to a normal customer on
every profile feature*. The tests assert this stays true:

| feature | rented mule | ordinary customer |
|---|---|---|
| account age (log) | 6.82 | 6.86 |
| KYC level | 1.83 | 1.83 |
| devices shared | 1.03 | 1.03 |
| active span (h) | 690 | 671 |
| k-core | 28.9 | 27.9 |

**2. Rings use several tactics.** A *structuring* ring splits money into
sub-threshold hops over a day; *smurf_wide* spreads it thin across many mules;
*slow_layer* moves ordinary-looking amounts over nine days. A detector tuned to
one tactic misses the others.

**3. The graph is salted with hard negatives** — legitimate structures that
reproduce a laundering signature exactly:

| decoy | why it fools a rule engine |
|---|---|
| payroll fan-out | employer pays 40–120 staff near-identical amounts in one window |
| settlement merchant | hundreds of customers fan in, merchant sweeps ~all of it out the same evening — high fan-in, ~1.0 pass-through, fast turnaround |
| chit fund | rotating savings circle: genuine dense cycles between the same members |
| licensed remitter | takes cash-in from strangers, forwards to strangers, keeps nothing |
| rent payments | 55% of people send a rent-sized sum in the 8–10k band monthly, which is what makes threshold structuring viable at all |

The money moved per mule is deliberately modest (~₹13k, inside the normal
distribution) rather than lakhs. A ring that pushes lakhs through each account is
caught by a single amount rule and proves nothing.

---

## Results so far

40,210 accounts · 1.4M transactions · 26 rings · 1.5% mule prevalence.
Ring-disjoint split, threshold tuned on validation, scored on a held-out test set
whose rings were never seen in training.

| model | precision | recall | F1 | PR-AUC | rings found | members recovered | P@100 |
|---|---|---|---|---|---|---|---|
| rule scorecard | 0.039 | 0.210 | 0.066 | 0.033 | 100% | 18% | 0.080 |
| gradient boosting | 0.918 | 0.471 | 0.622 | 0.660 | 100% | 45% | 0.680 |

The number that matters is not F1. It's this:

| | fresh mules | **rented mules** |
|---|---|---|
| rule scorecard | 0.05 | 0.26 |
| gradient boosting | **1.00** | **0.18** |

Gradient boosting catches *every single* disposable mule and **18% of the rented
ones** — the half of the ring with clean profiles and real history. It recovers
45% of the average ring's members, so the majority of every ring keeps operating.

That gap is the thesis, stated as a measurement: the accounts a row-wise model
cannot see are exactly the accounts that are only identifiable by who they
transact with. Closing it is what the graph model has to do, and "beat 0.660
PR-AUC / 45% member recovery" is the bar it will be held to.

Note the rule scorecard's precision of 0.039 is not a strawman — it is what a
threshold-based engine does at 1.5% prevalence against a graph seeded with
legitimate look-alikes. It finds every ring but drowns the investigator in 644
alerts to surface 25 mules.

---

## Layout

```
untangle/
  config.py     generator + split knobs, all in one dataclass
  generate.py   the synthetic world: legit traffic, rings, hard negatives
  features.py   30 per-account features in 6 families
  splits.py     ring-disjoint train/val/test masks
  baseline.py   rule scorecard + gradient boosting
  evaluate.py   node, ring and investigator-queue metrics
  cli.py        python -m untangle generate | bench
```

Both baselines consume *exactly* the same feature table, so any lift the graph
model shows comes from message passing and nothing else.

## Run it

```bash
pip install -r requirements.txt

python -m untangle generate          # writes data/accounts.csv, data/transactions.csv
python -m untangle bench             # generate, then score every detector
python -m untangle --reuse bench     # re-score the data already on disk
python -m untangle --accounts 8000 --rings 12 bench   # a quick one

pytest tests -q
```

Full pipeline on 40k accounts: ~9s to generate, ~2s to build features, a few
seconds to benchmark.

## Next

- [ ] GraphSAGE + GAT over the transaction graph (hand-rolled in PyTorch — no
      PyTorch Geometric install roulette)
- [ ] Ring extraction: from scored nodes to a highlighted cluster
- [ ] FastAPI scoring endpoint + reason codes
- [ ] Force-directed live visualisation
