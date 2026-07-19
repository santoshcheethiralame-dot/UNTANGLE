# Untangle

Graph-native detection of coordinated mule-account fraud rings in UPI-style
payment data.

Fraud isn't a transaction, it's a network. A payment engine that scores each
transfer in isolation cannot see a laundering ring, because no single hop in a
ring looks abnormal — that is the entire design of the crime. Untangle models the
payment system as one graph and asks a different question: *who is this account
moving money with, and does that pattern look coordinated?*

**Status: Day 1–3 complete.** Synthetic world, feature extraction, baselines,
evaluation harness, and the graph models (GraphSAGE + GAT) are built and
benchmarked. Ring extraction, the API and the visualiser are next.

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

**3. The population churns.** 12% of customers open their account *during* the
observation window, activity is heavy-tailed, and 8% of accounts are legitimately
dormant. Without this, every young sparse account in the world is a mule, and a
graph model scores a perfect 1.000 by learning to spot recently-created nodes
rather than by understanding laundering. (It did exactly that, once. See below.)

**4. Mules are embedded in the ordinary economy.** They shop, they send peer
transfers, and — critically — they *receive* peer transfers from ordinary people.
A mule that only pays merchants shares no edge with any normal customer, which
leaves its ring sitting in the graph as an isolated island that any clustering
algorithm finds for free.

**5. The graph is salted with hard negatives** — legitimate structures that
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

## Results

40,230 accounts · 1.15M transactions · 26 rings · 1.6% mule prevalence.
Ring-disjoint split, threshold tuned on validation, scored on a held-out test set
of 7,988 accounts whose 5 rings were never seen in training.

| model | precision | recall | F1 | PR-AUC | rings found | members recovered | P@100 | alerts |
|---|---|---|---|---|---|---|---|---|
| rule scorecard | 0.048 | 0.210 | 0.078 | 0.040 | 100% | 24% | 0.080 | 693 |
| gradient boosting | 0.763 | 0.452 | 0.568 | 0.637 | 100% | 49% | 0.730 | 93 |
| MLP *(ablation)* | 0.683 | 0.535 | 0.600 | 0.637 | 100% | 58% | 0.750 | 123 |
| **GraphSAGE** | **0.966** | **0.917** | **0.941** | **0.987** | 100% | **91%** | **1.000** | 149 |
| GAT | 0.961 | 0.465 | 0.627 | 0.912 | 80% | 49% | 0.950 | 76 |

The MLP row is the experiment, not filler. It is the same network, the same 30
features, the same training budget — with message passing switched off. Two
tests enforce that it is a real control: rewiring the graph must leave the MLP's
predictions *identical* and must change GraphSAGE's.

| | fresh mules | **rented mules** | ring members recovered |
|---|---|---|---|
| rule scorecard | 0.11 | 0.26 | 24% |
| gradient boosting | 0.35 | 0.49 | 49% |
| MLP (no message passing) | 0.67 | 0.44 | 58% |
| **GraphSAGE** | **0.98** | **0.88** | **91%** |

Rented mules — real accounts, years old, full KYC, genuine history, invisible on
every profile feature — go from **0.44 to 0.88 recall** when the model is allowed
to look at neighbours. Ring member recovery goes from 49% to 91%: the difference
between alerting on half a laundering network and dismantling it.

Nothing about the account changed. The only new information is who it transacts
with. That is the entire thesis, measured.

### Two honest caveats

**GAT currently underperforms GraphSAGE and we have not fixed it.** Its *ranking*
is fine (PR-AUC 0.912, ROC-AUC 0.998) but the F1-optimal threshold chosen on
validation transfers badly to test, so it fires on only 76 accounts and recall
collapses to 0.465. This is a threshold-selection problem, not a model failure —
a fixed alert-budget cut-off (top-K, which is how banks actually staff
investigations) would likely fix it. It is on the Day 4 list.

**These numbers come from synthetic data with a known generative process,** so a
graph model that recovers most of it is expected. The absolute PR-AUC is not the
claim. The claim is the *ablation gap* — 0.637 → 0.987 on identical features —
and the fact that the gap concentrates precisely on the accounts a row-wise model
provably cannot see.

The rule scorecard's 0.048 precision is not a strawman either. It is what a
threshold engine does at 1.6% prevalence against a graph seeded with legitimate
look-alikes: it finds every ring but buries the investigator in 693 alerts.

---

## Layout

```
untangle/
  config.py     generator + split knobs, all in one dataclass
  generate.py   the synthetic world: legit traffic, rings, hard negatives
  features.py   30 per-account features in 6 families
  splits.py     ring-disjoint train/val/test masks
  baseline.py   rule scorecard + gradient boosting
  gnn.py        GraphSAGE, GAT and the no-message-passing ablation
  evaluate.py   node, ring and investigator-queue metrics
  cli.py        python -m untangle generate | bench
```

Every model consumes *exactly* the same feature table, so any lift the graph
models show comes from message passing and nothing else.

## Run it

```bash
pip install -r requirements.txt

python -m untangle generate          # writes data/accounts.csv, data/transactions.csv
python -m untangle bench             # generate, then score every detector
python -m untangle --reuse bench     # re-score the data already on disk
python -m untangle --no-gnn bench    # baselines only, seconds instead of minutes
python -m untangle --accounts 8000 --rings 12 bench   # a quick one

pytest tests -q
```

On 40k accounts: ~13s to generate, ~2.5s to build features, ~2s for the
baselines. GraphSAGE trains in ~7 min on CPU and GAT in ~12; use `--no-gnn`
while iterating on anything upstream of the models.

## Next

- [ ] Fix GAT's threshold transfer (fixed alert budget rather than F1-optimal)
- [ ] Ring extraction: from scored nodes to a highlighted cluster
- [ ] FastAPI scoring endpoint + reason codes from GAT attention
- [ ] Force-directed live visualisation

See [HANDOFF.md](HANDOFF.md) for the full build state, the invariants that keep
the benchmark honest, and who owns what.
