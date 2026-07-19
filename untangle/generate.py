"""Synthetic UPI-style payment graph with injected laundering rings.

The point of this generator is not to make fraud *findable* -- it is to make it
findable only by something that looks at structure. Three design choices do that
work:

1.  Rings use several tactics. A structuring ring splits money into sub-threshold
    hops within hours; a slow-layering ring moves ordinary-looking amounts over
    a week. A detector tuned to one tactic misses the other.
2.  Mules carry cover traffic -- ordinary spend that makes their account history
    look lived-in.
3.  The graph is salted with *hard negatives*: payroll fan-outs, merchant
    settlement fan-ins, chit-fund cycles and licensed remitters. Each reproduces
    a structural signature of laundering while being entirely legitimate. They
    are what separates a model that learned the crime from one that learned
    "high pass-through = bad".

Output is two tables:

    accounts      one row per account, with the ground-truth label
    transactions  one row per transfer, directed src -> dst
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GraphConfig

RING_TACTICS = ("structuring", "smurf_wide", "slow_layer")

PERSONAL, BUSINESS, MERCHANT = "personal", "business", "merchant"

_ACCOUNT_COLS = [
    "account_id",
    "kind",
    "age_days",
    "kyc_level",
    "device_id",
    "community",
    "region",
    "is_mule",
    "ring_id",
    "ring_role",
    "ring_tactic",
    "decoy_type",
]

_TX_COLS = ["tx_id", "src", "dst", "amount", "t_hours", "channel", "ring_id"]


class _World:
    """Mutable scratch space while the graph is being assembled."""

    def __init__(self, cfg: GraphConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.accounts: list[dict] = []
        self.tx: list[tuple] = []
        self.horizon = cfg.n_days * 24.0

    # -- accounts -------------------------------------------------------

    def add_account(self, **kw) -> int:
        aid = len(self.accounts)
        row = {
            "account_id": aid,
            "kind": PERSONAL,
            "age_days": 900,
            "kyc_level": 2,
            "device_id": -1,
            "community": 0,
            "region": 0,
            "is_mule": 0,
            "ring_id": -1,
            "ring_role": "",
            "ring_tactic": "",
            "decoy_type": "",
        }
        row.update(kw)
        self.accounts.append(row)
        return aid

    # -- transactions ---------------------------------------------------

    def add_tx(self, src: int, dst: int, amount: float, t: float, channel: str, ring_id: int = -1):
        if src == dst:
            return
        self.tx.append((len(self.tx), int(src), int(dst), round(float(amount), 2),
                        float(np.clip(t, 0.0, self.horizon)), channel, int(ring_id)))

    # -- helpers --------------------------------------------------------

    def lognormal(self, median: float, sigma: float, size=None):
        return np.exp(self.rng.normal(np.log(median), sigma, size))


def _channel_for(amount: float, rng: np.random.Generator) -> str:
    """UPI dominates small-ticket; big transfers fall back to IMPS/NEFT."""
    if amount <= 100_000:
        return "UPI" if rng.random() < 0.93 else "IMPS"
    return "IMPS" if rng.random() < 0.6 else "NEFT"


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------


def _build_population(w: _World) -> dict:
    cfg, rng = w.cfg, w.rng
    n = cfg.n_accounts
    n_merch = int(n * cfg.merchant_frac)
    n_biz = int(n * cfg.business_frac)
    n_person = n - n_merch - n_biz

    for i in range(n):
        if i < n_person:
            kind = PERSONAL
            age = int(np.clip(rng.gamma(4.0, 260.0), 20, 6000))
            kyc = int(rng.choice([1, 2], p=[0.18, 0.82]))
        elif i < n_person + n_biz:
            kind = BUSINESS
            age = int(np.clip(rng.gamma(6.0, 300.0), 90, 8000))
            kyc = 2
        else:
            kind = MERCHANT
            age = int(np.clip(rng.gamma(5.0, 320.0), 60, 8000))
            kyc = 2
        w.add_account(
            kind=kind,
            age_days=age,
            kyc_level=kyc,
            device_id=i,  # honest accounts overwhelmingly own their device
            community=int(rng.integers(cfg.n_communities)),
            region=int(rng.integers(cfg.n_regions)),
        )

    # a small, realistic amount of benign device sharing (families, shared phones)
    shared = rng.choice(n_person, size=int(n_person * 0.03), replace=False)
    for a, b in zip(shared[::2], shared[1::2]):
        w.accounts[b]["device_id"] = w.accounts[a]["device_id"]

    people = np.arange(n_person)
    businesses = np.arange(n_person, n_person + n_biz)
    merchants = np.arange(n_person + n_biz, n)

    # merchant popularity is Zipf-ish: a few hubs take most of the spend
    pop = 1.0 / (1.0 + np.arange(len(merchants)))
    rng.shuffle(pop)
    merchant_p = pop / pop.sum()

    return {
        "people": people,
        "businesses": businesses,
        "merchants": merchants,
        "merchant_p": merchant_p,
        "n_person": n_person,
    }


# ---------------------------------------------------------------------------
# legitimate traffic
# ---------------------------------------------------------------------------


def _legit_salary(w: _World, pop: dict):
    """Monthly inflow from an employer. Gives most accounts a big regular credit."""
    cfg, rng = w.cfg, w.rng
    people, businesses = pop["people"], pop["businesses"]
    employer = rng.choice(businesses, size=len(people))
    n_pay = max(1, int(cfg.n_days / 30.0 * cfg.salary_per_month))
    for cycle in range(n_pay):
        base = cycle * 30.0 * 24.0 + 24.0
        amts = w.lognormal(cfg.salary_amount_median, 0.55, len(people))
        jitter = rng.normal(0, 4.0, len(people))
        for p, e, a, j in zip(people, employer, amts, jitter):
            t = base + j
            if 0 <= t <= w.horizon:
                w.add_tx(e, p, a, t, "NEFT")


def _legit_spend(w: _World, pop: dict):
    """Consumer -> merchant payments, the bulk of the graph's edges."""
    cfg, rng = w.cfg, w.rng
    people, merchants = pop["people"], pop["merchants"]
    n_tx = int(len(people) * cfg.n_days * cfg.spend_per_person_per_day)
    src = rng.choice(people, size=n_tx)
    dst = rng.choice(merchants, size=n_tx, p=pop["merchant_p"])
    amt = w.lognormal(cfg.spend_amount_median, 1.0, n_tx)
    # daytime-weighted timestamps
    day = rng.integers(0, cfg.n_days, n_tx)
    hour = np.clip(rng.normal(14.0, 4.0, n_tx), 0, 23.99)
    for s, d, a, dd, hh in zip(src, dst, amt, day, hour):
        w.add_tx(s, d, a, dd * 24.0 + hh, _channel_for(a, rng))


def _legit_p2p(w: _World, pop: dict):
    """Peer transfers, mostly inside a community -- this is what gives the graph
    its clustering, and what a naive 'unusual neighbour' rule trips over."""
    cfg, rng = w.cfg, w.rng
    people = pop["people"]
    comm = np.array([w.accounts[i]["community"] for i in people])
    by_comm = {c: people[comm == c] for c in np.unique(comm)}

    n_tx = int(len(people) * cfg.n_days * cfg.p2p_per_person_per_day)
    src = rng.choice(people, size=n_tx)
    dst = np.empty(n_tx, dtype=np.int64)
    same = rng.random(n_tx) < 0.75
    for i, (s, keep_local) in enumerate(zip(src, same)):
        peers = by_comm[w.accounts[s]["community"]] if keep_local else people
        dst[i] = peers[rng.integers(len(peers))]
    amt = w.lognormal(cfg.p2p_amount_median, 1.15, n_tx)
    t = rng.random(n_tx) * w.horizon
    for s, d, a, tt in zip(src, dst, amt, t):
        w.add_tx(s, d, a, tt, _channel_for(a, rng))


def _legit_rent(w: _World, pop: dict):
    """Monthly rent, paid person-to-person, in the 8-10k band.

    This is the flow that makes threshold structuring viable in the first place:
    sub-10k transfers are the most ordinary thing in the payment system, so an
    amount rule keyed to that band drowns in honest landlords."""
    cfg, rng = w.cfg, w.rng
    people = pop["people"]
    payers = rng.choice(people, size=int(len(people) * cfg.rent_per_month), replace=False)
    landlord = rng.choice(people, size=len(payers))
    for cycle in range(max(1, cfg.n_days // 30)):
        base = cycle * 30.0 * 24.0 + 6.0
        for p, l in zip(payers, landlord):
            amt = float(rng.uniform(0.70, 1.02) * cfg.reporting_threshold)
            w.add_tx(p, l, amt, base + rng.uniform(0, 72), "UPI")


def _legit_settlement_sweep(w: _World, pop: dict):
    """Merchants sweep takings to a business account -- legitimate pass-through."""
    rng = w.rng
    merchants, businesses = pop["merchants"], pop["businesses"]
    bank = rng.choice(businesses, size=len(merchants))
    for m, b in zip(merchants, businesses[rng.integers(0, len(businesses), len(merchants))]):
        for day in range(0, w.cfg.n_days, 2):
            amt = w.lognormal(28_000.0, 0.9)
            w.add_tx(m, b, amt, day * 24.0 + 23.0 + rng.normal(0, 0.5), "NEFT")


# ---------------------------------------------------------------------------
# hard negatives -- legitimate structures that mimic laundering
# ---------------------------------------------------------------------------


def _decoy_payroll(w: _World, pop: dict):
    """Employer fans out near-identical amounts to many accounts in one window.
    Structurally: a fan-out burst, exactly like the placement stage of a ring."""
    rng = w.rng
    for _ in range(w.cfg.n_payroll):
        employer = int(rng.choice(pop["businesses"]))
        staff = rng.choice(pop["people"], size=int(rng.integers(40, 120)), replace=False)
        w.accounts[employer]["decoy_type"] = "payroll_hub"
        base = float(rng.integers(1, w.cfg.n_days)) * 24.0 + 10.0
        amt = float(w.lognormal(26_000.0, 0.3))
        for s in staff:
            w.add_tx(employer, s, amt * rng.uniform(0.9, 1.1), base + rng.normal(0, 1.5), "NEFT")


def _decoy_settlement(w: _World, pop: dict):
    """Hundreds of customers pay one merchant, who forwards nearly all of it out
    the same evening. High fan-in, high pass-through, fast turnaround -- the
    textbook collector-account signature, and completely lawful."""
    rng = w.rng
    for _ in range(w.cfg.n_settlement):
        merch = int(rng.choice(pop["merchants"]))
        acquirer = int(rng.choice(pop["businesses"]))
        w.accounts[merch]["decoy_type"] = "settlement_merchant"
        for day in range(w.cfg.n_days):
            customers = rng.choice(pop["people"], size=int(rng.integers(20, 60)), replace=False)
            total = 0.0
            for c in customers:
                a = float(w.lognormal(700.0, 0.8))
                total += a
                w.add_tx(c, merch, a, day * 24.0 + rng.uniform(9, 21), "UPI")
            w.add_tx(merch, acquirer, total * rng.uniform(0.96, 0.99),
                     day * 24.0 + 22.5, "NEFT")


def _decoy_chitfund(w: _World, pop: dict):
    """Rotating savings circle: every member pays every other over time, and the
    pot rotates. Produces dense genuine cycles -- the motif cycle-detection
    heuristics flag first."""
    rng = w.rng
    for _ in range(w.cfg.n_chitfund):
        members = rng.choice(pop["people"], size=int(rng.integers(10, 22)), replace=False)
        for m in members:
            w.accounts[m]["decoy_type"] = "chitfund"
        contrib = float(w.lognormal(5_000.0, 0.3))
        rounds = max(1, w.cfg.n_days // 10)
        for r in range(rounds):
            winner = members[r % len(members)]
            t0 = r * 10.0 * 24.0 + 12.0
            for m in members:
                if m != winner:
                    w.add_tx(m, winner, contrib * rng.uniform(0.98, 1.02),
                             t0 + rng.uniform(0, 6), "UPI")


def _decoy_remitter(w: _World, pop: dict):
    """Licensed money-transfer agent: takes cash-in from many strangers and
    forwards to many strangers, keeping almost nothing. Pass-through ratio ~1.0,
    high in- and out-degree, no community loyalty. The hardest negative here."""
    rng = w.rng
    for _ in range(w.cfg.n_remitter):
        agent = int(rng.choice(pop["people"]))
        w.accounts[agent]["decoy_type"] = "remitter"
        w.accounts[agent]["kind"] = BUSINESS
        for day in range(w.cfg.n_days):
            for _ in range(int(rng.integers(6, 18))):
                sender = int(rng.choice(pop["people"]))
                recv = int(rng.choice(pop["people"]))
                amt = float(w.lognormal(6_500.0, 0.7))
                t = day * 24.0 + rng.uniform(9, 20)
                w.add_tx(sender, agent, amt, t, "UPI")
                w.add_tx(agent, recv, amt * rng.uniform(0.97, 0.995),
                         t + rng.uniform(0.1, 2.0), "UPI")


# ---------------------------------------------------------------------------
# laundering rings
# ---------------------------------------------------------------------------


def _spawn_mules(
    w: _World, pop: dict, k: int, ring_id: int, tactic: str, n_devices: int
) -> list[int]:
    """Recruit a ring's mules from two very different sources.

    *Fresh* mules are opened for the job: days old, thin KYC, a handful of
    handsets shared across the whole ring. Any profile rule catches these.

    *Rented* mules are ordinary accounts -- years old, full KYC, own device,
    a real history of salary and shopping -- whose owner has sold access for a
    cut. Nothing on the account profile distinguishes them from any other
    customer. The only thing that gives them away is who they suddenly start
    moving money with, which is precisely the signal a graph model sees and a
    row-wise model cannot.
    """
    rng = w.rng
    devices = [w.cfg.n_accounts * 10 + ring_id * 100 + d for d in range(n_devices)]
    n_fresh = int(round(k * w.cfg.mule_fresh_frac))

    mules = []
    for _ in range(n_fresh):
        # Not all "fresh" mules are new. A good recruiter keeps a stock of shell
        # accounts opened months or years earlier precisely so that an
        # account-age rule sees nothing unusual on the day they are switched on.
        if rng.random() < 0.4:
            age = int(np.clip(rng.gamma(5.0, 160.0), 240, 3000))
        else:
            age = int(np.clip(rng.gamma(2.0, 18.0), 3, 240))
        mules.append(
            w.add_account(
                kind=PERSONAL,
                age_days=age,
                kyc_level=int(rng.choice([0, 1, 2], p=[0.35, 0.5, 0.15])),
                device_id=int(rng.choice(devices)),
                community=int(rng.integers(w.cfg.n_communities)),
                region=int(rng.integers(w.cfg.n_regions)),
                is_mule=1,
                ring_id=ring_id,
                ring_role="mule_fresh",
                ring_tactic=tactic,
            )
        )

    for _ in range(k - n_fresh):
        for _attempt in range(20):
            cand = int(rng.choice(pop["people"]))
            acct = w.accounts[cand]
            if acct["is_mule"] == 0 and acct["ring_id"] == -1 and acct["decoy_type"] == "":
                acct.update(is_mule=1, ring_id=ring_id, ring_role="mule_rented",
                            ring_tactic=tactic)
                mules.append(cand)
                break
    return mules


def _ring_amounts(w: _World, tactic: str, k: int, pot: float) -> np.ndarray:
    """How the pot is sliced determines whether an amount rule can see it."""
    rng = w.rng
    thr = w.cfg.reporting_threshold
    if tactic == "structuring":
        # deliberately parked just under the reporting threshold
        return thr * rng.uniform(0.80, 0.985, k)
    if tactic == "smurf_wide":
        # spread thin across many mules; individually unremarkable
        return np.full(k, pot / k) * rng.uniform(0.6, 1.4, k)
    # slow_layer: ordinary-looking amounts, no threshold hugging at all
    return w.lognormal(pot / k, 0.5, k)


def _inject_ring(w: _World, pop: dict, ring_id: int) -> None:
    cfg, rng = w.cfg, w.rng
    tactic = str(rng.choice(RING_TACTICS, p=cfg.tactic_weights))
    k = int(rng.integers(cfg.ring_size_min, cfg.ring_size_max + 1))

    # the compromised source is an established, ordinary-looking account
    victim = int(rng.choice(pop["people"]))
    w.accounts[victim]["ring_role"] = "source"
    w.accounts[victim]["ring_id"] = ring_id

    # Handset reuse is real but far from one-phone-per-ring: operators know
    # device fingerprinting exists, so most mules get their own.
    n_devices = max(2, int(k * rng.uniform(0.45, 0.9)))
    mules = _spawn_mules(w, pop, k, ring_id, tactic, n_devices)
    collector = mules[int(rng.integers(len(mules)))]
    w.accounts[collector]["ring_role"] = "collector"

    pot = float(w.lognormal(cfg.pot_median, 0.8))
    # how long the whole operation takes -- this is the other axis rules key on
    span = {"structuring": 30.0, "smurf_wide": 48.0, "slow_layer": 9 * 24.0}[tactic]
    t0 = rng.uniform(0, max(1.0, w.horizon - span - 4.0))

    # 1. placement: source fans the pot out to the mules
    amts = _ring_amounts(w, tactic, k, pot)
    for m, a in zip(mules, amts):
        w.add_tx(victim, m, a, t0 + rng.uniform(0, span * 0.45),
                 _channel_for(a, rng), ring_id)

    # 2. layering: money hops between mules, shuffling provenance
    held = dict(zip(mules, amts))
    for hop in range(cfg.ring_layer_hops):
        order = rng.permutation(mules)
        for i, src in enumerate(order):
            dst = int(order[(i + 1 + rng.integers(0, max(1, len(order) - 1))) % len(order)])
            move = held[src] * rng.uniform(0.55, 0.9)
            if move < 100:
                continue
            held[src] -= move
            held[dst] = held.get(dst, 0.0) + move
            t = t0 + span * (0.30 + 0.18 * hop) + rng.uniform(0, span * 0.35)
            w.add_tx(src, dst, move, t, _channel_for(move, rng), ring_id)

    # 3. integration: everything converges on the collector, then exits
    for m in mules:
        if m == collector or held.get(m, 0) < 100:
            continue
        w.add_tx(m, collector, held[m] * rng.uniform(0.9, 0.99),
                 t0 + span * 0.70 + rng.uniform(0, span * 0.28),
                 _channel_for(held[m], rng), ring_id)
    exit_point = int(rng.choice(pop["merchants"]))
    w.add_tx(collector, exit_point, pot * rng.uniform(0.75, 0.9),
             t0 + span * 0.98, "IMPS", ring_id)

    # 4. cover traffic: some mules behave like ordinary customers too
    for m in mules:
        if rng.random() > cfg.mule_cover_traffic:
            continue
        for _ in range(int(rng.integers(2, 9))):
            d = int(rng.choice(pop["merchants"], p=pop["merchant_p"]))
            a = float(w.lognormal(cfg.spend_amount_median, 1.0))
            w.add_tx(m, d, a, rng.uniform(0, w.horizon), "UPI")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def generate(cfg: GraphConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the world. Returns (accounts, transactions)."""
    cfg = cfg or GraphConfig()
    rng = np.random.default_rng(cfg.seed)
    w = _World(cfg, rng)

    pop = _build_population(w)

    _legit_salary(w, pop)
    _legit_spend(w, pop)
    _legit_p2p(w, pop)
    _legit_rent(w, pop)
    _legit_settlement_sweep(w, pop)

    _decoy_payroll(w, pop)
    _decoy_settlement(w, pop)
    _decoy_chitfund(w, pop)
    _decoy_remitter(w, pop)

    for ring_id in range(cfg.n_rings):
        _inject_ring(w, pop, ring_id)

    accounts = pd.DataFrame(w.accounts, columns=_ACCOUNT_COLS)
    tx = pd.DataFrame(w.tx, columns=_TX_COLS).sort_values("t_hours", ignore_index=True)
    tx["tx_id"] = np.arange(len(tx))

    # device reuse is a feature, not a lookup -- compute it once here
    counts = accounts["device_id"].value_counts()
    accounts["device_share_count"] = accounts["device_id"].map(counts).astype(int)

    return accounts, tx


def summarize(accounts: pd.DataFrame, tx: pd.DataFrame) -> str:
    mules = int(accounts["is_mule"].sum())
    lines = [
        f"accounts      {len(accounts):>9,}   ({mules:,} mules, "
        f"{mules / len(accounts):.2%} positive rate)",
        f"transactions  {len(tx):>9,}",
        f"rings         {accounts.loc[accounts.is_mule == 1, 'ring_id'].nunique():>9,}",
        f"value moved   {tx['amount'].sum() / 1e7:>9,.1f} cr",
        "",
        "ring tactics:",
    ]
    tac = accounts[accounts.is_mule == 1].groupby("ring_tactic")["ring_id"].nunique()
    for name, n in tac.items():
        lines.append(f"  {name:<14} {n:>4} rings")
    lines.append("")
    lines.append("mule sourcing:")
    for name, n in accounts[accounts.is_mule == 1]["ring_role"].value_counts().items():
        lines.append(f"  {name:<22} {n:>6,} accounts")
    lines.append("")
    lines.append("hard negatives (legitimate, ring-shaped):")
    for name, n in accounts[accounts.decoy_type != ""]["decoy_type"].value_counts().items():
        lines.append(f"  {name:<22} {n:>6,} accounts")
    return "\n".join(lines)
