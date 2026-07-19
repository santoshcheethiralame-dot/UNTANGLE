"""Knobs for the synthetic payment-graph generator.

Everything the world-builder needs is here so an experiment is one dataclass
away from reproducible. Amounts are in rupees, times in hours from t=0.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GraphConfig:
    # --- scale ---
    n_accounts: int = 40_000
    n_days: int = 30
    seed: int = 7

    # --- population mix ---
    merchant_frac: float = 0.035
    business_frac: float = 0.055
    n_communities: int = 60
    n_regions: int = 12

    # --- legitimate traffic volume ---
    p2p_per_person_per_day: float = 0.22
    spend_per_person_per_day: float = 0.55
    salary_per_month: int = 1  # inflows per personal account per month
    rent_per_month: float = 0.55  # share of people paying a monthly rent-sized sum

    # --- amounts ---
    reporting_threshold: float = 10_000.0  # the number mules structure under
    p2p_amount_median: float = 900.0
    spend_amount_median: float = 480.0
    salary_amount_median: float = 34_000.0

    # --- laundering rings ---
    n_rings: int = 26
    ring_size_min: int = 8
    ring_size_max: int = 45
    # tactic mix must sum to 1.0 -- see generate.RING_TACTICS
    tactic_weights: tuple = (0.45, 0.30, 0.25)  # structuring, smurf_wide, slow_layer
    ring_layer_hops: int = 2
    # Mule sourcing. Fresh accounts are recruited and opened for the job -- young,
    # thin KYC, handsets shared across the ring. Rented accounts are real, aged,
    # fully-KYC'd accounts with genuine history whose owner sells access. Rented
    # mules are invisible to any profile rule, which is the whole point of the mix.
    mule_fresh_frac: float = 0.35
    mule_cover_traffic: float = 0.85  # fraction of mules that also make normal-looking spend
    # Size of the laundered pot. Deliberately modest: a ring that pushes lakhs
    # through each mule is caught by a single-account amount rule and proves
    # nothing. Real operations size each hop to disappear into normal traffic.
    pot_median: float = 350_000.0

    # --- hard negatives: legitimate structures that *look* like rings ---
    n_payroll: int = 60  # employer fans out to staff
    n_settlement: int = 120  # many customers fan into a merchant, merchant sweeps out
    n_chitfund: int = 80  # rotating savings circles -- genuine cycles
    n_remitter: int = 220  # licensed money-transfer agents -- genuine high pass-through

    def evolve(self, **kw) -> "GraphConfig":
        return replace(self, **kw)


@dataclass(frozen=True)
class SplitConfig:
    """Accounts are split by community so a ring never straddles train/test."""

    val_frac: float = 0.20
    test_frac: float = 0.20
    seed: int = 11
