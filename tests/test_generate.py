import numpy as np
import pytest

from untangle.config import GraphConfig
from untangle.generate import generate

SMALL = GraphConfig(n_accounts=3000, n_rings=6, n_remitter=20, n_settlement=10,
                    n_chitfund=8, n_payroll=6)


@pytest.fixture(scope="module")
def world():
    return generate(SMALL)


def test_deterministic():
    a1, t1 = generate(SMALL)
    a2, t2 = generate(SMALL)
    assert a1.equals(a2)
    assert t1.equals(t2)


def test_no_self_transfers(world):
    _, tx = world
    assert (tx["src"] != tx["dst"]).all()


def test_ids_and_times_in_range(world):
    accounts, tx = world
    n = len(accounts)
    assert accounts["account_id"].tolist() == list(range(n))
    assert tx["src"].between(0, n - 1).all()
    assert tx["dst"].between(0, n - 1).all()
    assert tx["t_hours"].between(0, SMALL.n_days * 24).all()
    assert (tx["amount"] > 0).all()


def test_every_ring_is_populated(world):
    accounts, _ = world
    mules = accounts[accounts.is_mule == 1]
    assert mules["ring_id"].nunique() == SMALL.n_rings
    assert mules.groupby("ring_id").size().min() >= SMALL.ring_size_min - 2


def test_rings_use_both_sourcing_routes(world):
    """Rented mules are the hard half of the problem -- if the mix collapses to
    all-fresh, every downstream benchmark becomes trivially easy."""
    accounts, _ = world
    roles = set(accounts.loc[accounts.is_mule == 1, "ring_role"])
    assert {"mule_fresh", "mule_rented"} <= roles


def test_decoys_are_labelled_legitimate(world):
    """The hard negatives must never be labelled as mules -- they are the whole
    reason the benchmark is not a pushover."""
    accounts, _ = world
    decoys = accounts[accounts.decoy_type != ""]
    assert len(decoys) > 0
    assert (decoys["is_mule"] == 0).all()


def test_rented_mules_look_ordinary_on_profile(world):
    """A rented mule must be indistinguishable from a normal customer on the
    static profile fields. If this drifts, profile rules start 'working' for the
    wrong reason and the graph model's lift becomes meaningless."""
    accounts, _ = world
    rented = accounts[accounts.ring_role == "mule_rented"]
    legit = accounts[(accounts.is_mule == 0) & (accounts.kind == "personal")]
    assert abs(rented["age_days"].median() - legit["age_days"].median()) < 200
    assert rented["kyc_level"].mean() > legit["kyc_level"].mean() - 0.2
    assert rented["device_share_count"].mean() < legit["device_share_count"].mean() + 0.5


def test_each_account_belongs_to_at_most_one_ring(world):
    """A ring must never recruit an account another ring already owns.

    If it does, that account's ring_id is overwritten -- which moves it into a
    different train/test split while its actual transaction structure still
    belongs to the original ring. That is a silent cross-split leak, and it
    inflates every graph-model score.
    """
    accounts, _ = world
    claimed = accounts[accounts.ring_id >= 0]
    assert (claimed["ring_role"] != "").all()
    # a source is a victim, never a mule; a mule is never someone else's source
    assert (accounts.loc[accounts.ring_role == "source", "is_mule"] == 0).all()
    assert (accounts.loc[accounts.is_mule == 1, "ring_role"] != "source").all()


def test_prevalence_is_realistic(world):
    accounts, _ = world
    assert 0.005 < accounts["is_mule"].mean() < 0.08
