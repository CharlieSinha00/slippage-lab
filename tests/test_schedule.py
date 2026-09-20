"""Schedules must place every share. Rounding is where shares go missing."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from slippage_lab.schedule import apportion, front_loaded, pov, twap
from slippage_lab.tape import build_tape

VOLUME = build_tape(n_bins=13, seed=3).bin_volume


@pytest.mark.parametrize(
    ("quantity", "n_bins", "lot"),
    list(itertools.product([1, 99, 100, 101, 7_777, 400_000, 1_000_003], [1, 7, 13, 26], [1, 100])),
)
def test_every_schedule_places_exactly_the_parent_quantity(
    quantity: int, n_bins: int, lot: int
) -> None:
    volume = build_tape(n_bins=n_bins, seed=1).bin_volume
    for plan in (
        twap(quantity, n_bins, lot=lot),
        pov(quantity, volume, lot=lot),
        front_loaded(quantity, n_bins, lot=lot),
    ):
        assert plan.sum() == quantity
        assert plan.min() >= 0
        assert plan.size == n_bins


def test_lot_sizing_is_respected_apart_from_one_odd_lot() -> None:
    plan = front_loaded(400_050, 26, lot=100)
    assert plan.sum() == 400_050
    odd = plan % 100
    assert (odd != 0).sum() == 1, "at most one bin should carry the odd lot"
    assert odd[np.flatnonzero(odd)[0]] == 50


def test_twap_is_flat_and_ignores_volume() -> None:
    plan = twap(26_000, 26)
    assert np.all(plan == 1_000)


def test_pov_tracks_volume_share_to_within_a_lot() -> None:
    quantity = 1_000_000
    plan = pov(quantity, VOLUME)
    expected = quantity * VOLUME / VOLUME.sum()
    assert np.max(np.abs(plan - expected)) < 1.0


def test_front_loaded_decays_geometrically() -> None:
    plan = front_loaded(1_000_000, 10, decay=0.5)
    ratios = plan[1:6] / plan[0:5]
    assert np.allclose(ratios, 0.5, atol=0.01)
    assert np.all(np.diff(plan) <= 0), "a front-loaded schedule never grows"


def test_a_slower_decay_leaves_more_for_later() -> None:
    fast = front_loaded(1_000_000, 20, decay=0.4)
    slow = front_loaded(1_000_000, 20, decay=0.8)
    assert fast[0] > slow[0]
    assert fast[10:].sum() < slow[10:].sum()


def test_apportion_uses_largest_remainder_not_truncation() -> None:
    # Three equal bins and ten shares: truncation would place 9 and lose one.
    plan = apportion(np.ones(3), 10)
    assert plan.sum() == 10
    assert sorted(plan.tolist()) == [3, 3, 4]


def test_apportion_ties_break_towards_the_earlier_bin() -> None:
    plan = apportion(np.ones(4), 6)
    assert plan.tolist() == [2, 2, 1, 1]


@pytest.mark.parametrize(
    "bad",
    [
        {"weights": np.zeros(4), "total_quantity": 100},
        {"weights": np.array([1.0, -1.0]), "total_quantity": 100},
        {"weights": np.ones(4), "total_quantity": -1},
    ],
)
def test_apportion_rejects_nonsense(bad: dict) -> None:
    with pytest.raises(ValueError, match=r"weights|quantity"):
        apportion(**bad)


def test_front_loaded_rejects_a_decay_outside_the_unit_interval() -> None:
    for decay in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="decay must lie"):
            front_loaded(1_000, 10, decay=decay)
