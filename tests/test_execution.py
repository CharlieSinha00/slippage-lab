"""Fills: the limit is a limit, the cap is a cap, and no share is dropped in silence."""

from __future__ import annotations

import numpy as np
import pytest

from slippage_lab.execution import ParentOrder, Side, execute, impact_bps
from slippage_lab.schedule import front_loaded, pov, twap
from slippage_lab.tape import build_tape

TAPE = build_tape(n_bins=26, drift_bps=250.0, vol_bps=60.0, seed=4)


def test_impact_grows_with_participation_but_concavely() -> None:
    small = impact_bps(0.01, impact_coef=50.0, half_spread_bps=1.0)
    large = impact_bps(0.16, impact_coef=50.0, half_spread_bps=1.0)
    huge = impact_bps(0.64, impact_coef=50.0, half_spread_bps=1.0)
    assert small < large < huge
    assert (huge - large) < (large - small) * 4, "square-root impact must not be linear"
    assert impact_bps(0.0, impact_coef=50.0, half_spread_bps=1.0) == 1.0


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_no_fill_ever_prints_through_the_limit(side: Side) -> None:
    arrival = TAPE.decision_mid
    limit = arrival * (1.0 + int(side) * 30.0 / 10_000.0)
    order = ParentOrder(side=side, quantity=400_000, limit_price=limit)
    execution = execute(TAPE, pov(400_000, TAPE.bin_volume), order, impact_coef=50.0)
    traded = execution.quantity > 0
    assert traded.any(), "the test is vacuous if nothing filled"
    assert np.all(int(side) * (execution.price[traded] - limit) <= 0.0)


def test_a_limit_the_market_never_reaches_leaves_the_order_untouched() -> None:
    order = ParentOrder(side=Side.BUY, quantity=400_000, limit_price=TAPE.decision_mid * 0.5)
    execution = execute(TAPE, twap(400_000, 26), order)
    assert execution.filled_quantity == 0
    assert execution.unfilled_quantity == 400_000
    assert np.isnan(execution.average_fill_price)
    assert execution.fill_span() is None


def test_participation_never_exceeds_the_cap() -> None:
    cap = 0.10
    order = ParentOrder(side=Side.BUY, quantity=2_000_000)
    execution = execute(TAPE, front_loaded(2_000_000, 26), order, max_participation=cap)
    rates = execution.quantity / TAPE.bin_volume
    assert rates.max() <= cap + 1e-12


def test_a_capped_slice_is_carried_forward_not_abandoned() -> None:
    """Front-loading asks for more than the early bins can absorb. It should still finish."""
    quantity = 400_000
    order = ParentOrder(side=Side.BUY, quantity=quantity)
    execution = execute(TAPE, front_loaded(quantity, 26), order, max_participation=0.10)
    assert execution.filled_quantity == quantity
    assert execution.quantity[:4].sum() < front_loaded(quantity, 26)[:4].sum()


def test_filled_plus_unfilled_is_always_the_parent_quantity() -> None:
    for limit_bps in (5.0, 30.0, 70.0, 500.0):
        order = ParentOrder(
            side=Side.BUY,
            quantity=400_000,
            limit_price=TAPE.decision_mid * (1.0 + limit_bps / 10_000.0),
        )
        execution = execute(TAPE, twap(400_000, 26), order)
        assert execution.filled_quantity + execution.unfilled_quantity == 400_000
        assert 0.0 <= execution.fill_ratio <= 1.0


def test_no_trading_happens_during_the_delay() -> None:
    order = ParentOrder(side=Side.BUY, quantity=400_000)
    execution = execute(TAPE, twap(400_000, 23), order, start_bin=3)
    assert execution.quantity[:3].sum() == 0
    assert execution.filled_quantity == 400_000


def test_a_schedule_that_does_not_sum_to_the_parent_is_refused() -> None:
    order = ParentOrder(side=Side.BUY, quantity=400_000)
    with pytest.raises(ValueError, match="does not sum"):
        execute(TAPE, twap(399_900, 26), order)


def test_a_schedule_of_the_wrong_length_is_refused() -> None:
    order = ParentOrder(side=Side.BUY, quantity=400_000)
    with pytest.raises(ValueError, match="trading bins"):
        execute(TAPE, twap(400_000, 26), order, start_bin=3)


def test_the_average_fill_price_is_quantity_weighted() -> None:
    order = ParentOrder(side=Side.BUY, quantity=400_000)
    execution = execute(
        TAPE, front_loaded(400_000, 26), order, impact_coef=0.0, half_spread_bps=0.0
    )
    traded = execution.quantity > 0
    expected = np.average(execution.price[traded], weights=execution.quantity[traded])
    assert execution.average_fill_price == pytest.approx(expected)


def test_a_zero_or_negative_parent_is_refused() -> None:
    with pytest.raises(ValueError, match="quantity must be positive"):
        ParentOrder(side=Side.BUY, quantity=0)
    with pytest.raises(ValueError, match="limit price must be positive"):
        ParentOrder(side=Side.BUY, quantity=100, limit_price=-1.0)
