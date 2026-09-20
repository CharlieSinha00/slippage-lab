"""Implementation shortfall: the decomposition must add up, and the parts must mean something."""

from __future__ import annotations

import itertools

import pytest

from slippage_lab.execution import Side
from slippage_lab.shortfall import implementation_shortfall

BASE = {
    "side": Side.BUY,
    "parent_quantity": 100_000,
    "arrival_price": 100.0,
    "market_entry_price": 100.2,
    "average_fill_price": 100.5,
    "filled_quantity": 80_000,
    "end_price": 101.0,
}


@pytest.mark.parametrize(
    ("side", "filled", "entry", "average", "end"),
    list(
        itertools.product(
            [Side.BUY, Side.SELL],
            [0, 1, 37_500, 99_999, 100_000],
            [99.0, 100.0, 100.4],
            [98.5, 100.0, 101.7],
            [97.0, 100.0, 103.3],
        )
    ),
)
def test_the_parts_sum_to_the_total(
    side: Side, filled: int, entry: float, average: float, end: float
) -> None:
    """``total_bps`` is computed independently of the three parts, so this is a real check."""
    shortfall = implementation_shortfall(
        side=side,
        parent_quantity=100_000,
        arrival_price=100.0,
        market_entry_price=entry,
        average_fill_price=average,
        filled_quantity=filled,
        end_price=end,
    )
    assert sum(shortfall.parts_bps) == pytest.approx(shortfall.total_bps, abs=1e-9)


def test_paying_exactly_arrival_for_everything_costs_nothing() -> None:
    shortfall = implementation_shortfall(
        side=Side.BUY,
        parent_quantity=50_000,
        arrival_price=100.0,
        market_entry_price=100.0,
        average_fill_price=100.0,
        filled_quantity=50_000,
        end_price=137.0,  # irrelevant: nothing was left unfilled
    )
    assert shortfall.total_bps == 0.0
    assert shortfall.parts_bps == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_a_completely_missed_order_is_pure_opportunity_cost(side: Side) -> None:
    shortfall = implementation_shortfall(
        side=side,
        parent_quantity=100_000,
        arrival_price=100.0,
        market_entry_price=101.0,
        average_fill_price=float("nan"),
        filled_quantity=0,
        end_price=102.0,
    )
    assert shortfall.filled_quantity == 0
    assert shortfall.unfilled_quantity == 100_000
    assert shortfall.delay_bps == 0.0
    assert shortfall.execution_bps == 0.0
    assert shortfall.opportunity_bps == pytest.approx(200.0 * int(side))
    assert shortfall.total_bps == pytest.approx(200.0 * int(side))


def test_opportunity_cost_is_marked_at_the_close_not_the_last_fill() -> None:
    """Half filled at arrival, then the market runs away.

    The miss is worth 100 bps on the half of the order that never traded.
    """
    shortfall = implementation_shortfall(
        side=Side.BUY,
        parent_quantity=100_000,
        arrival_price=100.0,
        market_entry_price=100.0,
        average_fill_price=100.0,
        filled_quantity=50_000,
        end_price=101.0,
    )
    assert shortfall.opportunity_bps == pytest.approx(50.0)  # 50% unfilled x 100 bps
    assert shortfall.total_bps == pytest.approx(50.0)


def test_delay_cost_is_the_move_before_the_order_went_live() -> None:
    shortfall = implementation_shortfall(**{**BASE, "average_fill_price": 100.2})
    # 20 bps of pre-trade drift on 80% of the order.
    assert shortfall.delay_bps == pytest.approx(16.0)
    assert shortfall.execution_bps == pytest.approx(0.0)


def test_execution_cost_is_measured_from_the_market_entry_price() -> None:
    shortfall = implementation_shortfall(**BASE)
    # 30 bps between entry (100.2) and the average fill (100.5), on 80% of the order.
    assert shortfall.execution_bps == pytest.approx(24.0)


def test_terms_are_bps_of_the_parent_notional_not_the_executed_notional() -> None:
    """An order that filled a fifth of itself at a good price is not four fifths of a hero."""
    tiny_fill = implementation_shortfall(
        side=Side.BUY,
        parent_quantity=100_000,
        arrival_price=100.0,
        market_entry_price=100.0,
        average_fill_price=100.1,
        filled_quantity=20_000,
        end_price=100.0,
    )
    assert tiny_fill.execution_bps == pytest.approx(2.0)  # 10 bps on 20% of the parent


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_buying_and_selling_the_same_tape_are_mirror_images(side: Side) -> None:
    shortfall = implementation_shortfall(
        side=side,
        parent_quantity=10_000,
        arrival_price=100.0,
        market_entry_price=100.5,
        average_fill_price=101.0,
        filled_quantity=6_000,
        end_price=102.0,
    )
    flipped = implementation_shortfall(
        side=Side(-int(side)),
        parent_quantity=10_000,
        arrival_price=100.0,
        market_entry_price=100.5,
        average_fill_price=101.0,
        filled_quantity=6_000,
        end_price=102.0,
    )
    assert shortfall.total_bps == pytest.approx(-flipped.total_bps)
    for part, mirrored in zip(shortfall.parts_bps, flipped.parts_bps, strict=True):
        assert part == pytest.approx(-mirrored)


@pytest.mark.parametrize(
    "bad",
    [
        {"parent_quantity": 0},
        {"arrival_price": 0.0},
        {"filled_quantity": 100_001},
        {"filled_quantity": -1},
    ],
)
def test_impossible_inputs_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError, match=r"must be|must lie"):
        implementation_shortfall(**{**BASE, **bad})
