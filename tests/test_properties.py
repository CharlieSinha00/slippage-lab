"""End-to-end properties. These are the tests that would fail if the maths were wrong.

Each one asserts something that must hold for a reason, not a number that
happened to come out of a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest

from slippage_lab.execution import Side
from slippage_lab.report import Scenario, render, run_scenario

FRICTIONLESS = {
    "impact_coef": 0.0,
    "half_spread_bps": 0.0,
    "limit_bps": None,
    "delay_bins": 0,
    "lot": 1,
}


def test_a_flat_tape_with_no_frictions_costs_exactly_nothing() -> None:
    """Zero drift, zero volatility, zero trading cost, full fills. The answer is zero.

    Any spurious term anywhere in the pipeline -- a stray half-spread, a
    misplaced sign, a rounding leak -- shows up here as a non-zero number.
    """
    comparison = run_scenario(Scenario(regime="flat", vol_bps=0.0, **FRICTIONLESS))
    for result in comparison.results:
        assert result.execution.unfilled_quantity == 0
        assert result.shortfall.total_bps == pytest.approx(0.0, abs=1e-9)
        assert result.shortfall.parts_bps == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
        assert result.vwap_slippage_bps == pytest.approx(0.0, abs=1e-9)


def test_frictionless_pov_has_no_interval_vwap_slippage_by_construction() -> None:
    """Trade a constant fraction of every print and your average price *is* the VWAP.

    This is the closed form that makes POV's VWAP score close to meaningless as
    a measure of skill, and it holds whatever the price path does.
    """
    for seed in range(6):
        comparison = run_scenario(
            Scenario(regime="trending", seed=seed, vol_bps=80.0, **FRICTIONLESS)
        )
        slippage = comparison.by_name("pov").vwap_slippage_bps
        assert abs(slippage) < 0.1, f"seed {seed} gave {slippage:.4f} bps"


def test_a_frictionless_buy_and_sell_are_exact_mirrors_end_to_end() -> None:
    """Same tape, same schedules, opposite sides. Every cost must flip sign and nothing else."""
    buy = run_scenario(Scenario(side=Side.BUY, **FRICTIONLESS))
    sell = run_scenario(Scenario(side=Side.SELL, **FRICTIONLESS))
    for left, right in zip(buy.results, sell.results, strict=True):
        assert left.name == right.name
        assert left.execution.average_fill_price == pytest.approx(
            right.execution.average_fill_price
        )
        assert left.shortfall.total_bps == pytest.approx(-right.shortfall.total_bps)
        assert left.vwap_slippage_bps == pytest.approx(-right.vwap_slippage_bps)


@pytest.mark.parametrize("regime", ["flat", "trending"])
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_the_decomposition_adds_up_over_the_whole_pipeline(regime: str, side: Side) -> None:
    for seed in range(4):
        comparison = run_scenario(Scenario(regime=regime, seed=seed, side=side))
        for result in comparison.results:
            parts = sum(result.shortfall.parts_bps)
            assert parts == pytest.approx(result.shortfall.total_bps, abs=1e-9)


def test_delay_cost_is_zero_without_a_delay_and_adverse_with_one() -> None:
    no_delay = run_scenario(Scenario(delay_bins=0))
    for result in no_delay.results:
        assert result.shortfall.delay_bps == 0.0

    # Buying into a rally: waiting four bins can only hurt.
    delayed = run_scenario(Scenario(delay_bins=4, side=Side.BUY))
    assert all(r.shortfall.delay_bps > 0.0 for r in delayed.results)

    # Selling into the same rally: waiting helps, so delay cost is negative.
    helped = run_scenario(Scenario(delay_bins=4, side=Side.SELL, limit_bps=None))
    assert all(r.shortfall.delay_bps < 0.0 for r in helped.results)


def test_unfilled_shares_are_charged_at_the_close() -> None:
    comparison = run_scenario(Scenario())
    close_move_bps = (comparison.tape.close_mid / comparison.tape.decision_mid - 1.0) * 10_000.0
    for result in comparison.results:
        execution = result.execution
        assert execution.filled_quantity + execution.unfilled_quantity == execution.order.quantity
        expected = close_move_bps * execution.unfilled_quantity / execution.order.quantity
        assert result.shortfall.opportunity_bps == pytest.approx(expected, abs=1e-9)


def test_the_two_benchmarks_cannot_disagree_when_everyone_fills_over_one_window() -> None:
    """With full fills and a common interval, both metrics are increasing affine functions
    of the average fill price, so they must induce the same ranking. If this test ever
    fails, one of the two metrics has a sign or a scaling error."""
    for seed in range(8):
        comparison = run_scenario(
            Scenario(seed=seed, n_bins=8, limit_bps=None, lot=1, max_participation=1.0)
        )
        spans = {r.execution.fill_span() for r in comparison.results}
        assert len(spans) == 1, "precondition: one common benchmark window"
        assert all(r.execution.unfilled_quantity == 0 for r in comparison.results)

        by_vwap = [r.name for r in sorted(comparison.results, key=lambda r: r.vwap_slippage_bps)]
        by_shortfall = [
            r.name for r in sorted(comparison.results, key=lambda r: r.shortfall.total_bps)
        ]
        assert by_vwap == by_shortfall


def test_the_headline_divergence_actually_appears() -> None:
    """The claim on the tin: on the default scenario the VWAP winner loses on shortfall."""
    comparison = run_scenario(Scenario())
    vwap_winner = comparison.best_on_vwap()
    shortfall_winner = comparison.best_on_shortfall()

    assert vwap_winner.name != shortfall_winner.name
    assert shortfall_winner.name == "front_loaded"
    # The winner on VWAP is the one that did not get done.
    assert vwap_winner.execution.fill_ratio < 0.5
    assert shortfall_winner.execution.fill_ratio > 0.9
    assert vwap_winner.shortfall.total_bps > shortfall_winner.shortfall.total_bps + 50.0


def test_the_divergence_is_structural_rather_than_a_lucky_seed() -> None:
    """It should survive re-drawing the tape. A couple of quiet sessions will not diverge,
    because a quiet session fills everything and then the benchmarks provably agree."""
    diverged = 0
    for seed in range(24):
        comparison = run_scenario(Scenario(seed=seed))
        if comparison.best_on_vwap().name != comparison.best_on_shortfall().name:
            assert comparison.best_on_shortfall().name == "front_loaded"
            diverged += 1
    assert diverged >= 18, f"only {diverged}/24 seeds diverged"


def test_the_same_seed_gives_the_same_report_every_time() -> None:
    scenario = Scenario(seed=11)
    first, second = run_scenario(scenario), run_scenario(scenario)
    assert render(first) == render(second)
    for left, right in zip(first.results, second.results, strict=True):
        assert np.array_equal(left.execution.quantity, right.execution.quantity)
        assert left.shortfall == right.shortfall
        assert left.vwap_slippage_bps == right.vwap_slippage_bps


def test_a_different_seed_gives_different_numbers() -> None:
    assert render(run_scenario(Scenario(seed=1))) != render(run_scenario(Scenario(seed=2)))


def test_an_unknown_regime_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        Scenario(regime="sideways-ish")
