"""Run the three schedules over one tape and lay the answers side by side."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from slippage_lab.benchmarks import (
    arrival_price,
    close_price,
    interval_vwap,
    interval_vwap_including_own_fills,
    slippage_bps,
)
from slippage_lab.execution import Execution, ParentOrder, Side, execute
from slippage_lab.schedule import SCHEDULES
from slippage_lab.shortfall import Shortfall, shortfall_for
from slippage_lab.tape import REGIMES, Tape, build_tape


@dataclass(frozen=True)
class Scenario:
    """Everything needed to reproduce one comparison."""

    regime: str = "trending"
    seed: int = 0
    n_bins: int = 26
    side: Side = Side.BUY
    quantity: int = 400_000
    session_volume: float = 8_000_000.0
    delay_bins: int = 1
    limit_bps: float | None = 70.0
    lot: int = 100
    impact_coef: float = 50.0
    half_spread_bps: float = 1.0
    max_participation: float = 0.25
    vol_bps: float = 60.0
    start_price: float = 100.0

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"unknown regime {self.regime!r}; choose from {sorted(REGIMES)}")
        if not 0 <= self.delay_bins < self.n_bins:
            raise ValueError("delay_bins must leave at least one trading bin")

    def build_tape(self) -> Tape:
        return build_tape(
            n_bins=self.n_bins,
            start_price=self.start_price,
            drift_bps=REGIMES[self.regime],
            vol_bps=self.vol_bps,
            session_volume=self.session_volume,
            seed=self.seed,
        )

    def build_order(self, tape: Tape) -> ParentOrder:
        limit = None
        if self.limit_bps is not None:
            # A limit quoted as "how far through the arrival mid we will go".
            limit = arrival_price(tape) * (1.0 + int(self.side) * self.limit_bps / 10_000.0)
        return ParentOrder(side=self.side, quantity=self.quantity, limit_price=limit)


@dataclass(frozen=True)
class ScheduleResult:
    """One schedule's outcome, measured against both benchmarks."""

    name: str
    execution: Execution
    shortfall: Shortfall
    interval_vwap: float
    vwap_slippage_bps: float
    naive_interval_vwap: float
    naive_vwap_slippage_bps: float
    participation_rate: float


@dataclass(frozen=True)
class Comparison:
    scenario: Scenario
    tape: Tape
    order: ParentOrder
    results: list[ScheduleResult] = field(default_factory=list)

    def by_name(self, name: str) -> ScheduleResult:
        return next(result for result in self.results if result.name == name)

    def best_on_vwap(self) -> ScheduleResult:
        return min(self.results, key=lambda r: r.vwap_slippage_bps)

    def best_on_shortfall(self) -> ScheduleResult:
        return min(self.results, key=lambda r: r.shortfall.total_bps)

    def table(self) -> pd.DataFrame:
        rows = []
        for result in self.results:
            execution = result.execution
            shortfall = result.shortfall
            rows.append(
                {
                    "schedule": result.name,
                    "filled": execution.filled_quantity,
                    "fill %": 100.0 * execution.fill_ratio,
                    "part %": 100.0 * result.participation_rate,
                    "avg px": execution.average_fill_price,
                    "vwap slip bps": result.vwap_slippage_bps,
                    "IS total bps": shortfall.total_bps,
                    "  delay": shortfall.delay_bps,
                    "  exec": shortfall.execution_bps,
                    "  opp": shortfall.opportunity_bps,
                }
            )
        return pd.DataFrame(rows).set_index("schedule")

    def own_fills_table(self) -> pd.DataFrame:
        rows = []
        for result in self.results:
            rows.append(
                {
                    "schedule": result.name,
                    "interval VWAP": result.interval_vwap,
                    "...incl own fills": result.naive_interval_vwap,
                    "slip bps (correct)": result.vwap_slippage_bps,
                    "slip bps (contaminated)": result.naive_vwap_slippage_bps,
                    "error bps": result.naive_vwap_slippage_bps - result.vwap_slippage_bps,
                }
            )
        return pd.DataFrame(rows).set_index("schedule")


def measure(name: str, execution: Execution, tape: Tape) -> ScheduleResult:
    """Score one execution against both benchmarks over its own fill span."""
    shortfall = shortfall_for(execution, tape)
    span = execution.fill_span()
    if span is None:
        nan = float("nan")
        return ScheduleResult(name, execution, shortfall, nan, nan, nan, nan, 0.0)

    first, last = span
    window = slice(first, last + 1)
    market_vwap = interval_vwap(tape.bin_vwap[window], tape.bin_volume[window])
    naive_vwap = interval_vwap_including_own_fills(
        tape.bin_vwap[window],
        tape.bin_volume[window],
        np.nan_to_num(execution.price[window]),
        execution.quantity[window].astype(np.float64),
    )
    average = execution.average_fill_price
    side = execution.order.side
    market_volume = float(tape.bin_volume[window].sum())
    return ScheduleResult(
        name=name,
        execution=execution,
        shortfall=shortfall,
        interval_vwap=market_vwap,
        vwap_slippage_bps=slippage_bps(average, market_vwap, side),
        naive_interval_vwap=naive_vwap,
        naive_vwap_slippage_bps=slippage_bps(average, naive_vwap, side),
        participation_rate=execution.filled_quantity / market_volume,
    )


def run_scenario(scenario: Scenario) -> Comparison:
    """Execute every schedule in ``SCHEDULES`` over one tape and measure each."""
    tape = scenario.build_tape()
    order = scenario.build_order(tape)
    trading_volume = tape.bin_volume[scenario.delay_bins :]

    results = []
    for name, build in SCHEDULES.items():
        target = build(scenario.quantity, trading_volume, scenario.lot)
        execution = execute(
            tape,
            target,
            order,
            start_bin=scenario.delay_bins,
            impact_coef=scenario.impact_coef,
            half_spread_bps=scenario.half_spread_bps,
            max_participation=scenario.max_participation,
        )
        results.append(measure(name, execution, tape))

    return Comparison(scenario=scenario, tape=tape, order=order, results=results)


def render(comparison: Comparison) -> str:
    """Format a comparison for a terminal. The divergence is the point, so it is spelled out."""
    scenario = comparison.scenario
    tape = comparison.tape
    order = comparison.order
    arrival = arrival_price(tape)
    close = close_price(tape)

    limit = "none" if order.limit_price is None else f"{order.limit_price:.4f}"
    header = [
        "slippage-lab  (synthetic data; nothing here is a real market)",
        "",
        f"  regime            {scenario.regime}"
        f"  (expected drift {REGIMES[scenario.regime]:+.0f} bps)",
        f"  seed              {scenario.seed}",
        f"  order             {order.side.name} {order.quantity:,} shares",
        f"  session           {tape.n_bins} bins x {tape.bin_minutes:.0f} min,"
        f" {tape.total_volume:,.0f} shares of market volume",
        f"  order / session   {order.quantity / tape.total_volume:.1%}",
        f"  decision delay    {scenario.delay_bins} bin(s)",
        f"  limit price       {limit}"
        + ("" if order.limit_price is None else f"  ({scenario.limit_bps:+.0f} bps vs arrival)"),
        "",
        f"  arrival mid       {arrival:.4f}",
        f"  close mid         {close:.4f}"
        f"   ({slippage_bps(close, arrival, Side.BUY):+.1f} bps vs arrival)",
        "",
    ]

    floats = {
        "fill %": "{:.1f}",
        "part %": "{:.1f}",
        "avg px": "{:.4f}",
        "vwap slip bps": "{:+.1f}",
        "IS total bps": "{:+.1f}",
        "  delay": "{:+.1f}",
        "  exec": "{:+.1f}",
        "  opp": "{:+.1f}",
    }
    table = comparison.table().to_string(
        formatters={k: v.format for k, v in floats.items()},
        columns=list(floats | {"filled": "{:,}"}),
    )
    body = [
        "  Cost against each benchmark (positive = it cost you)",
        "",
        *(f"  {line}" for line in table.splitlines()),
        "",
    ]

    own = comparison.own_fills_table().to_string(
        formatters={
            "interval VWAP": "{:.4f}".format,
            "...incl own fills": "{:.4f}".format,
            "slip bps (correct)": "{:+.1f}".format,
            "slip bps (contaminated)": "{:+.1f}".format,
            "error bps": "{:+.1f}".format,
        }
    )
    body += [
        "  Interval VWAP with and without the order's own prints in it",
        "",
        *(f"  {line}" for line in own.splitlines()),
        "",
    ]

    vwap_winner = comparison.best_on_vwap()
    shortfall_winner = comparison.best_on_shortfall()
    spans = {result.execution.fill_span() for result in comparison.results}
    common_window = len(spans) == 1
    all_filled = all(result.execution.unfilled_quantity == 0 for result in comparison.results)

    verdict = [
        f"  Interval VWAP ranks first:            {vwap_winner.name}"
        f"  ({vwap_winner.vwap_slippage_bps:+.1f} bps)",
        f"  Implementation shortfall ranks first: {shortfall_winner.name}"
        f"  ({shortfall_winner.shortfall.total_bps:+.1f} bps)",
        "",
    ]
    if vwap_winner.name != shortfall_winner.name:
        vwap_gap = shortfall_winner.vwap_slippage_bps - vwap_winner.vwap_slippage_bps
        is_gap = vwap_winner.shortfall.total_bps - shortfall_winner.shortfall.total_bps
        verdict += [
            f"  The benchmarks disagree. {vwap_winner.name} beats {shortfall_winner.name}"
            f" by {vwap_gap:.1f} bps on interval VWAP",
            f"  and loses to it by {is_gap:.1f} bps on shortfall."
            f" {vwap_winner.name} left {vwap_winner.execution.unfilled_quantity:,} shares"
            f" ({1 - vwap_winner.execution.fill_ratio:.0%}) unfilled;",
            "  interval VWAP never charged for them and shortfall charged"
            f" {vwap_winner.shortfall.opportunity_bps:+.1f} bps.",
        ]
    elif all_filled and common_window:
        verdict += [
            f"  Both benchmarks agree on {vwap_winner.name}, and here they had no choice."
            " Every schedule",
            "  filled completely over the same benchmark window, which makes both metrics"
            " increasing",
            "  affine functions of the average fill price. They must induce the same ranking.",
        ]
    else:
        verdict += [
            f"  Both benchmarks agree on {vwap_winner.name} in this session, but not because"
            " they must:",
            "  the schedules differ in how much they filled or in the interval they were"
            " measured over.",
            "  Try another seed.",
        ]

    if not common_window:
        windows = ", ".join(
            f"{result.name} {result.execution.fill_span()}" for result in comparison.results
        )
        verdict += [
            "",
            f"  Note: the schedules were measured over different intervals (bins: {windows}).",
            "  An order that finishes early is benchmarked against the part of the session"
            " it chose",
            "  to trade in. That is the other way interval VWAP moves under you.",
        ]

    return "\n".join([*header, *body, *verdict, ""])
