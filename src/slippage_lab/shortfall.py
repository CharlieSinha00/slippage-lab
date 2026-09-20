"""Implementation shortfall against arrival, with the Perold decomposition.

Shortfall is the difference between the paper portfolio the decision asked for
-- the whole parent, filled instantly at the arrival mid -- and the portfolio
actually obtained. It is the benchmark that pays the bill, because it charges
for the shares you did not get as well as the price you paid for the ones you
did.

Three conventions are worth stating, because reasonable implementations differ
and a decomposition is only comparable to itself:

* **Denominator.** Every term is expressed in basis points of the *parent*
  notional at arrival (``quantity x arrival_price``). Using the executed
  notional instead would make an order that filled 20% of itself look
  wonderful, which is precisely the failure this package is about.
* **Delay cost applies to the executed quantity only.** The unexecuted
  quantity's entire adverse move, from arrival all the way to the close, is
  already charged as opportunity cost. Charging it delay as well would double
  count it.
* **Opportunity cost marks the unfilled shares at the end-of-window price.**
  The order is over; the close is the price at which the gap between the paper
  portfolio and the real one becomes permanent.

The three terms sum to the total exactly, which is asserted in the tests rather
than assumed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from slippage_lab.execution import Execution, Side
from slippage_lab.tape import Tape


@dataclass(frozen=True)
class Shortfall:
    """A Perold decomposition, all terms in bps of parent notional at arrival."""

    delay_bps: float
    """Price drift between the decision and the moment child orders went live."""

    execution_bps: float
    """Everything between going live and the average fill: spread, impact, timing."""

    opportunity_bps: float
    """Unfilled shares marked from arrival to the close."""

    total_bps: float
    """Computed independently of the three parts, so the parts can be checked."""

    filled_quantity: int
    unfilled_quantity: int
    average_fill_price: float

    @property
    def parts_bps(self) -> tuple[float, float, float]:
        return (self.delay_bps, self.execution_bps, self.opportunity_bps)


def implementation_shortfall(
    *,
    side: Side | int,
    parent_quantity: int,
    arrival_price: float,
    market_entry_price: float,
    average_fill_price: float,
    filled_quantity: int,
    end_price: float,
) -> Shortfall:
    """Shortfall in bps of parent notional, decomposed into delay/execution/opportunity.

    ``market_entry_price`` is the mid at the instant the first child order went
    live -- the arrival price *at the market*, as distinct from the arrival
    price *of the decision*. The gap between the two is the delay.
    """
    if parent_quantity <= 0:
        raise ValueError("parent quantity must be positive")
    if arrival_price <= 0.0:
        raise ValueError("arrival price must be positive")
    if not 0 <= filled_quantity <= parent_quantity:
        raise ValueError("filled quantity must lie between zero and the parent quantity")

    sign = int(side)
    unfilled = parent_quantity - filled_quantity
    scale = 10_000.0 * sign / (parent_quantity * arrival_price)

    if filled_quantity == 0:
        # average_fill_price is NaN in this case; keep it out of the arithmetic.
        delay_bps = 0.0
        execution_bps = 0.0
        filled_cost = 0.0
    else:
        delay_bps = scale * filled_quantity * (market_entry_price - arrival_price)
        execution_bps = scale * filled_quantity * (average_fill_price - market_entry_price)
        filled_cost = filled_quantity * (average_fill_price - arrival_price)

    opportunity_bps = scale * unfilled * (end_price - arrival_price)
    total_bps = scale * (filled_cost + unfilled * (end_price - arrival_price))

    return Shortfall(
        delay_bps=delay_bps,
        execution_bps=execution_bps,
        opportunity_bps=opportunity_bps,
        total_bps=total_bps,
        filled_quantity=filled_quantity,
        unfilled_quantity=unfilled,
        average_fill_price=average_fill_price,
    )


def shortfall_for(execution: Execution, tape: Tape) -> Shortfall:
    """Convenience wrapper: pull arrival, entry and close prices off the tape."""
    return implementation_shortfall(
        side=execution.order.side,
        parent_quantity=execution.order.quantity,
        arrival_price=tape.decision_mid,
        market_entry_price=float(tape.bin_open_mid[execution.start_bin]),
        average_fill_price=execution.average_fill_price,
        filled_quantity=execution.filled_quantity,
        end_price=tape.close_mid,
    )
