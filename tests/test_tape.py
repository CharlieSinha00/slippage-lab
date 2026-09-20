"""The tape must be reproducible, and shaped the way intraday markets are shaped."""

from __future__ import annotations

import numpy as np
import pytest

from slippage_lab.tape import Tape, build_tape


def test_the_same_seed_gives_the_same_tape() -> None:
    first = build_tape(seed=42)
    second = build_tape(seed=42)
    assert np.array_equal(first.bin_vwap, second.bin_vwap)
    assert np.array_equal(first.bin_volume, second.bin_volume)
    assert np.array_equal(first.bin_open_mid, second.bin_open_mid)
    assert first.close_mid == second.close_mid


def test_a_different_seed_gives_a_different_tape() -> None:
    assert not np.array_equal(build_tape(seed=1).bin_vwap, build_tape(seed=2).bin_vwap)


def test_volume_is_u_shaped() -> None:
    volume = build_tape(n_bins=26, volume_noise=0.0).bin_volume
    open_bins, midday, close_bins = volume[:3].sum(), volume[11:15].sum(), volume[-3:].sum()
    assert open_bins > midday
    assert close_bins > midday
    assert volume.argmin() in range(10, 16)


def test_volume_sums_to_the_requested_session_volume() -> None:
    tape = build_tape(session_volume=5_000_000.0, seed=9)
    assert tape.total_volume == pytest.approx(5_000_000.0)


def test_zero_volatility_makes_the_path_exactly_deterministic() -> None:
    tape = build_tape(vol_bps=0.0, drift_bps=0.0, seed=5)
    assert np.allclose(tape.bin_vwap, 100.0, atol=1e-12)
    assert tape.close_mid == pytest.approx(100.0)


def test_drift_with_no_noise_lands_on_the_requested_drift() -> None:
    tape = build_tape(vol_bps=0.0, drift_bps=250.0, seed=5, start_price=100.0)
    assert tape.close_mid == pytest.approx(102.5, rel=1e-9)
    assert np.all(np.diff(tape.bin_open_mid) > 0.0)


def test_bin_vwap_sits_inside_the_bin_it_describes() -> None:
    """A within-bin average must lie between the bin's opening and closing mid."""
    tape = build_tape(vol_bps=0.0, drift_bps=250.0, seed=1, n_bins=26)
    boundaries = np.append(tape.bin_open_mid, tape.close_mid)
    assert np.all(tape.bin_vwap > boundaries[:-1])
    assert np.all(tape.bin_vwap < boundaries[1:])


def test_decision_mid_is_the_open_of_the_first_bin() -> None:
    tape = build_tape(seed=3)
    assert tape.decision_mid == tape.bin_open_mid[0]


def test_a_malformed_tape_is_rejected() -> None:
    good = build_tape(n_bins=4, seed=0)
    with pytest.raises(ValueError, match="same length"):
        Tape(
            bin_vwap=good.bin_vwap,
            bin_volume=good.bin_volume[:3],
            bin_open_mid=good.bin_open_mid,
            close_mid=good.close_mid,
            bin_minutes=15.0,
        )
    with pytest.raises(ValueError, match="positive market volume"):
        Tape(
            bin_vwap=good.bin_vwap,
            bin_volume=np.zeros(4),
            bin_open_mid=good.bin_open_mid,
            close_mid=good.close_mid,
            bin_minutes=15.0,
        )


@pytest.mark.parametrize("bad", [{"n_bins": 0}, {"steps_per_bin": 0}, {"session_volume": 0.0}])
def test_impossible_tape_parameters_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError, match="must be"):
        build_tape(**bad)
