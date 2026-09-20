"""The CLI has one job: make the divergence visible in a terminal."""

from __future__ import annotations

import pytest

from slippage_lab.cli import main


def run(capsys: pytest.CaptureFixture[str], *args: str) -> str:
    assert main(list(args)) == 0
    return capsys.readouterr().out


def test_the_default_run_reports_a_disagreement(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys)
    assert "The benchmarks disagree" in out
    for name in ("twap", "pov", "front_loaded"):
        assert name in out
    assert "synthetic" in out


def test_every_schedule_appears_with_both_benchmarks(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "--regime", "trending", "--seed", "7")
    assert "vwap slip bps" in out
    assert "IS total bps" in out
    assert "delay" in out
    assert "exec" in out
    assert "opp" in out


def test_removing_the_limit_makes_every_schedule_complete(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = run(capsys, "--no-limit")
    assert out.count("100.0") >= 3  # three fill percentages
    assert "The benchmarks disagree" not in out


def test_the_run_is_reproducible(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(capsys, "--seed", "5") == run(capsys, "--seed", "5")


def test_both_sides_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert "BUY" in run(capsys, "--side", "buy")
    assert "SELL" in run(capsys, "--side", "sell")
