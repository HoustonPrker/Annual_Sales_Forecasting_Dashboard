"""Tests for tier_correction.apply_tier_correction."""
import os
import pytest
from tier_correction import apply_tier_correction


def _run(raw, lower=None, upper=None):
    lo = lower if lower is not None else raw * 0.8
    hi = upper if upper is not None else raw * 1.2
    return apply_tier_correction(raw, lo, hi)


def test_small_tier():
    r = _run(200_000)
    assert r["tier"] == "Small"
    assert r["applied"] is True
    assert abs(r["corrected"] - 189_920) < 1
    assert abs(r["factor"] - 0.9496) < 1e-6


def test_medium_tier_no_correction():
    r = _run(375_000)
    assert r["tier"] == "Medium"
    assert r["applied"] is False
    assert r["corrected"] == 375_000
    assert r["factor"] == 1.0


def test_large_tier():
    r = _run(700_000)
    assert r["tier"] == "Large"
    assert r["applied"] is True
    # 700_000 * 1.5956 = 1_116_920, but capped at 700_000 * 1.4 = 980_000
    assert abs(r["corrected"] - 980_000) < 1


def test_small_medium_buffer():
    r = _run(290_000)
    assert r["applied"] is False
    assert r["factor"] == 1.0
    assert r["corrected"] == 290_000


def test_medium_large_buffer():
    r = _run(480_000)
    assert r["applied"] is False
    assert r["factor"] == 1.0
    assert r["corrected"] == 480_000


def test_impact_cap_large():
    # 700_000 * 1.5956 would be 1_116_920; cap is 700_000 * 1.4 = 980_000
    r = _run(700_000)
    assert r["corrected"] == pytest.approx(980_000, abs=1)


def test_ci_scales_with_correction():
    r = apply_tier_correction(200_000, 160_000, 240_000)
    assert r["applied"] is True
    assert r["lower"] == pytest.approx(160_000 * r["factor"], abs=1)
    assert r["upper"] == pytest.approx(240_000 * r["factor"], abs=1)


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("TIER_CORRECTION_ENABLED", "false")
    r = _run(700_000)
    assert r["applied"] is False
    assert r["corrected"] == 700_000
