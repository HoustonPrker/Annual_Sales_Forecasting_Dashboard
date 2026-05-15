"""Tests for affiliation_override.assign_affiliation."""
import pytest
from unittest.mock import MagicMock
from affiliation_override import assign_affiliation, AFFILIATION_DEFAULTS, FTE_VERY_LARGE_THRESHOLD


def _clf(flag: int):
    """Return a mock flag classifier that always predicts `flag`."""
    m = MagicMock()
    m.predict.return_value = [flag]
    return m


def test_flag1_default():
    r = assign_affiliation(_clf(1), giftshop_sqft=400, occupancy_rate=0.6, fte=1000)
    assert r["tier_label"] == "Default"
    assert r["affiliation_enc"] == AFFILIATION_DEFAULTS["Default"]
    assert r["escalated"] is False
    assert r["flag_predicted"] == 1
    assert r["flag_used"] == 1


def test_flag2_default():
    r = assign_affiliation(_clf(2), giftshop_sqft=400, occupancy_rate=0.6, fte=1000)
    assert r["tier_label"] == "Default"
    assert r["affiliation_enc"] == AFFILIATION_DEFAULTS["Default"]
    assert r["escalated"] is False


def test_flag3_default():
    r = assign_affiliation(_clf(3), giftshop_sqft=600, occupancy_rate=0.8, fte=3000)
    assert r["tier_label"] == "Default"
    assert r["affiliation_enc"] == AFFILIATION_DEFAULTS["Default"]
    assert r["escalated"] is False


def test_flag4_no_escalation():
    r = assign_affiliation(_clf(4), giftshop_sqft=1200, occupancy_rate=0.9, fte=2000)
    assert r["tier_label"] == "Large"
    assert r["affiliation_enc"] == AFFILIATION_DEFAULTS["Large"]
    assert r["escalated"] is False
    assert r["flag_used"] == 4


def test_flag4_fte_escalation():
    r = assign_affiliation(_clf(4), giftshop_sqft=1200, occupancy_rate=0.9, fte=10000)
    assert r["tier_label"] == "Very Large"
    assert r["affiliation_enc"] == AFFILIATION_DEFAULTS["Very Large"]
    assert r["escalated"] is True
    assert r["flag_predicted"] == 4
    assert r["flag_used"] == 5


def test_flag5_very_large_no_escalation():
    r = assign_affiliation(_clf(5), giftshop_sqft=2000, occupancy_rate=1.05, fte=500)
    assert r["tier_label"] == "Very Large"
    assert r["affiliation_enc"] == AFFILIATION_DEFAULTS["Very Large"]
    assert r["escalated"] is False
    assert r["flag_used"] == 5


def test_boundary_fte_below_threshold():
    r = assign_affiliation(_clf(4), giftshop_sqft=1200, occupancy_rate=0.9,
                           fte=FTE_VERY_LARGE_THRESHOLD - 1)
    assert r["tier_label"] == "Large"
    assert r["escalated"] is False


def test_boundary_fte_at_threshold():
    r = assign_affiliation(_clf(4), giftshop_sqft=1200, occupancy_rate=0.9,
                           fte=FTE_VERY_LARGE_THRESHOLD)
    assert r["tier_label"] == "Very Large"
    assert r["escalated"] is True
