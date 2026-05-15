"""Tier-conditional bias correction for the gift shop revenue ensemble.

Correction factors are derived from leave-one-out cross-validation on a
42-store cold-start cohort (validated May 2026).
"""
import os


def _correction_enabled() -> bool:
    """Read kill-switch from st.secrets (Streamlit Cloud) with os.environ fallback (local dev)."""
    try:
        import streamlit as st
        val = st.secrets.get("TIER_CORRECTION_ENABLED", None)
        if val is not None:
            return str(val).lower() != "false"
    except Exception:
        pass
    return os.environ.get("TIER_CORRECTION_ENABLED", "true").lower() != "false"


# Tier boundaries
_SMALL_MAX  = 300_000
_LARGE_MIN  = 450_000

# Buffer zone: ±10% of each boundary
_BUF_SMALL_LO = _SMALL_MAX * 0.90   # 270_000
_BUF_SMALL_HI = _SMALL_MAX * 1.10   # 330_000
_BUF_LARGE_LO = _LARGE_MIN * 0.90   # 405_000
_BUF_LARGE_HI = _LARGE_MIN * 1.10   # 495_000

_FACTORS = {
    "Small":  0.9496,
    "Medium": 1.0000,
    "Large":  1.5956,
}

_IMPACT_CAP = 0.40  # correction may not shift prediction more than 40%


def _tier(raw: float) -> str:
    if raw <= _SMALL_MAX:
        return "Small"
    if raw <= _LARGE_MIN:
        return "Medium"
    return "Large"


def _in_buffer(raw: float) -> bool:
    return (_BUF_SMALL_LO <= raw <= _BUF_SMALL_HI or
            _BUF_LARGE_LO <= raw <= _BUF_LARGE_HI)


def apply_tier_correction(
    raw: float,
    lower: float,
    upper: float,
) -> dict:
    """Return a dict with raw, corrected, tier, factor, applied, lower, upper.

    Guardrails applied in order:
      1. Env-var kill-switch (TIER_CORRECTION_ENABLED=false disables entirely).
      2. Tier-boundary buffer: no correction within ±10% of a boundary.
      3. Medium tier: no correction (factor = 1.0).
      4. Impact cap: correction cannot shift prediction by more than 40%.
    """
    enabled = _correction_enabled()

    tier   = _tier(raw)
    factor = _FACTORS[tier]

    if not enabled or _in_buffer(raw) or tier == "Medium":
        return {
            "raw":       raw,
            "corrected": raw,
            "tier":      tier,
            "factor":    1.0,
            "applied":   False,
            "lower":     lower,
            "upper":     upper,
        }

    corrected = raw * factor

    # Impact cap: clamp to [raw * 0.6, raw * 1.4]
    cap_lo = raw * (1 - _IMPACT_CAP)
    cap_hi = raw * (1 + _IMPACT_CAP)
    corrected = max(cap_lo, min(cap_hi, corrected))

    # Recalculate effective factor after capping
    effective_factor = corrected / raw if raw else factor

    ci_lower = lower * effective_factor
    ci_upper = upper * effective_factor

    return {
        "raw":       raw,
        "corrected": corrected,
        "tier":      tier,
        "factor":    effective_factor,
        "applied":   True,
        "lower":     ci_lower,
        "upper":     ci_upper,
    }
