"""
Pre-prediction affiliation default selector.

Picks an Affiliation_enc value based on the flag classifier's tier
prediction plus a tight FTE escalation rule for true flagship hospitals.

Derived from leave-one-out analysis of 37-store training cohort and
validated on 4 holdout stores, May 2026.

NO hardcoded store IDs. All logic is feature-based.
"""
import math

import joblib
import pandas as pd

# Tier-conditional affiliation defaults (median Affiliation_enc within tier)
AFFILIATION_DEFAULTS = {
    "Default":    34252,   # global median -- unchanged for flags 1, 2, 3
    "Large":      40885,   # median of Large-tier training stores
    "Very Large": 78181,   # median of Very Large flagship stores
}

# FTE threshold for escalating flag=4 (Large) -> flag=5 (Very Large).
# Tight: catches Strong Memorial (10,009 FTE) without catching NM Memorial
# (4,811 FTE) or any other flag=4 store in training data.
FTE_VERY_LARGE_THRESHOLD = 8000


def load_flag_classifier(path: str = "model_files/flag_classifier.joblib"):
    """Load the trained flag classifier. Cache the result in the caller."""
    return joblib.load(path)


def assign_affiliation(
    flag_classifier,
    giftshop_sqft: float,
    occupancy_rate: float,
    fte: float,
) -> dict:
    """
    Determine the Affiliation_enc value to feed to the main model.

    Args:
        flag_classifier: trained sklearn classifier (output: int 1-5)
        giftshop_sqft:   interior square footage of the gift shop
        occupancy_rate:  hospital occupancy rate (ADC / Staffed_Beds)
        fte:             hospital total FTE count

    Returns:
        dict with:
            affiliation_enc: float -- value to plug into the model feature
            flag_predicted:  int   -- 1-5 from the flag classifier
            flag_used:       int   -- may differ if FTE escalation fired
            tier_label:      str   -- 'Default' | 'Large' | 'Very Large'
            escalated:       bool  -- True if FTE escalation promoted flag=4 to 5
    """
    # Step 1: flag classifier predicts tier from sqft + occupancy
    X = pd.DataFrame([{
        "log_Giftshop_Sq_Ft": math.log(max(giftshop_sqft, 1)),
        "Occupancy_Rate":     occupancy_rate,
    }])
    flag_predicted = int(flag_classifier.predict(X)[0])
    flag_used  = flag_predicted
    escalated  = False

    # Step 2: FTE escalation -- promote flag=4 to flag=5 for true flagships
    if flag_predicted == 4 and fte >= FTE_VERY_LARGE_THRESHOLD:
        flag_used = 5
        escalated = True

    # Step 3: map final flag to affiliation default
    if flag_used == 5:
        tier_label = "Very Large"
    elif flag_used == 4:
        tier_label = "Large"
    else:
        tier_label = "Default"

    return {
        "affiliation_enc": AFFILIATION_DEFAULTS[tier_label],
        "flag_predicted":  flag_predicted,
        "flag_used":       flag_used,
        "tier_label":      tier_label,
        "escalated":       escalated,
    }
