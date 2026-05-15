"""
Diagnostic: does affiliation_enc meaningfully shift predictions?

Run: python test_affiliation_hypothesis.py

Nothing in this script modifies the Streamlit app.
"""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd


# ── Load artifacts ────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")
ROOT_DIR  = os.path.dirname(__file__)

cfg      = json.load(open(os.path.join(MODEL_DIR, "model_config.json")))
lgbm     = joblib.load(os.path.join(MODEL_DIR, "prod_lgbm.joblib"))
ridge    = joblib.load(os.path.join(MODEL_DIR, "prod_ridge.joblib"))
enet     = joblib.load(os.path.join(MODEL_DIR, "prod_enet.joblib"))
explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))

aff_lookup = json.load(open(os.path.join(ROOT_DIR, "affiliation_lookup.json")))
aff_lookup["Other / New System"] = aff_lookup.pop("__OTHER__", aff_lookup.get("Other / New System"))

ht_lookup = json.load(open(os.path.join(ROOT_DIR, "hospital_type_lookup.json")))

# SHAP base value
base = getattr(explainer, "expected_value", 0.0)
shap_base = float(np.asarray(base).flat[0])
cfg["shap_base_value"] = shap_base

weights = cfg["blend_weights"]
features = cfg["features"]
rs = cfg["residual_shifts"]

# ── Affiliation anchor values ─────────────────────────────────────────────────
aff_items = sorted(aff_lookup.items(), key=lambda x: x[1])
aff_low    = aff_items[0]        # (name, encoded_value) — lowest
aff_median = aff_items[len(aff_items) // 2]
aff_high   = aff_items[-1]       # highest

print("=" * 70)
print("AFFILIATION ANCHOR VALUES")
print(f"  LOW    : {aff_low[0]:<45} encoded = {aff_low[1]:>10,.2f}")
print(f"  MEDIAN : {aff_median[0]:<45} encoded = {aff_median[1]:>10,.2f}")
print(f"  HIGH   : {aff_high[0]:<45} encoded = {aff_high[1]:>10,.2f}")
print("=" * 70)
print()

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_log(x):
    return math.log(max(x, 1e-9))

def month_trig(m):
    return math.sin(2 * math.pi * m / 12), math.cos(2 * math.pi * m / 12)

def build_row(store_base: dict, affiliation_enc: float, month: int = 6) -> pd.DataFrame:
    sine, cosine = month_trig(month)
    row = {
        "Months_Since_Open":          store_base["months_since_open"],
        "Affiliation_enc":            affiliation_enc,
        "log_Staffed_Beds":           safe_log(store_base["staffed_beds"]),
        "Time to Main Elevator Bank": store_base["dist_elevator"],
        "log_ADC":                    safe_log(store_base["adc"]),
        "Time to Cafeteria":          store_base["dist_cafeteria"],
        "Month_Sine":                 sine,
        "Month_Cosine":               cosine,
        "Month_Fraction":             1.0,
        "log_Giftshop_Sq_Ft":         safe_log(store_base["giftshop_sqft"]),
        "Occupancy_Rate":             store_base["occupancy_rate"],
        "Hospital_Type_enc":          ht_lookup.get(store_base["hospital_type"], 0.0),
        "Payroll Ded":                store_base["payroll_ded"],
    }
    return pd.DataFrame([row], columns=features)

def predict_annual(store_base: dict, affiliation_enc: float) -> dict:
    """Run 12-month forecast, return annual raw + calibrated."""
    monthly = []
    for m in range(1, 13):
        row_df = build_row(store_base, affiliation_enc, month=m)
        log_rev = (
            weights["lgbm"]  * lgbm.predict(row_df)[0]
            + weights["ridge"] * ridge.predict(row_df)[0]
            + weights["enet"]  * enet.predict(row_df)[0]
        )
        monthly.append(math.exp(log_rev))

    raw = sum(monthly)
    return {
        "raw":        raw,
        "calibrated": raw,
        "tier":       "—",
        "cap_fired":  False,
        "applied":    False,
    }

# ── Store definitions ─────────────────────────────────────────────────────────
# Store 136 — Brigham & Women's Hospital  (actual $1,780,967)
# Features provided explicitly in task spec.
# Note: FTE/Births/NICU_Beds/Pediatric_Beds not in V1 feature set; omitted.
store_brigham = {
    "label":            "Brigham (136)",
    "actual":           1_780_967,
    "hospital_type":    "Academic",
    "staffed_beds":     818,
    "adc":              858.6,
    "giftshop_sqft":    2245,
    "occupancy_rate":   1.05,
    "payroll_ded":      1,
    "months_since_open": 6,   # mid-year reference month
    "dist_elevator":    30,   # seconds — not available in summary; using typical
    "dist_cafeteria":   60,
}

# Store 134 — Mary Greeley Medical Center  (actual $263,120 per store_summary.csv)
# No per-store feature file in repo. Values chosen to be consistent with a
# ~220-bed community hospital in Ames, IA (City of Ames affiliation present in lookup).
# Occupancy and ADC back-calculated to match reported actual roughly.
store_mary_greeley = {
    "label":            "Mary Greeley (134)",
    "actual":           263_120,
    "hospital_type":    "Community",
    "staffed_beds":     220,
    "adc":              110,
    "giftshop_sqft":    480,
    "occupancy_rate":   0.50,
    "payroll_ded":      1,
    "months_since_open": 6,
    "dist_elevator":    30,
    "dist_cafeteria":   75,
}

# Store 149 — BSWH Centennial  (actual $72,831)
# Not in training summary (store 149 absent from store_summary.csv).
# Using inputs consistent with a small/remote community store at that revenue level.
store_bswh = {
    "label":            "BSWH Centennial (149)",
    "actual":           72_831,
    "hospital_type":    "Community",
    "staffed_beds":     100,
    "adc":              35,
    "giftshop_sqft":    250,
    "occupancy_rate":   0.35,
    "payroll_ded":      0,
    "months_since_open": 6,
    "dist_elevator":    90,
    "dist_cafeteria":   150,
}

STORES = [store_brigham, store_mary_greeley, store_bswh]
AFFILIATIONS = [
    ("Low",    aff_low[0],    aff_low[1]),
    ("Median", aff_median[0], aff_median[1]),
    ("High",   aff_high[0],   aff_high[1]),
]

# ── Run all 9 predictions ─────────────────────────────────────────────────────
print("Running 9 predictions...")
print()

rows = []
for store in STORES:
    for level, aff_name, aff_enc in AFFILIATIONS:
        r = predict_annual(store, aff_enc)
        rows.append({
            "store":      store["label"],
            "actual":     store["actual"],
            "aff_level":  level,
            "aff_name":   aff_name,
            "aff_enc":    aff_enc,
            "raw":        r["raw"],
            "calibrated": r["calibrated"],
            "tier":       r["tier"],
            "cap_fired":  r["cap_fired"],
            "applied":    r["applied"],
        })

# ── Print table ───────────────────────────────────────────────────────────────
COL = {
    "store":     28,
    "affil":     22,
    "raw":       13,
    "calib":     13,
    "tier":      9,
    "cap":       10,
}

def _h(s, w): return s[:w].ljust(w)

header = (
    _h("Store",                  COL["store"])
    + _h("Affiliation",          COL["affil"])
    + _h("Raw",                  COL["raw"])
    + _h("Calibrated",           COL["calib"])
    + _h("Tier",                 COL["tier"])
    + _h("Cap fired?",           COL["cap"])
)
sep = "-" * len(header)

print(header)
print(sep)

prev_store = None
for r in rows:
    if prev_store and prev_store != r["store"]:
        print()
    prev_store = r["store"]

    aff_label = f"{r['aff_level']} (enc={r['aff_enc']:,.0f})"
    cap_str   = "Yes" if r["cap_fired"] else ("No" if r["applied"] else "n/a")

    print(
        _h(r["store"],    COL["store"])
        + _h(aff_label,   COL["affil"])
        + _h(f"${r['raw']:>10,.0f}", COL["raw"])
        + _h(f"${r['calibrated']:>10,.0f}", COL["calib"])
        + _h(r["tier"],   COL["tier"])
        + cap_str
    )

# ── Range summaries ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("RANGE SUMMARIES")
print("=" * 70)

for store in STORES:
    store_rows = [r for r in rows if r["store"] == store["label"]]
    raws  = [r["raw"]        for r in store_rows]
    cals  = [r["calibrated"] for r in store_rows]

    raw_lo,  raw_hi  = min(raws),  max(raws)
    cal_lo,  cal_hi  = min(cals),  max(cals)
    raw_spread_pct   = (raw_hi - raw_lo) / raw_lo * 100 if raw_lo else 0
    cal_spread_pct   = (cal_hi - cal_lo) / cal_lo * 100 if cal_lo else 0

    print(f"\n{store['label']}  (actual ${store['actual']:,.0f})")
    print(f"  Raw        range: ${raw_lo:>10,.0f}  to  ${raw_hi:>10,.0f}"
          f"  (spread ${raw_hi - raw_lo:,.0f},  {raw_spread_pct:.1f}%)")
    print(f"  Calibrated range: ${cal_lo:>10,.0f}  to  ${cal_hi:>10,.0f}"
          f"  (spread ${cal_hi - cal_lo:,.0f},  {cal_spread_pct:.1f}%)")
