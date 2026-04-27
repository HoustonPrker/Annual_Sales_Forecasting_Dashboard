"""Model loading, feature engineering, and prediction logic."""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")
ROOT_DIR  = os.path.dirname(__file__)


@st.cache_resource
def load_artifacts():
    cfg = json.load(open(os.path.join(MODEL_DIR, "model_config.json")))

    affil = json.load(open(os.path.join(ROOT_DIR, "affiliation_lookup.json")))
    affil["Other / New System"] = affil.pop("__OTHER__", affil.get("Other / New System"))
    cfg["affiliation_lookup"] = affil

    cfg["hospital_type_lookup"] = json.load(
        open(os.path.join(ROOT_DIR, "hospital_type_lookup.json"))
    )

    lgbm      = joblib.load(os.path.join(MODEL_DIR, "prod_lgbm.joblib"))
    ridge     = joblib.load(os.path.join(MODEL_DIR, "prod_ridge.joblib"))
    enet      = joblib.load(os.path.join(MODEL_DIR, "prod_enet.joblib"))
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))
    def _read_csv_optional(path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_csv(path, **kwargs)
        except FileNotFoundError:
            return pd.DataFrame()

    existing = _read_csv_optional(os.path.join(MODEL_DIR, "existing_predictions.csv"))
    summary  = _read_csv_optional(os.path.join(MODEL_DIR, "store_summary.csv"))
    return cfg, lgbm, ridge, enet, explainer, existing, summary


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-9))


def month_trig(m: int) -> tuple[float, float]:
    return math.sin(2 * math.pi * m / 12), math.cos(2 * math.pi * m / 12)


def blend_predict(lgbm, ridge, enet, weights: dict, row_df: pd.DataFrame) -> float:
    return (
        weights["lgbm"]  * lgbm.predict(row_df)[0]
        + weights["ridge"] * ridge.predict(row_df)[0]
        + weights["enet"]  * enet.predict(row_df)[0]
    )


def build_feature_row(
    cfg: dict,
    staffed_beds: float, adc: float,
    affiliation: str, dist_elevator: float, dist_cafeteria: float,
    months_since_open: int, calendar_month: int,
    giftshop_sqft: float, occupancy_rate: float,
    hospital_type: str, payroll_ded: int,
) -> pd.DataFrame:
    sine, cosine = month_trig(calendar_month)
    row = {
        "Months_Since_Open":          months_since_open,
        "Affiliation_enc":            cfg["affiliation_lookup"].get(
                                          affiliation,
                                          cfg["affiliation_lookup"]["Other / New System"]
                                      ),
        "log_Staffed_Beds":           safe_log(staffed_beds),
        "Time to Main Elevator Bank": dist_elevator,
        "log_ADC":                    safe_log(adc),
        "Time to Cafeteria":          dist_cafeteria,
        "Month_Sine":                 sine,
        "Month_Cosine":               cosine,
        "Month_Fraction":             1.0,
        "log_Giftshop_Sq_Ft":         safe_log(giftshop_sqft),
        "Occupancy_Rate":             occupancy_rate,
        "Hospital_Type_enc":          cfg["hospital_type_lookup"].get(hospital_type, 0.0),
        "Payroll Ded":                payroll_ded,
    }
    return pd.DataFrame([row], columns=cfg["features"])


def predict_12_months(artifacts: tuple, inputs: dict) -> dict:
    cfg, lgbm, ridge, enet, explainer, _, _ = artifacts
    occupancy_rate = inputs["adc"] / inputs["staffed_beds"]

    monthly_revenue, feature_rows = [], []
    # Iterate Jan–Dec (months 1–12). Months_Since_Open mirrors calendar position
    # so the model sees a full first-year ramp regardless of when the store opens.
    for m in range(1, 13):
        row_df = build_feature_row(
            cfg,
            inputs["staffed_beds"], inputs["adc"],
            inputs["affiliation"], inputs["dist_elevator"], inputs["dist_cafeteria"],
            m, m,
            inputs["giftshop_sqft"], occupancy_rate,
            inputs["hospital_type"], inputs["payroll_ded"],
        )
        feature_rows.append(row_df)
        log_rev = blend_predict(lgbm, ridge, enet, cfg["blend_weights"], row_df)
        monthly_revenue.append(math.exp(log_rev))

    shap_row  = feature_rows[5]
    shap_vals = explainer.shap_values(shap_row)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    total = sum(monthly_revenue)
    rs    = cfg["residual_shifts"]
    return {
        "monthly_revenue": monthly_revenue,
        "monthly_labels":  ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "shap_values":     np.array(shap_vals).flatten(),
        "shap_row":        shap_row,
        "conservative":    total * (1 + rs["conservative"]),
        "accurate":        total,
        "optimistic":      total * (1 + rs["optimistic"]),
    }
