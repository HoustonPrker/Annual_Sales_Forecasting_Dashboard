"""Model loading, feature engineering, and prediction logic."""
import json
import math
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")


@st.cache_resource
def load_artifacts():
    cfg = json.load(open(os.path.join(MODEL_DIR, "model_config.json")))
    lgbm = joblib.load(os.path.join(MODEL_DIR, "prod_lgbm.joblib"))
    ridge = joblib.load(os.path.join(MODEL_DIR, "prod_ridge.joblib"))
    enet = joblib.load(os.path.join(MODEL_DIR, "prod_enet.joblib"))
    flag_clf = joblib.load(os.path.join(MODEL_DIR, "flag_classifier.joblib"))
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))
    existing = pd.read_csv(os.path.join(MODEL_DIR, "existing_predictions.csv"))
    summary = pd.read_csv(os.path.join(MODEL_DIR, "store_summary.csv"))
    return cfg, lgbm, ridge, enet, flag_clf, explainer, existing, summary


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-9))


def month_trig(m: int) -> tuple[float, float]:
    return math.sin(2 * math.pi * m / 12), math.cos(2 * math.pi * m / 12)


def month_fraction(opening_date: date, cal_month: date) -> float:
    """Fraction of cal_month the store was open. 1.0 for every month after opening."""
    if cal_month.year == opening_date.year and cal_month.month == opening_date.month:
        days_in_month = (
            date(cal_month.year + (cal_month.month // 12), (cal_month.month % 12) + 1, 1)
            - date(cal_month.year, cal_month.month, 1)
        ).days
        return max(0.0, min(1.0, (days_in_month - opening_date.day + 1) / days_in_month))
    return 1.0


def get_outlier_flag(flag_clf, flag_features: list[str], log_sqft: float, occupancy: float) -> int:
    X = pd.DataFrame([[log_sqft, occupancy]], columns=flag_features)
    return int(flag_clf.predict(X)[0])


def blend_predict(lgbm, ridge, enet, weights: dict, row_df: pd.DataFrame) -> float:
    return (
        weights["lgbm"] * lgbm.predict(row_df)[0]
        + weights["ridge"] * ridge.predict(row_df)[0]
        + weights["enet"] * enet.predict(row_df)[0]
    )


def build_feature_row(
    cfg: dict,
    staffed_beds: float, fte: float, adc: float,
    affiliation: str, dist_elevator: float, dist_cafeteria: float,
    opening_date: date, cal_date: date,
    months_since_open: int, outlier_flag: int,
) -> pd.DataFrame:
    sine, cosine = month_trig(cal_date.month)
    row = {
        "Gift_Shop_Outlier_Flag": outlier_flag,
        "Months_Since_Open": months_since_open,
        "log_FTE": safe_log(fte),
        "Affiliation_enc": cfg["affiliation_lookup"].get(
            affiliation, cfg["affiliation_lookup"]["Other / New System"]
        ),
        "log_Staffed_Beds": safe_log(staffed_beds),
        "Time to Main Elevator Bank": dist_elevator,
        "log_ADC": safe_log(adc),
        "Time to Cafeteria": dist_cafeteria,
        "Month_Sine": sine,
        "Month_Cosine": cosine,
        "Month_Fraction": month_fraction(opening_date, cal_date),
    }
    return pd.DataFrame([row], columns=cfg["final_features"])


def predict_12_months(artifacts: tuple, inputs: dict) -> dict:
    cfg, lgbm, ridge, enet, flag_clf, explainer, _, _ = artifacts
    od = inputs["opening_date"]
    outlier_flag = get_outlier_flag(
        flag_clf,
        cfg["flag_classifier_features"],
        safe_log(inputs["giftshop_sqft"]),
        inputs["adc"] / inputs["staffed_beds"],
    )

    monthly_revenue, monthly_dates, feature_rows = [], [], []
    for m in range(1, 13):
        raw = od.month + m - 1
        cal_date = date(od.year + (raw - 1) // 12, ((raw - 1) % 12) + 1, 1)
        row_df = build_feature_row(
            cfg, inputs["staffed_beds"], inputs["fte"], inputs["adc"],
            inputs["affiliation"], inputs["dist_elevator"], inputs["dist_cafeteria"],
            od, cal_date, m, outlier_flag,
        )
        feature_rows.append(row_df)
        log_rev = blend_predict(lgbm, ridge, enet, cfg["blend_weights"], row_df)
        monthly_revenue.append(math.exp(log_rev) * month_fraction(od, cal_date))
        monthly_dates.append(cal_date)

    shap_row = feature_rows[5]
    shap_vals = explainer.shap_values(shap_row)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    total = sum(monthly_revenue)
    ci = cfg["confidence_intervals"]
    return {
        "monthly_revenue": monthly_revenue,
        "monthly_dates": monthly_dates,
        "shap_values": np.array(shap_vals).flatten(),
        "shap_row": shap_row,
        "outlier_flag": outlier_flag,
        "conservative": total * (1 + ci["r_low"]),
        "accurate": total,
        "optimistic": total * (1 + ci["r_high"]),
    }
