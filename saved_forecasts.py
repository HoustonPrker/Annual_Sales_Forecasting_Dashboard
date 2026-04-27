"""Helpers for persisting forecasts to saved_forecasts.json."""
import io
import json
import uuid
from datetime import date, datetime
from pathlib import Path

import pandas as pd

_PATH = Path(__file__).parent / "saved_forecasts.json"


def _load_all() -> list[dict]:
    if not _PATH.exists():
        return []
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_all(records: list[dict]) -> None:
    _PATH.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def save_forecast(
    hospital_name: str, inputs: dict, result: dict,
    shap_drivers: dict | None = None, shap_base: float = 0.0,
) -> None:
    records = _load_all()
    records.append({
        "id":            str(uuid.uuid4()),
        "hospital_name": hospital_name.strip() or "Unnamed Hospital",
        "saved_at":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "inputs":        inputs,
        "conservative":  result["conservative"],
        "accurate":      result["accurate"],
        "optimistic":    result["optimistic"],
        "monthly_revenue": result["monthly_revenue"],
        "monthly_labels":  result["monthly_labels"],
        "shap_drivers":    shap_drivers or {},
        "shap_base":       shap_base,
    })
    _save_all(records)


def list_forecasts() -> list[dict]:
    return _load_all()


def delete_forecast(forecast_id: str) -> None:
    records = [r for r in _load_all() if r["id"] != forecast_id]
    _save_all(records)


def _dollar_impact(shap_val: float, shap_base: float, all_vals: list[float]) -> float:
    """Annualized marginal dollar impact of one SHAP value (log-space model)."""
    import math
    monthly_pred = math.exp(shap_base + sum(all_vals))
    return monthly_pred * (1 - math.exp(-shap_val)) * 12


def to_excel_bytes(forecasts: list[dict]) -> bytes:
    """Return a .xlsx workbook as bytes. One summary sheet + one sheet per forecast."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Summary sheet
        summary_rows = []
        for f in forecasts:
            summary_rows.append({
                "Hospital Name":  f["hospital_name"],
                "Saved":          f["saved_at"],
                "Conservative":   f["conservative"],
                "Most Likely":    f["accurate"],
                "Optimistic":     f["optimistic"],
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # One sheet per forecast
        seen: dict[str, int] = {}
        for f in forecasts:
            base_name = f["hospital_name"][:28]
            seen[base_name] = seen.get(base_name, 0) + 1
            sheet_name = base_name if seen[base_name] == 1 else f"{base_name[:25]} ({seen[base_name]})"

            inputs_df = pd.DataFrame([{
                "Hospital Name":    f["hospital_name"],
                "Saved":            f["saved_at"],
                "Conservative":     f["conservative"],
                "Most Likely":      f["accurate"],
                "Optimistic":       f["optimistic"],
                "Staffed Beds":     f["inputs"].get("staffed_beds"),
                "ADC":              f["inputs"].get("adc"),
                "Sq Ft":            f["inputs"].get("giftshop_sqft"),
                "Health System":    f["inputs"].get("affiliation"),
                "Hospital Type":    f["inputs"].get("hospital_type"),
                "Payroll Deduction": "Yes" if f["inputs"].get("payroll_ded") else "No",
                "Dist. Elevator":   f["inputs"].get("dist_elevator"),
                "Dist. Cafeteria":  f["inputs"].get("dist_cafeteria"),
            }])
            inputs_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

            monthly = pd.DataFrame({
                "Month":             f.get("monthly_labels", [str(i) for i in range(1, 13)]),
                "Predicted Revenue": f["monthly_revenue"],
            })
            monthly.to_excel(writer, sheet_name=sheet_name, index=False, startrow=3)

            # SHAP drivers table
            drivers = f.get("shap_drivers", {})
            shap_base = f.get("shap_base", 0.0)
            if drivers:
                all_vals = list(drivers.values())
                shap_rows = []
                for feature, val in sorted(drivers.items(),
                                           key=lambda kv: abs(kv[1]), reverse=True):
                    dollar = _dollar_impact(val, shap_base, all_vals)
                    shap_rows.append({
                        "Revenue Driver":         feature,
                        "Direction":              "Increases forecast" if val > 0 else "Decreases forecast",
                        "Annual $ Impact":        round(dollar),
                    })
                shap_df = pd.DataFrame(shap_rows)
                startrow = 3 + len(monthly) + 3
                shap_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)

    return buf.getvalue()


def single_forecast_excel_bytes(
    hospital_name: str, inputs: dict, result: dict,
    shap_drivers: dict | None = None, shap_base: float = 0.0,
) -> bytes:
    """Excel export for a single just-generated forecast (not yet saved)."""
    fake = {
        "hospital_name":   hospital_name or "Unnamed Hospital",
        "saved_at":        date.today().strftime("%Y-%m-%d"),
        "inputs":          inputs,
        "conservative":    result["conservative"],
        "accurate":        result["accurate"],
        "optimistic":      result["optimistic"],
        "monthly_revenue": result["monthly_revenue"],
        "monthly_labels":  result["monthly_labels"],
        "shap_drivers":    shap_drivers or {},
        "shap_base":       shap_base,
    }
    return to_excel_bytes([fake])
