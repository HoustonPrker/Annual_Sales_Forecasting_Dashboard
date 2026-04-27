import os

import pandas as pd
import streamlit as st

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "model_files", "store_summary.csv")


def _load() -> pd.DataFrame:
    try:
        return pd.read_csv(_CSV_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


def _metrics_html(avg_accuracy: float, typical_error: float, within: int, total: int) -> str:
    return f"""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
      * {{ box-sizing:border-box; margin:0; padding:0; }}
      body {{ background:transparent; font-family:'Inter',sans-serif; }}
      .row {{ display:flex; gap:16px; }}
      .card {{
        flex:1; background:#fff; border:1px solid #E2E8F0; border-radius:12px;
        padding:22px 28px; box-shadow:0 1px 4px rgba(0,0,0,0.05);
      }}
      .label {{
        font-size:11px; font-weight:700; text-transform:uppercase;
        letter-spacing:.08em; margin-bottom:8px;
      }}
      .value {{ font-size:36px; font-weight:800; line-height:1; }}
      .sub {{ font-size:12px; color:#94A3B8; margin-top:6px; }}
    </style></head>
    <body>
      <div class="row">
        <div class="card">
          <div class="label" style="color:#1D4ED8;">Average Accuracy</div>
          <div class="value" style="color:#1E3A5F;">{avg_accuracy:.0f}%</div>
          <div class="sub">Across all 37 stores</div>
        </div>
        <div class="card">
          <div class="label" style="color:#15803D;">Typical Store</div>
          <div class="value" style="color:#14532D;">{100 - typical_error:.0f}%</div>
          <div class="sub">Accuracy for a typical store</div>
        </div>
        <div class="card">
          <div class="label" style="color:#B45309;">Forecast Range</div>
          <div class="value" style="color:#92400E;">{within} <span style="font-size:20px; color:#64748B;">of {total}</span></div>
          <div class="sub">Stores where actual revenue landed in our range</div>
        </div>
      </div>
    </body></html>
    """


def render(cfg: dict) -> None:
    st.markdown("## Store Performance")
    st.markdown(
        "<p style='color:#64748B; font-size:14px; margin-bottom:20px;'>"
        "How close were our forecasts to reality across all 37 existing Cloverkey stores? "
        "Closest predictions shown first.</p>",
        unsafe_allow_html=True,
    )

    raw = _load()
    if raw.empty:
        st.info("Store performance data is not available in this deployment.")
        return

    raw = raw.sort_values("APE").reset_index(drop=True)

    avg_accuracy  = 100 - raw["APE"].mean()
    typical_error = raw["APE"].median()
    within        = int(raw["In_CI"].sum())
    total         = len(raw)

    st.html(_metrics_html(avg_accuracy, typical_error, within, total))
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    disp = raw[["Store", "Actual_Annual", "Predicted_Annual",
                "Conservative", "Optimistic", "APE", "In_CI"]].copy()

    in_ci_mask = disp["In_CI"].astype(bool)

    def _style_rows(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        styles.loc[in_ci_mask, :]  = "background-color: #F0FDF4"
        styles.loc[~in_ci_mask, :] = "background-color: #FEF2F2"
        return styles

    for col in ["Actual_Annual", "Predicted_Annual", "Conservative", "Optimistic"]:
        disp[col] = disp[col].map("${:,.0f}".format)
    disp["APE"]   = disp["APE"].map(lambda v: f"{v:.1f}% off")
    disp["In_CI"] = in_ci_mask.map({True: "Yes", False: "No"})

    disp = disp.rename(columns={
        "Store":            "Store",
        "Actual_Annual":    "Actual Revenue",
        "Predicted_Annual": "Our Forecast",
        "Conservative":     "Low Estimate",
        "Optimistic":       "High Estimate",
        "APE":              "How Far Off",
        "In_CI":            "Hit the Range?",
    })

    styled = disp.style.apply(_style_rows, axis=None)
    st.dataframe(styled, width="stretch", hide_index=True, height=660)

    st.caption(
        "Green rows = actual revenue landed between our low and high estimates. "
        "Red rows = actual revenue fell outside our estimated range. "
        "The 19 stores outside the range are mostly very small or specialty hospitals "
        "with unusual revenue patterns that are harder to predict."
    )
