import os

import pandas as pd
import streamlit as st

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "model_files", "store_summary.csv")


def _load() -> pd.DataFrame:
    try:
        return pd.read_csv(_CSV_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


def _summary_html(stores_within_10pct: int, total: int, avg_dollar_off: float, best_store: str, worst_store: str) -> str:
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
          <div class="label" style="color:#1D4ED8;">Spot-On Forecasts</div>
          <div class="value" style="color:#1E3A5F;">{stores_within_10pct} <span style="font-size:20px; color:#64748B;">of {total}</span></div>
          <div class="sub">Stores where we landed within 10% of actual revenue</div>
        </div>
        <div class="card">
          <div class="label" style="color:#15803D;">Average Miss</div>
          <div class="value" style="color:#14532D;">${avg_dollar_off:,.0f}</div>
          <div class="sub">How far off our forecast was, on average</div>
        </div>
        <div class="card">
          <div class="label" style="color:#B45309;">Best vs. Hardest</div>
          <div class="value" style="color:#92400E; font-size:22px; line-height:1.3;">#{best_store}</div>
          <div class="sub">Best predicted store &nbsp;·&nbsp; Hardest: #{worst_store}</div>
        </div>
      </div>
    </body></html>
    """


def render(cfg: dict) -> None:
    st.markdown("## Store Performance")
    st.markdown(
        "<p style='color:#64748B; font-size:14px; margin-bottom:20px;'>"
        "How close were our revenue forecasts to what each of our 37 stores actually earned? "
        "Best predictions shown first.</p>",
        unsafe_allow_html=True,
    )

    raw = _load()
    if raw.empty:
        st.info("Store performance data is not available in this deployment.")
        return

    raw = raw.sort_values("APE").reset_index(drop=True)

    # Dollar difference (absolute)
    raw["Dollar_Off"] = (raw["Predicted_Annual"] - raw["Actual_Annual"]).abs()

    stores_within_10pct = int((raw["APE"] <= 10).sum())
    avg_dollar_off      = raw["Dollar_Off"].mean()
    best_store          = str(int(raw.iloc[0]["Store"]))
    worst_store         = str(int(raw.iloc[-1]["Store"]))

    st.html(_summary_html(stores_within_10pct, len(raw), avg_dollar_off, best_store, worst_store))
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Build display table
    disp = raw[["Store", "Actual_Annual", "Predicted_Annual", "Dollar_Off",
                "Conservative", "Optimistic", "In_CI"]].copy()

    in_ci_mask = disp["In_CI"].astype(bool)

    # Direction of miss
    direction = (raw["Predicted_Annual"] - raw["Actual_Annual"]).apply(
        lambda v: "We over-forecast" if v > 0 else "We under-forecast"
    )
    disp["Verdict"] = direction

    def _style_rows(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        styles.loc[in_ci_mask, :]  = "background-color: #F0FDF4"
        styles.loc[~in_ci_mask, :] = "background-color: #FFF7ED"
        return styles

    for col in ["Actual_Annual", "Predicted_Annual", "Conservative", "Optimistic"]:
        disp[col] = disp[col].map("${:,.0f}".format)
    disp["Dollar_Off"] = disp["Dollar_Off"].map("${:,.0f}".format)
    disp["In_CI"] = in_ci_mask.map({True: "✓ Yes", False: "✗ No"})

    disp = disp.rename(columns={
        "Store":             "Store",
        "Actual_Annual":     "Actual Revenue",
        "Predicted_Annual":  "Our Forecast",
        "Dollar_Off":        "$ Difference",
        "Conservative":      "Low Estimate",
        "Optimistic":        "High Estimate",
        "In_CI":             "Actual in Range?",
        "Verdict":           "Direction",
    })

    styled = disp.style.apply(_style_rows, axis=None)
    st.dataframe(styled, width="stretch", hide_index=True, height=660)

    st.caption(
        "Green rows: actual revenue fell inside our estimated range (Low → High Estimate). "
        "Orange rows: actual revenue landed outside our range. "
        "$ Difference shows how far our central forecast was from reality in dollar terms."
    )
