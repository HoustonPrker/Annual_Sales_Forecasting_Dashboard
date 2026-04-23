import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _accuracy(ape: float) -> str:
    return f"{max(0.0, 100.0 - ape):.0f}%"


def _status(ape: float) -> str:
    if ape < 15:
        return "✅  On target"
    if ape <= 30:
        return "🟡  Acceptable"
    return "🔴  Wide miss"


def _summary_cards_html(avg_acc: float, within: int, total: int) -> str:
    pct = f"{within / total:.0%}"
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
          <div class="label" style="color:#15803D;">Average Prediction Accuracy</div>
          <div class="value" style="color:#14532D;">{avg_acc:.0f}%</div>
          <div class="sub">Across all 37 training stores</div>
        </div>
        <div class="card">
          <div class="label" style="color:#1D4ED8;">Stores Within Forecast Range</div>
          <div class="value" style="color:#1E3A5F;">{within} <span style="font-size:20px;color:#64748B;">of {total}</span></div>
          <div class="sub">{pct} fall within the predicted range</div>
        </div>
      </div>
    </body></html>
    """


def render(existing_df: pd.DataFrame) -> None:
    st.markdown("## Store Performance")
    st.markdown(
        "<p style='color:#64748B; font-size:14px; margin-bottom:20px;'>"
        "Actual vs. predicted revenue for all 37 stores used to train the model.</p>",
        unsafe_allow_html=True,
    )

    raw = existing_df.copy()
    avg_acc    = max(0.0, 100.0 - raw["ape"].mean())
    within     = int(
        ((raw["annualized_actual"] >= raw["conservative"])
         & (raw["annualized_actual"] <= raw["optimistic"])).sum()
    )
    total      = len(raw)

    components.html(_summary_cards_html(avg_acc, within, total), height=120, scrolling=False)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    disp = raw[["Store", "Hospital Name", "annualized_actual", "annualized_predicted", "ape"]].copy()
    disp["Accuracy"] = disp["ape"].apply(_accuracy)
    disp["Status"]   = disp["ape"].apply(_status)
    disp["annualized_actual"]    = disp["annualized_actual"].map("${:,.0f}".format)
    disp["annualized_predicted"] = disp["annualized_predicted"].map("${:,.0f}".format)
    disp = disp.drop(columns=["ape"]).rename(columns={
        "Store":                 "Store #",
        "Hospital Name":         "Hospital",
        "annualized_actual":     "Actual Annual Revenue",
        "annualized_predicted":  "Predicted Annual Revenue",
    })

    st.dataframe(disp, use_container_width=True, hide_index=True, height=660)
