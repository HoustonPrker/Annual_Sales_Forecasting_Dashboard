from datetime import date

import streamlit as st
import streamlit.components.v1 as components

from charts import revenue_chart, shap_impact_chart
from model import month_fraction, predict_12_months, safe_log

_HELP = {
    "beds":      "Total staffed beds from AHD or CMS data.",
    "fte":       "Total hospital employees (full-time equivalents).",
    "adc":       "Average Daily Census — average number of inpatients per day.",
    "sqft":      "Interior square footage of the gift shop retail floor.",
    "affil":     "Parent health system. Affects expected visitor volume and spending patterns.",
    "elevator":  "Walking time in seconds from the gift shop entrance to the main elevator bank.",
    "cafeteria": "Walking time in seconds from the gift shop entrance to the main cafeteria.",
    "date":      "First day the store is expected to be open to the public.",
}


def render(artifacts: tuple) -> None:
    cfg        = artifacts[0]
    summary_df = artifacts[7]
    affil_opts = sorted(cfg["affiliation_lookup"].keys())

    st.markdown("## Revenue Forecast")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("forecast_inputs"):
        _section("Hospital Information")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            staffed_beds = st.number_input("Staffed Beds", min_value=1, max_value=2000, value=200, step=1)
            st.caption(_HELP["beds"])
            adc = st.number_input("Average Daily Census (ADC)", min_value=1, max_value=2000, value=150, step=1)
            st.caption(_HELP["adc"])
        with c2:
            fte = st.number_input("Total Employees (FTE)", min_value=1, max_value=20000, value=500, step=1)
            st.caption(_HELP["fte"])
            affiliation = st.selectbox(
                "Health System Affiliation",
                options=affil_opts,
                index=affil_opts.index("Other / New System"),
            )
            st.caption(_HELP["affil"])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _section("Gift Shop Details")
        c3, c4 = st.columns(2, gap="large")
        with c3:
            giftshop_sqft = st.number_input("Gift Shop Square Footage", min_value=100, max_value=5000, value=600, step=10)
            st.caption(_HELP["sqft"])
            dist_elevator = st.number_input("Distance to Elevator Bank (seconds)", min_value=0, max_value=300, value=30, step=1)
            st.caption(_HELP["elevator"])
        with c4:
            opening_date = st.date_input(
                "Planned Opening Date",
                value=date.today(),
                min_value=date(2000, 1, 1),
                max_value=date(2035, 12, 31),
            )
            st.caption(_HELP["date"])
            dist_cafeteria = st.number_input("Distance to Cafeteria (seconds)", min_value=0, max_value=1000, value=55, step=1)
            st.caption(_HELP["cafeteria"])

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Generate Forecast", type="primary", use_container_width=True)

    if not submitted:
        return

    with st.spinner("Calculating forecast…"):
        try:
            result = predict_12_months(artifacts, dict(
                staffed_beds=staffed_beds, fte=fte, adc=adc,
                giftshop_sqft=giftshop_sqft, affiliation=affiliation,
                dist_elevator=dist_elevator, dist_cafeteria=dist_cafeteria,
                opening_date=opening_date,
            ))
        except Exception as e:
            st.error("Forecast could not be generated. Please check your inputs and try again.")
            with st.expander("Error details"):
                st.exception(e)
            return

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _render_hero(result)
    _render_monthly_chart(result, cfg["confidence_intervals"])
    _render_impact(result, cfg)
    _render_comparable_stores(summary_df, adc, staffed_beds)
    _render_technical_details(staffed_beds, fte, adc, giftshop_sqft, opening_date, result)


# ── Section helpers ───────────────────────────────────────────────────────────

def _section(label: str) -> None:
    st.markdown(
        f"<p style='font-size:13px; font-weight:700; color:#64748B; text-transform:uppercase; "
        f"letter-spacing:.07em; margin:0 0 12px 0;'>{label}</p>",
        unsafe_allow_html=True,
    )


def _card_html(lo: float, mid: float, hi: float) -> str:
    return f"""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
      * {{ box-sizing:border-box; margin:0; padding:0; }}
      body {{ background:transparent; font-family:'Inter',sans-serif; }}
      .row {{
        display:flex; align-items:stretch; gap:0;
        background:#fff; border:1px solid #E2E8F0;
        border-radius:14px; overflow:hidden;
        box-shadow:0 2px 12px rgba(0,0,0,0.06);
      }}
      .cell {{
        flex:1; padding:24px 20px 20px; text-align:center;
      }}
      .cell.mid {{
        flex:1.35; border-left:1px solid #E2E8F0; border-right:1px solid #E2E8F0;
        background:#F8FAFF;
      }}
      .label {{
        font-size:11px; font-weight:700; letter-spacing:.09em;
        text-transform:uppercase; margin-bottom:10px;
      }}
      .amount {{ font-weight:800; line-height:1; }}
      .sub {{ font-size:11px; margin-top:8px; color:#94A3B8; }}\n    </style></head>
    <body>
      <div class="row">
        <div class="cell">
          <div class="label" style="color:#B45309;">Conservative</div>
          <div class="amount" style="font-size:28px;color:#92400E;">${lo:,.0f}</div>
          <div class="sub">Lower bound</div>
        </div>
        <div class="cell mid">
          <div class="label" style="color:#1D4ED8;">Most Likely</div>
          <div class="amount" style="font-size:40px;color:#1E3A5F;">${mid:,.0f}</div>
          <div class="sub">Best estimate</div>
        </div>
        <div class="cell">
          <div class="label" style="color:#15803D;">Optimistic</div>
          <div class="amount" style="font-size:28px;color:#14532D;">${hi:,.0f}</div>
          <div class="sub">Upper bound</div>
        </div>
      </div>
    </body></html>
    """


def _render_hero(result: dict) -> None:
    lo, mid, hi = result["conservative"], result["accurate"], result["optimistic"]

    st.markdown(
        "<p style='font-size:18px; font-weight:700; color:#1E3A5F; margin-bottom:14px;'>"
        "First-Year Revenue Projections</p>",
        unsafe_allow_html=True,
    )

    # components.html renders in a real iframe — guaranteed HTML rendering
    components.html(_card_html(lo, mid, hi), height=148, scrolling=False)

    # Summary sentence — dollar signs in HTML to avoid LaTeX interpretation
    st.markdown(
        f"Based on similar hospitals in our network, we expect this gift shop to generate "
        f"between <b>${lo:,.0f}</b> and <b>${hi:,.0f}</b> in its first year, "
        f"with a most likely outcome of <b>${mid:,.0f}</b>.",
        unsafe_allow_html=True,
    )
    _divider()


def _render_monthly_chart(result: dict, ci: dict) -> None:
    st.markdown(
        "<p style='font-size:18px; font-weight:700; color:#1E3A5F; margin-bottom:4px;'>"
        "Monthly Revenue Forecast</p>",
        unsafe_allow_html=True,
    )
    # Drop heavily-prorated leading months (< 50% of month 2) so the chart
    # starts from the first meaningful revenue month
    rev   = result["monthly_revenue"]
    dates = result["monthly_dates"]
    cutoff = rev[1] * 0.5 if len(rev) > 1 else 0
    trimmed_rev   = [r for r, d in zip(rev, dates) if r >= cutoff]
    trimmed_dates = [d for r, d in zip(rev, dates) if r >= cutoff]

    fig = revenue_chart(trimmed_rev, trimmed_dates, ci)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    _divider()


def _render_impact(result: dict, cfg: dict) -> None:
    st.markdown(
        "<p style='font-size:18px; font-weight:700; color:#1E3A5F; margin-bottom:4px;'>"
        "What's Driving This Forecast?</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Factors in red push the forecast higher. Factors in blue pull it lower. "
        "Longer bars mean stronger influence."
    )
    fig = shap_impact_chart(
        cfg["final_features"],
        result["shap_values"],
        cfg["shap_base_value"],
    )
    st.plotly_chart(fig, use_container_width=True)
    _divider()


def _render_comparable_stores(summary_df, adc: float, staffed_beds: float) -> None:
    st.markdown(
        "<p style='font-size:18px; font-weight:700; color:#1E3A5F; margin-bottom:4px;'>"
        "Similar Hospitals in Our Network</p>",
        unsafe_allow_html=True,
    )
    comp = summary_df.copy()
    comp["_dist"] = (
        (comp["log_adc"]  - safe_log(adc)).abs()
        + (comp["log_beds"] - safe_log(staffed_beds)).abs()
    )
    comp = comp.nsmallest(5, "_dist").copy()
    comp["Similarity"] = comp["_dist"].apply(lambda d: f"{max(0, round(100 - d * 28))}%")
    comp = comp[["Store", "Hospital Name", "annualized_revenue", "Similarity"]].rename(columns={
        "Store":              "Store #",
        "Hospital Name":      "Hospital",
        "annualized_revenue": "Annual Revenue",
    })
    comp["Annual Revenue"] = comp["Annual Revenue"].map("${:,.0f}".format)

    rows = "".join(
        f"<tr>"
        f"<td style='text-align:center'>{row['Store #']}</td>"
        f"<td style='text-align:left'>{row['Hospital']}</td>"
        f"<td style='text-align:center'>{row['Annual Revenue']}</td>"
        f"<td style='text-align:center'>{row['Similarity']}</td>"
        f"</tr>"
        for _, row in comp.iterrows()
    )
    st.markdown(
        f"""
        <table style="width:100%; border-collapse:collapse; font-size:14px;
                      font-family:Inter,Segoe UI,sans-serif; color:#1E293B;">
          <thead>
            <tr style="border-bottom:2px solid #E2E8F0;">
              <th style="text-align:center; padding:10px 12px; font-weight:600;
                         color:#64748B; font-size:12px; text-transform:uppercase;
                         letter-spacing:.05em;">Store #</th>
              <th style="text-align:left; padding:10px 12px; font-weight:600;
                         color:#64748B; font-size:12px; text-transform:uppercase;
                         letter-spacing:.05em;">Hospital</th>
              <th style="text-align:center; padding:10px 12px; font-weight:600;
                         color:#64748B; font-size:12px; text-transform:uppercase;
                         letter-spacing:.05em;">Annual Revenue</th>
              <th style="text-align:center; padding:10px 12px; font-weight:600;
                         color:#64748B; font-size:12px; text-transform:uppercase;
                         letter-spacing:.05em;">Similarity</th>
            </tr>
          </thead>
          <tbody>
            {rows.replace("<tr>", '<tr style="border-bottom:1px solid #F1F5F9;">')}
          </tbody>
        </table>
        <div style="height:8px"></div>
        """,
        unsafe_allow_html=True,
    )
    _divider()


def _render_technical_details(beds, fte, adc, sqft, opening_date, result):
    with st.expander("Technical Details", expanded=False):
        st.caption("Internal model inputs — for data team reference only.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Occupancy Rate",    f"{adc / beds:.3f}")
        c2.metric("log(Sq Ft)",        f"{safe_log(sqft):.3f}")
        c3.metric("Outlier Flag",      str(result["outlier_flag"]))
        c1.metric("log(Staffed Beds)", f"{safe_log(beds):.3f}")
        c2.metric("log(FTE)",          f"{safe_log(fte):.3f}")
        c3.metric("log(ADC)",          f"{safe_log(adc):.3f}")
        if opening_date.day > 1:
            frac = month_fraction(opening_date, opening_date)
            st.info(
                f"Opening on day {opening_date.day} — Month 1 revenue prorated to {frac:.1%} of a full month."
            )


def _divider() -> None:
    st.markdown(
        "<div style='height:1px; background:#E2E8F0; margin:20px 0 24px;'></div>",
        unsafe_allow_html=True,
    )
