import streamlit as st

from charts import FEATURE_LABELS, revenue_chart, shap_impact_chart
from model import predict_12_months, safe_log
from saved_forecasts import save_forecast, single_forecast_excel_bytes

_HELP = {
    "beds":        "Total staffed beds from AHD or CMS data.",
    "adc":         "Average Daily Census — average number of inpatients per day.",
    "sqft":        "Interior square footage of the gift shop retail floor.",
    "hosp_type":   "Hospital classification — Community, Specialty, or Academic.",
    "payroll_ded": "Whether the hospital offers payroll deduction for gift shop purchases.",
    "elevator":    "Walking time in seconds from the gift shop entrance to the main elevator bank.",
    "cafeteria":   "Walking time in seconds from the gift shop entrance to the main cafeteria.",
}

_HOSP_TYPES = ["Community", "Specialty", "Academic"]


def render(artifacts: tuple) -> None:
    cfg        = artifacts[0]
    summary_df = artifacts[6]

    st.markdown("## Revenue Forecast")

    with st.form("forecast_inputs"):
        hospital_name = st.text_input(
            "Hospital Name",
            placeholder="e.g. St. Mary's Medical Center",
            help="Used to label saved and exported forecasts.",
        )

        _section("Hospital Information")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            staffed_beds = st.number_input("Staffed Beds", min_value=1, max_value=2000, value=200, step=1)
            st.caption(_HELP["beds"])
        with c2:
            adc = st.number_input("Avg Daily Census (ADC)", min_value=1, max_value=2000, value=150, step=1)
            st.caption(_HELP["adc"])
        with c3:
            hospital_type = st.selectbox("Hospital Type", options=_HOSP_TYPES, index=0)
            st.caption(_HELP["hosp_type"])

        payroll_ded_bool = st.toggle("Payroll Deduction Available", value=True)
        st.caption(_HELP["payroll_ded"])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _section("Gift Shop Details")
        c4, c5, c6 = st.columns(3, gap="large")
        with c4:
            giftshop_sqft = st.number_input("Square Footage", min_value=100, max_value=5000, value=600, step=10)
            st.caption(_HELP["sqft"])
        with c5:
            dist_elevator = st.number_input("Distance to Elevator (sec)", min_value=0, max_value=300, value=30, step=1)
            st.caption(_HELP["elevator"])
        with c6:
            dist_cafeteria = st.number_input("Distance to Cafeteria (sec)", min_value=0, max_value=1000, value=55, step=1)
            st.caption(_HELP["cafeteria"])

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Generate Forecast", type="primary", width='stretch')

    if submitted:
        payroll_ded = 1 if payroll_ded_bool else 0
        inputs = dict(
            staffed_beds=staffed_beds, adc=adc,
            giftshop_sqft=giftshop_sqft, affiliation="Other / New System",
            hospital_type=hospital_type, payroll_ded=payroll_ded,
            dist_elevator=dist_elevator, dist_cafeteria=dist_cafeteria,
        )
        with st.spinner("Calculating forecast…"):
            try:
                result = predict_12_months(artifacts, inputs)
            except Exception as e:
                st.error("Forecast could not be generated. Please check your inputs and try again.")
                with st.expander("Error details"):
                    st.exception(e)
                return

        shap_drivers = {
            FEATURE_LABELS.get(f, f): float(v)
            for f, v in zip(cfg["features"], result["shap_values"])
        }
        st.session_state["last_forecast"] = {
            "hospital_name": hospital_name,
            "inputs":        inputs,
            "result":        result,
            "adc":           adc,
            "staffed_beds":  staffed_beds,
            "shap_drivers":  shap_drivers,
            "shap_base":     float(cfg.get("shap_base_value", 0.0)),
        }
        st.session_state.pop("forecast_saved", None)

    if "last_forecast" not in st.session_state:
        return

    fc            = st.session_state["last_forecast"]
    result        = fc["result"]
    inputs        = fc["inputs"]
    hospital_name = fc["hospital_name"]

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _render_print_header(hospital_name, inputs)
    _render_hero(result)
    _render_monthly_chart(result, cfg["residual_shifts"])
    _render_impact(result, cfg)
    _render_comparable_stores(summary_df, fc["adc"], fc["staffed_beds"])
    _render_technical_details(
        fc["inputs"]["staffed_beds"], fc["inputs"]["adc"],
        fc["inputs"]["giftshop_sqft"],
    )
    _render_actions(hospital_name, inputs, result,
                    fc.get("shap_drivers", {}), fc.get("shap_base", 0.0))


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


def _render_print_header(hospital_name: str, inputs: dict) -> None:
    label = hospital_name.strip() or "Unnamed Hospital"
    hosp_type  = inputs.get("hospital_type", "—")
    affil      = inputs.get("affiliation", "—")
    beds       = inputs.get("staffed_beds", "—")
    adc        = inputs.get("adc", "—")
    sqft       = inputs.get("giftshop_sqft", "—")
    elevator   = inputs.get("dist_elevator", "—")
    cafeteria  = inputs.get("dist_cafeteria", "—")
    payroll    = "Yes" if inputs.get("payroll_ded") else "No"

    st.markdown(
        f"""
        <div class="print-header">
          <div style="font-size:22px; font-weight:800; color:#1E3A5F; margin-bottom:2px;">{label}</div>
          <div style="font-size:13px; color:#64748B; margin-bottom:16px;">Gift Shop Revenue Forecast — Cloverkey</div>
          <table style="width:100%; border-collapse:collapse; font-size:13px; color:#334155;">
            <tr>
              <td style="padding:4px 16px 4px 0;"><b>Hospital Type</b></td><td style="padding:4px 24px 4px 0;">{hosp_type}</td>
              <td style="padding:4px 16px 4px 0;"><b>Health System</b></td><td style="padding:4px 0;">{affil}</td>
            </tr>
            <tr>
              <td style="padding:4px 16px 4px 0;"><b>Staffed Beds</b></td><td style="padding:4px 24px 4px 0;">{beds:,}</td>
              <td style="padding:4px 16px 4px 0;"><b>Avg Daily Census (ADC)</b></td><td style="padding:4px 0;">{adc:,}</td>
            </tr>
            <tr>
              <td style="padding:4px 16px 4px 0;"><b>Gift Shop Sq Ft</b></td><td style="padding:4px 24px 4px 0;">{sqft:,}</td>
              <td style="padding:4px 16px 4px 0;"><b>Payroll Deduction</b></td><td style="padding:4px 0;">{payroll}</td>
            </tr>
            <tr>
              <td style="padding:4px 16px 4px 0;"><b>Distance to Elevator</b></td><td style="padding:4px 24px 4px 0;">{elevator}s walk</td>
              <td style="padding:4px 16px 4px 0;"><b>Distance to Cafeteria</b></td><td style="padding:4px 0;">{cafeteria}s walk</td>
            </tr>
          </table>
          <div style="height:1px; background:#E2E8F0; margin:14px 0;"></div>
        </div>
        <style>
          /* On screen: only show hospital name, hide the full input table */
          .print-header table {{ display: none; }}
          .print-header > div:nth-child(2) {{ display: none; }}
          /* On print: show everything */
          @media print {{
            .print-header table {{ display: table !important; }}
            .print-header > div:nth-child(2) {{ display: block !important; }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(result: dict) -> None:
    lo, mid, hi = result["conservative"], result["accurate"], result["optimistic"]
    st.markdown(
        "<p style='font-size:18px; font-weight:700; color:#1E3A5F; margin-bottom:14px;'>"
        "First-Year Revenue Projections</p>",
        unsafe_allow_html=True,
    )
    st.html(_card_html(lo, mid, hi))
    st.markdown(
        f"Based on similar hospitals in our network, we expect this gift shop to generate "
        f"between **${lo:,.0f}** and **${hi:,.0f}** in its first year, "
        f"with a most likely outcome of **${mid:,.0f}**."
    )
    _divider()


def _render_monthly_chart(result: dict, residual_shifts: dict) -> None:
    st.markdown(
        "<p style='font-size:18px; font-weight:700; color:#1E3A5F; margin-bottom:4px;'>"
        "Monthly Revenue Forecast</p>",
        unsafe_allow_html=True,
    )
    fig = revenue_chart(result["monthly_revenue"], result["monthly_labels"], residual_shifts)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
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
        cfg["features"],
        result["shap_values"],
        cfg.get("shap_base_value", 0.0),
    )
    st.plotly_chart(fig, width='stretch')
    _divider()


def _render_comparable_stores(summary_df, adc: float, staffed_beds: float) -> None:
    if summary_df.empty or "log_adc" not in summary_df.columns:
        return
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


def _render_technical_details(beds, adc, sqft):
    with st.expander("Technical Details", expanded=False):
        st.caption("Internal model inputs — for data team reference only.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Occupancy Rate",    f"{adc / beds:.3f}")
        c2.metric("log(Sq Ft)",        f"{safe_log(sqft):.3f}")
        c3.metric("log(Staffed Beds)", f"{safe_log(beds):.3f}")
        c1.metric("log(ADC)",          f"{safe_log(adc):.3f}")


def _render_actions(
    hospital_name: str, inputs: dict, result: dict,
    shap_drivers: dict, shap_base: float,
) -> None:
    label = hospital_name.strip() or "Unnamed Hospital"
    st.markdown('<div class="forecast-actions-section">', unsafe_allow_html=True)
    _divider()
    st.markdown(
        "<p style='font-size:18px; font-weight:700; color:#1E3A5F; margin-bottom:10px;'>"
        "Save or Export This Forecast</p>",
        unsafe_allow_html=True,
    )

    col_save, col_dl, col_print, col_spacer = st.columns([1, 1, 1, 1], gap="small")

    with col_save:
        already_saved = st.session_state.get("forecast_saved", False)
        btn_label = "Saved!" if already_saved else "Save Forecast"
        if st.button(btn_label, width='stretch', disabled=already_saved,
                     help="Save this forecast so you can find it later in the sidebar."):
            save_forecast(label, inputs, result, shap_drivers, shap_base)
            st.session_state["forecast_saved"] = True
            st.rerun()

    with col_dl:
        xlsx = single_forecast_excel_bytes(label, inputs, result, shap_drivers, shap_base)
        safe_name = label.replace(" ", "_").replace("/", "-")
        st.download_button(
            label="Download Excel",
            data=xlsx,
            file_name=f"{safe_name}_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
            help="Downloads a .xlsx file with projections, monthly breakdown, and revenue drivers.",
        )

    with col_print:
        if st.button("Print", width="stretch",
                     help="Open the browser print dialog for this page."):
            st.session_state["trigger_print"] = True

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.pop("trigger_print", False):
        import time
        nonce = int(time.time() * 1000)
        st.html(
            f"""<script>
            (function(){{
                /* nonce:{nonce} */
                var win = window.parent;
                // Scroll to bottom so Streamlit lazy-renders all off-screen content,
                // then scroll back to top, then wait for charts to finish rendering.
                win.scrollTo(0, win.document.body.scrollHeight);
                setTimeout(function() {{
                    win.scrollTo(0, 0);
                    setTimeout(function() {{
                        win.print();
                    }}, 1200);
                }}, 600);
            }})();
            </script>""",
            unsafe_allow_javascript=True,
        )


def _divider() -> None:
    st.markdown(
        "<div style='height:1px; background:#E2E8F0; margin:20px 0 24px;'></div>",
        unsafe_allow_html=True,
    )
