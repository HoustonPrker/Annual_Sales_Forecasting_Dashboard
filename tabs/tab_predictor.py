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
    cfg = artifacts[0]

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
    _render_hero(result)
    _render_monthly_chart(result, cfg["residual_shifts"])
    _render_impact(result, cfg)
    _render_technical_details(
        fc["inputs"]["staffed_beds"], fc["inputs"]["adc"],
        fc["inputs"]["giftshop_sqft"],
    )
    _render_actions(hospital_name, inputs, result,
                    fc.get("shap_drivers", {}), fc.get("shap_base", 0.0), cfg)


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



def _render_technical_details(beds, adc, sqft):
    with st.expander("Technical Details", expanded=False):
        st.caption("Internal model inputs — for data team reference only.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Occupancy Rate",    f"{adc / beds:.3f}")
        c2.metric("log(Sq Ft)",        f"{safe_log(sqft):.3f}")
        c3.metric("log(Staffed Beds)", f"{safe_log(beds):.3f}")
        c1.metric("log(ADC)",          f"{safe_log(adc):.3f}")


def _build_print_html(
    hospital_name: str, inputs: dict, result: dict, cfg: dict,
) -> str:
    import math as _math
    from charts import FEATURE_LABELS

    label   = hospital_name.strip() or "Unnamed Hospital"
    lo, mid, hi = result["conservative"], result["accurate"], result["optimistic"]

    # ── Build HTML/CSS diverging SHAP chart ──────────────────────────────────
    features    = cfg["features"]
    shap_vals   = list(result["shap_values"])
    shap_base   = cfg.get("shap_base_value", 0.0)
    log_pred    = shap_base + sum(shap_vals)
    monthly_pred = _math.exp(log_pred)

    def _dollar_impact(v):
        d = monthly_pred * (1 - _math.exp(-v)) * 12
        sign = "+" if d >= 0 else "-"
        a = abs(d)
        if a >= 1_000_000:
            return f"{sign}${a/1_000_000:.1f}M/yr"
        if a >= 1_000:
            return f"{sign}${a/1_000:.0f}K/yr"
        return f"{sign}${a:.0f}/yr"

    rows = sorted(zip(features, shap_vals), key=lambda x: x[1])
    max_abs = max(abs(v) for _, v in rows) if rows else 1.0

    # SVG-based bars — SVG fill always prints; background-color is stripped by browsers
    chart_w  = 400   # px — wider to fill the page
    row_h    = 30    # px height per row
    bar_h    = 20    # px height of each bar
    bar_top  = (row_h - bar_h) // 2
    center   = chart_w // 2

    shap_rows_html = ""
    for feat, val in rows:
        name       = FEATURE_LABELS.get(feat, feat)
        bar_px     = max(3, int(abs(val) / max_abs * (center - 10)))
        dollar_lbl = _dollar_impact(val)
        is_pos     = val >= 0
        color      = "#EF4444" if is_pos else "#3B82F6"
        text_color = "#B91C1C" if is_pos else "#1D4ED8"

        rect_x = center + 3 if is_pos else center - 3 - bar_px
        svg = (
            f'<svg viewBox="0 0 {chart_w} {row_h}" width="100%" height="{row_h}" '
            f'preserveAspectRatio="none" style="display:block;overflow:visible;">'
            f'<line x1="{center}" y1="0" x2="{center}" y2="{row_h}" stroke="#CBD5E1" stroke-width="2"/>'
            f'<rect x="{rect_x}" y="{bar_top}" width="{bar_px}" height="{bar_h}" fill="{color}" rx="3"/>'
            f'</svg>'
        )

        shap_rows_html += (
            f'<tr>'
            f'<td style="text-align:right;padding:0 10px 0 0;font-size:12px;'
            f'color:#334155;white-space:nowrap;width:160px;height:{row_h}px;">{name}</td>'
            f'<td style="padding:0;">{svg}</td>'
            f'<td style="padding:0 0 0 10px;font-size:12px;font-weight:700;'
            f'color:{text_color};white-space:nowrap;width:90px;">{dollar_lbl}</td>'
            f'</tr>'
        )

    # Legend: use a table row so swatch and label stay on the same line reliably
    legend_html = (
        f'<table style="border-collapse:collapse;margin:0 auto 10px;">'
        f'<tr>'
        f'<td style="padding:0 6px 0 0;">'
        f'<svg width="14" height="14"><rect width="14" height="14" fill="#3B82F6" rx="2"/></svg>'
        f'</td>'
        f'<td style="font-size:12px;color:#334155;padding-right:20px;">Decreases forecast</td>'
        f'<td style="padding:0 6px 0 0;">'
        f'<svg width="14" height="14"><rect width="14" height="14" fill="#EF4444" rx="2"/></svg>'
        f'</td>'
        f'<td style="font-size:12px;color:#334155;">Increases forecast</td>'
        f'</tr>'
        f'</table>'
    )

    shap_section = f"""
    <div style="text-align:center;margin-bottom:8px;font-size:11px;color:#94A3B8;">
      &larr; Decreases forecast &nbsp;|&nbsp; Increases forecast &rarr;
    </div>
    {legend_html}
    <table style="border-collapse:collapse;width:100%;">{shap_rows_html}</table>
    """

    # ── Inputs table ─────────────────────────────────────────────────────────
    beds      = inputs.get("staffed_beds", "—")
    adc       = inputs.get("adc", "—")
    sqft      = inputs.get("giftshop_sqft", "—")
    elevator  = inputs.get("dist_elevator", "—")
    cafeteria = inputs.get("dist_cafeteria", "—")
    hosp_type = inputs.get("hospital_type", "—")
    affil     = inputs.get("affiliation", "—")
    payroll   = "Yes" if inputs.get("payroll_ded") else "No"

    def _fmt(v):
        try:
            return f"{int(v):,}"
        except Exception:
            return str(v)

    rows_html = f"""
      <tr class="trow">
        <td class="lbl">Hospital Type</td><td class="val">{hosp_type}</td>
        <td class="lbl">Health System</td><td class="val">{affil}</td>
      </tr>
      <tr class="trow">
        <td class="lbl">Staffed Beds</td><td class="val">{_fmt(beds)}</td>
        <td class="lbl">Avg Daily Census (ADC)</td><td class="val">{_fmt(adc)}</td>
      </tr>
      <tr class="trow">
        <td class="lbl">Gift Shop Sq Ft</td><td class="val">{_fmt(sqft)}</td>
        <td class="lbl">Payroll Deduction</td><td class="val">{payroll}</td>
      </tr>
      <tr class="trow">
        <td class="lbl">Distance to Elevator</td><td class="val">{elevator}s walk</td>
        <td class="lbl">Distance to Cafeteria</td><td class="val">{cafeteria}s walk</td>
      </tr>
    """

    css = """
      #ck-print { font-family:'Inter',sans-serif; color:#1E293B;
                  padding:32px 40px; background:#fff; box-sizing:border-box; }
      #ck-print * { box-sizing:border-box; }
      #ck-print h1 { font-size:28px; font-weight:800; color:#1E3A5F; margin-bottom:3px; }
      #ck-print .subtitle { font-size:13px; color:#64748B; margin-bottom:18px; }
      #ck-print table.inputs { width:100%; border-collapse:collapse; font-size:13px;
        margin-bottom:20px; border:1px solid #E2E8F0; }
      #ck-print tr.trow { border-bottom:1px solid #E2E8F0; }
      #ck-print tr.trow:last-child { border-bottom:none; }
      #ck-print tr.trow:nth-child(odd) { background:#F8FAFC; }
      #ck-print table.inputs td { padding:8px 14px; }
      #ck-print td.lbl { font-weight:700; color:#475569; padding-left:20px; width:22%;
        border-right:1px solid #E2E8F0; }
      #ck-print td.val { color:#1E293B; width:28%; border-right:1px solid #E2E8F0; }
      #ck-print td.val:last-child { border-right:none; }
      #ck-print .divider { height:1px; background:#E2E8F0; margin:18px 0; }
      #ck-print .section-title { font-size:15px; font-weight:700; color:#1E3A5F; margin:18px 0 10px; }
      #ck-print .cards { display:flex; border:1px solid #E2E8F0; border-radius:12px;
        overflow:hidden; margin-bottom:16px; }
      #ck-print .card { flex:1; padding:18px; text-align:center; }
      #ck-print .card.mid { flex:1.35; border-left:1px solid #E2E8F0;
        border-right:1px solid #E2E8F0; background:#F8FAFF; }
      #ck-print .clbl { font-size:10px; font-weight:700; text-transform:uppercase;
        letter-spacing:.09em; margin-bottom:7px; }
      #ck-print .cval { font-weight:800; line-height:1; }
      #ck-print .csub { font-size:11px; color:#94A3B8; margin-top:5px; }
      @media print {
        -webkit-print-color-adjust: exact; print-color-adjust: exact;
        @page { margin:0; size:letter portrait; }
        body > *:not(#ck-print) { display:none !important; }
        #ck-print { display:block !important; padding:24px 36px; }
      }
    """

    body = f"""
<h1>{label}</h1>
<div class="subtitle">Gift Shop Revenue Forecast &mdash; Cloverkey</div>
<table class="inputs">{rows_html}</table>
<div class="divider"></div>
<div class="section-title">First-Year Revenue Projections</div>
<div class="cards">
  <div class="card">
    <div class="clbl" style="color:#B45309;">Conservative</div>
    <div class="cval" style="font-size:26px;color:#92400E;">${lo:,.0f}</div>
    <div class="csub">Lower bound</div>
  </div>
  <div class="card mid">
    <div class="clbl" style="color:#1D4ED8;">Most Likely</div>
    <div class="cval" style="font-size:36px;color:#1E3A5F;">${mid:,.0f}</div>
    <div class="csub">Best estimate</div>
  </div>
  <div class="card">
    <div class="clbl" style="color:#15803D;">Optimistic</div>
    <div class="cval" style="font-size:26px;color:#14532D;">${hi:,.0f}</div>
    <div class="csub">Upper bound</div>
  </div>
</div>
{shap_section}
"""

    return css, body


def _render_actions(
    hospital_name: str, inputs: dict, result: dict,
    shap_drivers: dict, shap_base: float, cfg: dict,
) -> None:
    label = hospital_name.strip() or "Unnamed Hospital"
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
                     help="Opens the browser print dialog for this forecast."):
            st.session_state["trigger_print"] = True

    if st.session_state.pop("trigger_print", False):
        import base64, time
        css, body = _build_print_html(hospital_name, inputs, result, cfg)
        css_b64  = base64.b64encode(css.encode("utf-8")).decode("ascii")
        body_b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
        nonce    = int(time.time() * 1000)
        st.html(
            f"""<script>
            (function(){{
                /* nonce:{nonce} */
                var doc  = window.parent.document;
                var dec  = function(b64) {{
                    return decodeURIComponent(
                        atob(b64).split('').map(function(c) {{
                            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
                        }}).join('')
                    );
                }};

                // Remove any leftover overlay from a previous print
                var old = doc.getElementById('ck-print');
                if (old) old.remove();
                var oldStyle = doc.getElementById('ck-print-style');
                if (oldStyle) oldStyle.remove();

                // Inject styles
                var styleEl = doc.createElement('style');
                styleEl.id  = 'ck-print-style';
                styleEl.textContent = dec('{css_b64}');
                doc.head.appendChild(styleEl);

                // Inject content div (hidden on screen, shown only during print)
                var div = doc.createElement('div');
                div.id  = 'ck-print';
                div.style.display = 'none';
                div.innerHTML = dec('{body_b64}');
                doc.body.appendChild(div);

                // Print, then clean up
                setTimeout(function() {{
                    window.parent.print();
                    window.parent.addEventListener('afterprint', function cleanup() {{
                        div.remove();
                        styleEl.remove();
                        window.parent.removeEventListener('afterprint', cleanup);
                    }}, {{ once: true }});
                }}, 200);
            }})();
            </script>""",
            unsafe_allow_javascript=True,
        )


def _divider() -> None:
    st.markdown(
        "<div style='height:1px; background:#E2E8F0; margin:20px 0 24px;'></div>",
        unsafe_allow_html=True,
    )
