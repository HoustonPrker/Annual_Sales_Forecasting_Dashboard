import streamlit as st

from charts import prediction_accuracy_chart

_FEATURES = [
    ("Patient Volume",           "How many patients are in the hospital on a given day."),
    ("Hospital Beds",            "The total number of staffed beds — a proxy for hospital size."),
    ("Health System",            "Which health system operates the hospital."),
    ("Hospital Type",            "Whether the hospital is a Community, Specialty, or Academic institution."),
    ("Gift Shop Square Footage", "The size of the gift shop retail floor — larger stores tend to generate more revenue."),
    ("Hospital Occupancy Rate",  "Average Daily Census divided by staffed beds — a measure of how busy the hospital is."),
    ("Payroll Deduction",        "Whether employees can purchase from the gift shop via payroll deduction."),
    ("Store Maturity",           "How long the store has been open. Newer stores typically ramp up over time."),
    ("Proximity to Elevator",    "Walking time from the gift shop to the main elevator bank."),
    ("Proximity to Cafeteria",   "Walking time from the gift shop to the cafeteria."),
    ("Seasonal Pattern",         "Whether the current month falls in a high- or low-traffic season."),
    ("Seasonal Cycle",           "A second seasonal signal capturing cyclical revenue patterns."),
    ("Partial Opening Month",    "Accounts for stores that opened partway through a month."),
]



def _accuracy_stats_html() -> str:
    return """
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
      * { box-sizing:border-box; margin:0; padding:0; }
      body { background:transparent; font-family:'Inter',sans-serif; }
      .row { display:flex; gap:16px; }
      .card {
        flex:1; border-radius:12px; padding:20px 22px;
        border:1px solid transparent; text-align:center;
      }
      .old  { background:#FEF2F2; border-color:#FECACA; }
      .new  { background:#F0FDF4; border-color:#BBF7D0; }
      .impr { background:#EEF2FF; border-color:#C7D2FE; }
      .label { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px; }
      .old  .label { color:#991B1B; }
      .new  .label { color:#15803D; }
      .impr .label { color:#3730A3; }
      .big  { font-size:32px; font-weight:800; line-height:1; margin-bottom:6px; }
      .old  .big { color:#7F1D1D; }
      .new  .big { color:#14532D; }
      .impr .big { color:#312E81; }
      .sub  { font-size:12px; color:#6B7280; }
    </style></head>
    <body>
      <div class="row">
        <div class="card old">
          <div class="label">Previous Method (Excel)</div>
          <div class="big">50%</div>
          <div class="sub">Average Error</div>
        </div>
        <div class="card new">
          <div class="label">New Model (ML Ensemble)</div>
          <div class="big">25%</div>
          <div class="sub">Average Error</div>
        </div>
        <div class="card impr">
          <div class="label">Improvement</div>
          <div class="big">2.5×</div>
          <div class="sub">more accurate<br>Wins on 29 of 37 stores</div>
        </div>
      </div>
    </body></html>
    """


def render(cfg: dict) -> None:
    st.markdown("## About This Model")

    # ── Prediction accuracy comparison chart (top of tab) ─────────────────────
    _heading("Prediction Accuracy Comparison")
    st.markdown(
        "<p style='color:#64748B; font-size:13px; margin-top:-6px; margin-bottom:12px;'>"
        "Dots closer to the diagonal line = more accurate predictions</p>",
        unsafe_allow_html=True,
    )
    fig = prediction_accuracy_chart()
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.html(_accuracy_stats_html())
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    _heading("How It Works")
    st.markdown(
        "This model analyzes **13 characteristics** of a hospital and its gift shop to predict "
        "annual revenue. It was trained on data from **37 Cloverkey gift shops** across the "
        "country and combines three independent forecasting methods to produce a single, "
        "reliable estimate with a built-in confidence range."
    )

    # ── What it considers ─────────────────────────────────────────────────────
    _heading("What It Considers")
    for name, desc in _FEATURES:
        st.markdown(
            f"<div style='padding:10px 14px; margin-bottom:6px; background:#FFFFFF; "
            f"border:1px solid #E2E8F0; border-radius:8px;'>"
            f"<span style='font-weight:700; color:#1E3A5F;'>{name}</span>"
            f"<span style='color:#64748B; font-size:13px;'> — {desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── How accurate ─────────────────────────────────────────────────────────
    _heading("How Accurate Is It?")
    st.markdown(
        "On average, the model's predictions are **within 25% of actual revenue**. "
        "For **7 out of 10 stores**, the actual result falls within the forecast range. "
        "Accuracy is expected to improve significantly as Cloverkey reaches 70+ stores by early 2027."
    )

    # ── Questions ─────────────────────────────────────────────────────────────
    _heading("Questions?")
    st.markdown(
        "For questions about the methodology, to request a detailed technical review, "
        "or to report an unexpected forecast, contact the Cloverkey data team."
    )


def _heading(text: str) -> None:
    st.markdown(
        f"<p style='font-size:17px; font-weight:700; color:#1E3A5F; "
        f"margin:24px 0 10px; padding-bottom:8px; border-bottom:2px solid #EFF6FF;'>"
        f"{text}</p>",
        unsafe_allow_html=True,
    )
