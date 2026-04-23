import streamlit as st
import streamlit.components.v1 as components

_FEATURES = [
    ("Patient Volume",           "How many patients are in the hospital on a given day."),
    ("Hospital Beds",            "The total number of staffed beds — a proxy for hospital size."),
    ("Hospital Staff Size",      "Total number of employees across the hospital."),
    ("Health System",            "Which health system operates the hospital."),
    ("Store Size Category",      "Whether the gift shop is classified as a boutique, standard, or flagship store."),
    ("Store Maturity",           "How long the store has been open. Newer stores typically ramp up over time."),
    ("Proximity to Elevator",    "Walking time from the gift shop to the main elevator bank."),
    ("Proximity to Cafeteria",   "Walking time from the gift shop to the cafeteria."),
    ("Seasonal Pattern",         "Whether the current month falls in a high- or low-traffic season."),
    ("Seasonal Cycle",           "A second seasonal signal capturing cyclical revenue patterns."),
    ("Partial Opening Month",    "Accounts for stores that opened partway through a month."),
]


def _comparison_html() -> str:
    return """
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
      * { box-sizing:border-box; margin:0; padding:0; }
      body { background:transparent; font-family:'Inter',sans-serif; }
      .row { display:flex; gap:16px; }
      .card {
        flex:1; border-radius:12px; padding:22px 24px;
        border:1px solid transparent;
      }
      .old { background:#FEF2F2; border-color:#FECACA; }
      .new { background:#F0FDF4; border-color:#BBF7D0; }
      .badge {
        font-size:11px; font-weight:700; text-transform:uppercase;
        letter-spacing:.08em; margin-bottom:10px;
      }
      .old .badge { color:#991B1B; }
      .new .badge { color:#15803D; }
      .big { font-size:40px; font-weight:800; line-height:1; margin-bottom:8px; }
      .old .big { color:#7F1D1D; }
      .new .big { color:#14532D; }
      .desc { font-size:13px; color:#6B7280; }
    </style></head>
    <body>
      <div class="row">
        <div class="card old">
          <div class="badge">Previous Method</div>
          <div class="big">50%</div>
          <div class="desc">Average prediction error<br>
            <span style="font-size:12px;color:#9CA3AF;margin-top:4px;display:block;">
              Single system-wide average. No confidence ranges.
            </span>
          </div>
        </div>
        <div class="card new">
          <div class="badge">New Model</div>
          <div class="big">25%</div>
          <div class="desc">Average prediction error<br>
            <span style="font-size:12px;color:#9CA3AF;margin-top:4px;display:block;">
              Tailored to each hospital. Includes a forecast range.
            </span>
          </div>
        </div>
      </div>
    </body></html>
    """


def render(cfg: dict) -> None:
    st.markdown("## About This Model")

    # ── How it works ──────────────────────────────────────────────────────────
    _heading("How It Works")
    st.markdown(
        "This model analyzes **11 characteristics** of a hospital and its gift shop to predict "
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

    # ── Comparison ───────────────────────────────────────────────────────────
    _heading("New Model vs. Previous Method")
    components.html(_comparison_html(), height=148, scrolling=False)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.success(
        "The new model is **2.5× more accurate** than the previous method and provides a "
        "forecast range — a conservative estimate and an optimistic estimate — for every prediction."
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
