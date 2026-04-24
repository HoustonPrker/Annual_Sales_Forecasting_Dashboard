import streamlit as st

from model import load_artifacts
from saved_forecasts import delete_forecast, list_forecasts, to_excel_bytes
from tabs import tab_info, tab_performance, tab_predictor

st.set_page_config(
    page_title="Gift Shop Revenue Forecast — Cloverkey",
    page_icon="🏥",
    layout="wide",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* Page background and container */
      .stApp { background-color: #F1F5F9; }
      .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1100px;
      }

      /* st.html embeds need transparent bg */
      iframe { background: transparent !important; }

      /* Clean sans-serif everywhere */
      html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        color: #1E293B;
      }

      /* Main content area white card */
      section[data-testid="stMain"] > div {
        background: #F8FAFC;
      }

      /* Sidebar */
      [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
      }
      [data-testid="stSidebar"] * { color: #1E293B !important; }

      /* Tab bar */
      .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0;
      }
      .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 22px;
        font-size: 14px;
        font-weight: 600;
        color: #64748B;
        background: transparent;
        border-radius: 6px 6px 0 0;
      }
      .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        background: #EFF6FF !important;
        border-bottom: 2px solid #2563EB;
      }

      /* Form container */
      [data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px 28px 20px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      }

      /* Input fields */
      .stNumberInput input, .stTextInput input {
        background: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        color: #1E293B !important;
        font-size: 15px !important;
      }
      .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
      }

      /* Selectbox */
      [data-baseweb="select"] > div {
        background: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        color: #1E293B !important;
      }

      /* Date input */
      .stDateInput input {
        background: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        color: #1E293B !important;
      }

      /* Form labels */
      .stNumberInput label, .stSelectbox label, .stDateInput label,
      .stTextInput label {
        font-weight: 600 !important;
        font-size: 13px !important;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }

      /* Caption / help text */
      .stCaption, small {
        color: #94A3B8 !important;
        font-size: 12px !important;
      }

      /* Primary submit button */
      .stFormSubmitButton > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 2px 6px rgba(37,99,235,0.30) !important;
        transition: background 0.15s;
      }
      .stFormSubmitButton > button:hover {
        background: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
      }

      /* Metric cards */
      [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      }
      [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #1E293B !important;
      }

      /* Section headings */
      h1 { color: #1E293B !important; font-weight: 800 !important; }
      h2 { color: #1E293B !important; font-weight: 700 !important; }
      h3 {
        color: #1E3A5F !important;
        font-weight: 700 !important;
        margin-top: 1.6rem !important;
        font-size: 18px !important;
      }
      h4 {
        color: #334155 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        margin-bottom: 4px !important;
      }

      /* Horizontal rule */
      hr { border-color: #E2E8F0 !important; margin: 1.4rem 0 !important; }

      /* Expander */
      [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
      }

      /* Dataframe */
      [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
      }

      /* ── Print styles ── */
      @media print {
        /* Hide everything that isn't the main content */
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stTabs [data-baseweb="tab-list"],
        .stAppDeployButton,
        footer,
        #MainMenu { display: none !important; }

        /* Remove gray page background */
        .stApp, html, body { background: white !important; }

        /* Remove padding so content fills the page */
        .block-container {
          padding: 0 !important;
          max-width: 100% !important;
        }

        /* Make the main content fill the full width */
        section[data-testid="stMain"] {
          width: 100% !important;
          margin: 0 !important;
          padding: 0 !important;
        }

        /* Hide the Save/Export/Print action buttons when printing */
        [data-testid="stHorizontalBlock"]:has(
          [data-testid="stButton"] [kind="secondary"]
        ) { display: none !important; }
      }

      /* Success/info banners */
      .stSuccess {
        background: #F0FDF4 !important;
        border-left: 4px solid #16A34A !important;
        color: #14532D !important;
        border-radius: 6px !important;
      }
      .stInfo {
        background: #EFF6FF !important;
        border-left: 4px solid #2563EB !important;
        color: #1E3A5F !important;
        border-radius: 6px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

artifacts = load_artifacts()
cfg, _, _, _, _, _, existing_df, _ = artifacts

st.sidebar.markdown(
    """
    <div style="padding: 8px 0;">
      <div style="font-size:20px; font-weight:800; color:#1E3A5F; margin-bottom:4px;">
        Cloverkey
      </div>
      <div style="font-size:13px; color:#64748B;">Gift Shop Revenue Forecast</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:12px; color:#94A3B8;'>"
    "Trained on 37 hospital gift shops · Updated 2025"
    "</div>",
    unsafe_allow_html=True,
)

# ── Saved forecasts panel ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:13px; font-weight:700; color:#1E3A5F; margin-bottom:6px;'>"
    "Saved Forecasts</p>",
    unsafe_allow_html=True,
)

saved = list_forecasts()

if not saved:
    st.sidebar.markdown(
        "<div style='font-size:12px; color:#94A3B8;'>No forecasts saved yet.<br>"
        "Generate a forecast and click <b>Save Forecast</b>.</div>",
        unsafe_allow_html=True,
    )
else:
    for fc in reversed(saved):
        with st.sidebar.expander(fc["hospital_name"], expanded=False):
            st.markdown(
                f"<div style='font-size:12px; color:#64748B; margin-bottom:6px;'>"
                f"Saved {fc['saved_at']}</div>",
                unsafe_allow_html=True,
            )
            lo, mid, hi = fc["conservative"], fc["accurate"], fc["optimistic"]
            st.markdown(
                f"<div style='font-size:13px;'>"
                f"<b style='color:#1E3A5F;'>${mid:,.0f}</b> most likely<br>"
                f"<span style='color:#94A3B8; font-size:11px;'>"
                f"${lo:,.0f} – ${hi:,.0f}</span></div>",
                unsafe_allow_html=True,
            )
            xlsx = to_excel_bytes([fc])
            safe_name = fc["hospital_name"].replace(" ", "_").replace("/", "-")
            st.download_button(
                label="📥 Download Excel",
                data=xlsx,
                file_name=f"{safe_name}_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch',
                key=f"dl_{fc['id']}",
            )
            if st.button("🗑 Delete", width='stretch', key=f"del_{fc['id']}"):
                delete_forecast(fc["id"])
                st.rerun()

    if len(saved) > 1:
        st.sidebar.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        all_xlsx = to_excel_bytes(saved)
        st.sidebar.download_button(
            label="📥 Download All as Excel",
            data=all_xlsx,
            file_name="all_forecasts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
        )

tab1, tab2, tab3 = st.tabs(["🏪  Revenue Forecast", "📊  Store Performance", "ℹ️  About This Model"])

with tab1:
    tab_predictor.render(artifacts)

with tab2:
    tab_performance.render(existing_df)

with tab3:
    tab_info.render(cfg)
