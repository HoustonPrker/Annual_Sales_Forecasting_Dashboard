"""Plotly chart builders — no Streamlit imports."""
from datetime import date

import pandas as pd
import plotly.graph_objects as go

FEATURE_LABELS = {
    "Gift_Shop_Outlier_Flag":        "Store Size Category",
    "Months_Since_Open":             "Store Maturity",
    "log_FTE":                       "Hospital Staff Size",
    "Affiliation_enc":               "Health System",
    "log_Staffed_Beds":              "Hospital Beds",
    "Time to Main Elevator Bank":    "Proximity to Elevator",
    "log_ADC":                       "Patient Volume",
    "Time to Cafeteria":             "Proximity to Cafeteria",
    "Month_Sine":                    "Seasonal Pattern",
    "Month_Cosine":                  "Seasonal Cycle",
    "Month_Fraction":                "Partial Opening Month",
}

# ── Shared layout defaults ────────────────────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#334155"),
    margin=dict(t=24, b=48, l=16, r=16),
)


def revenue_chart(monthly_revenue: list[float], monthly_dates: list[date], ci: dict) -> go.Figure:
    labels = [d.strftime("%b %Y") for d in monthly_dates]
    low  = [r * (1 + ci["r_low"])  for r in monthly_revenue]
    high = [r * (1 + ci["r_high"]) for r in monthly_revenue]

    fig = go.Figure()

    # Shaded confidence band (filled area)
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=high + low[::-1],
        fill="toself",
        fillcolor="rgba(37,99,235,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Forecast Range",
        hoverinfo="skip",
    ))
    # Outer boundary lines (subtle, no fill)
    for band, name, dash in [(low, "Low", "dot"), (high, "High", "dot")]:
        fig.add_trace(go.Scatter(
            x=labels, y=band,
            mode="lines",
            line=dict(color="rgba(37,99,235,0.25)", width=1, dash=dash),
            name=name, showlegend=False, hoverinfo="skip",
        ))
    # Main prediction line
    fig.add_trace(go.Scatter(
        x=labels,
        y=monthly_revenue,
        mode="lines+markers",
        name="Most Likely",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=7, color="#2563EB", line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x}</b><br><b>$%{y:,.0f}</b><extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT,
        height=360,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        yaxis=dict(
            title="Monthly Revenue",
            tickprefix="$", tickformat=",.0f",
            gridcolor="#F1F5F9", gridwidth=1,
            zeroline=False,
            tickfont=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=11),
        ),
    )
    return fig


def shap_impact_chart(
    features: list[str],
    shap_values,
    shap_base: float,
) -> go.Figure:
    """Clean horizontal bar chart of feature impacts — no waterfall, no base bar."""

    labels  = [FEATURE_LABELS.get(f, f) for f in features]
    vals    = [float(v) for v in shap_values]

    df = pd.DataFrame({"Feature": labels, "Impact": vals})
    df = df.sort_values("Impact", ascending=True)   # most positive at top (h chart is bottom-up)

    colors = ["#EF4444" if v > 0 else "#3B82F6" for v in df["Impact"]]
    hover  = [
        f"<b>{row.Feature}</b><br>{'Increases' if row.Impact > 0 else 'Decreases'} forecast"
        for row in df.itertuples()
    ]

    fig = go.Figure()

    # Reference line at zero
    fig.add_vline(x=0, line_width=1.5, line_color="#CBD5E1")

    # Bars
    fig.add_trace(go.Bar(
        x=df["Impact"],
        y=df["Feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        showlegend=False,
        width=0.55,
    ))

    # Legend proxy traces
    for label, color in [("Increases forecast", "#EF4444"), ("Decreases forecast", "#3B82F6")]:
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            orientation="h",
            marker=dict(color=color),
            name=label,
            showlegend=True,
        ))

    fig.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k != "margin"},
        height=380,
        margin=dict(t=16, b=56, l=180, r=24),
        barmode="overlay",
        xaxis=dict(
            title=dict(text="← Decreases forecast  |  Increases forecast →",
                       font=dict(size=12, color="#64748B")),
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=11),
            showticklabels=False,  # hide raw log values — not meaningful to executives
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=12, color="#334155"),
        ),
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig
