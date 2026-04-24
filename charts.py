"""Plotly chart builders — no Streamlit imports."""
import math
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


def prediction_accuracy_chart() -> go.Figure:
    """Scatter plot comparing ML model vs Excel model predictions against actual revenue."""

    data = {
        "Store": [101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 140, 141, 142, 144, 145],
        "Actual":       [1115053, 201567, 422939, 316316, 149033, 127107, 626383, 644790, 250150, 327023, 295269, 164401, 168665, 535015, 586053, 189010, 118555, 269363, 453540, 1381525, 876269, 285720, 474422, 632755, 406451, 268286, 350661, 331315, 81310, 278446, 940192, 1787826, 1089915, 230884, 359008, 298794, 209200],
        "ML_Predicted": [1275494, 246678, 341895, 292731, 131784, 239343, 506702, 463959, 260255, 255706, 278255, 172364, 244588, 443219, 794249, 149308, 145480, 251740, 337716, 1278894, 650342, 302509, 508583, 490675, 228674, 240915, 378608, 266788, 109838, 183607, 526642, 1483753, 646844, 202551, 320344, 328468, 174065],
        "Excel_Predicted": [629392, 405601, 365592, 440519, 321790, 543024, 327904, 411076, 359472, 313408, 355413, 201598, 277351, 579626, 1265759, 331651, 267622, 266400, 537817, 1069438, 542055, 388599, 504811, 567930, 383998, 370973, 406039, 410552, 238332, 330571, 854461, 964357, 693961, 304131, 320169, 427139, 235109],
    }

    max_val = 2_000_000
    tick_vals = [0, 500_000, 1_000_000, 1_500_000, 2_000_000]
    tick_text = ["$0", "$500K", "$1.0M", "$1.5M", "$2.0M"]

    fig = go.Figure()

    # ±20% shaded band around perfect prediction line
    band_x = [0, max_val]
    fig.add_trace(go.Scatter(
        x=band_x + band_x[::-1],
        y=[v * 1.2 for v in band_x] + [v * 0.8 for v in band_x[::-1]],
        fill="toself",
        fillcolor="rgba(180,180,180,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Within 20%",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Perfect prediction diagonal
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(color="#9CA3AF", width=1.5, dash="dash"),
        name="Perfect Prediction",
        hoverinfo="skip",
    ))

    # Excel model dots — diamond shape
    fig.add_trace(go.Scatter(
        x=data["Actual"],
        y=data["Excel_Predicted"],
        mode="markers",
        name="Previous Method (Excel) ◆",
        marker=dict(
            color="#E24B4A", size=10, opacity=0.75,
            symbol="diamond",
            line=dict(color="#C03130", width=1),
        ),
        customdata=list(zip(data["Store"], data["Excel_Predicted"], data["Actual"])),
        hovertemplate=(
            "<b>Store %{customdata[0]}</b><br>"
            "Actual: $%{x:,.0f}<br>"
            "Excel Predicted: $%{customdata[1]:,.0f}<extra></extra>"
        ),
    ))

    # ML model dots — circle shape
    fig.add_trace(go.Scatter(
        x=data["Actual"],
        y=data["ML_Predicted"],
        mode="markers",
        name="New Model (ML) ●",
        marker=dict(
            color="#534AB7", size=10, opacity=0.75,
            symbol="circle",
            line=dict(color="#3B329A", width=1),
        ),
        customdata=list(zip(data["Store"], data["ML_Predicted"], data["Actual"])),
        hovertemplate=(
            "<b>Store %{customdata[0]}</b><br>"
            "Actual: $%{x:,.0f}<br>"
            "ML Predicted: $%{customdata[1]:,.0f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k != "margin"},
        height=480,
        margin=dict(t=40, b=60, l=80, r=24),
        legend=dict(
            orientation="h", y=-0.14, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title="Actual Annual Revenue",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[0, max_val],
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Predicted Annual Revenue",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[0, max_val],
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=11),
        ),
    )
    return fig


def _fmt_dollar_impact(dollars: float) -> str:
    """Format a dollar impact value as +$XXK / -$XXK / +$X.XM etc."""
    sign = "+" if dollars >= 0 else "-"
    abs_val = abs(dollars)
    if abs_val >= 1_000_000:
        return f"{sign}${abs_val/1_000_000:.1f}M/yr"
    if abs_val >= 1_000:
        return f"{sign}${abs_val/1_000:.0f}K/yr"
    return f"{sign}${abs_val:.0f}/yr"


def shap_impact_chart(
    features: list[str],
    shap_values,
    shap_base: float,
) -> go.Figure:
    """Horizontal bar chart of feature impacts with annualized dollar labels."""

    labels = [FEATURE_LABELS.get(f, f) for f in features]
    vals   = [float(v) for v in shap_values]

    # Compute annualized dollar impact for each feature.
    # Model predicts log(monthly_revenue); base + sum(vals) = log(month-6 prediction).
    # Marginal dollar impact of feature i = monthly_pred × (1 - exp(-shap_i)) × 12.
    log_pred   = shap_base + sum(vals)
    monthly_pred = math.exp(log_pred)
    dollar_impacts = [monthly_pred * (1 - math.exp(-v)) * 12 for v in vals]

    df = pd.DataFrame({
        "Feature":       labels,
        "Impact":        vals,
        "DollarImpact":  dollar_impacts,
    })
    df = df.sort_values("Impact", ascending=True)

    colors     = ["#EF4444" if v > 0 else "#3B82F6" for v in df["Impact"]]
    text_labels = [_fmt_dollar_impact(d) for d in df["DollarImpact"]]
    text_colors = ["#B91C1C" if v > 0 else "#1D4ED8" for v in df["Impact"]]

    hover = [
        f"<b>{row.Feature}</b><br>"
        f"{'Increases' if row.Impact > 0 else 'Decreases'} forecast<br>"
        f"Annual impact: {_fmt_dollar_impact(row.DollarImpact)}"
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
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11, color=text_colors),
        cliponaxis=False,
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
        margin=dict(t=16, b=56, l=180, r=110),
        barmode="overlay",
        xaxis=dict(
            title=dict(text="← Decreases forecast  |  Increases forecast →",
                       font=dict(size=12, color="#64748B")),
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=11),
            showticklabels=False,
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=12, color="#334155"),
        ),
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
        ),
        uniformtext=dict(mode="hide", minsize=9),
    )
    return fig
