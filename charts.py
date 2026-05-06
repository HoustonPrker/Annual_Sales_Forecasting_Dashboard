"""Plotly chart builders — no Streamlit imports."""
import math

import pandas as pd
import plotly.graph_objects as go

FEATURE_LABELS = {
    "Months_Since_Open":             "Store Maturity",
    "Affiliation_enc":               "Health System",
    "log_Staffed_Beds":              "Hospital Beds",
    "Time to Main Elevator Bank":    "Proximity to Elevator",
    "log_ADC":                       "Patient Volume",
    "Time to Cafeteria":             "Proximity to Cafeteria",
    "Month_Sine":                    "Seasonal Pattern",
    "Month_Cosine":                  "Seasonal Cycle",
    "Month_Fraction":                "Partial Opening Month",
    "log_Giftshop_Sq_Ft":            "Gift Shop Square Footage",
    "Occupancy_Rate":                "Hospital Occupancy Rate",
    "Hospital_Type_enc":             "Hospital Type",
    "Payroll Ded":                   "Payroll Deduction",
}

# ── Shared layout defaults ────────────────────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#334155"),
    margin=dict(t=24, b=48, l=16, r=16),
)


def revenue_chart(monthly_revenue: list[float], labels: list[str], residual_shifts: dict) -> go.Figure:
    low  = [r * (1 + residual_shifts["conservative"]) for r in monthly_revenue]
    high = [r * (1 + residual_shifts["optimistic"])   for r in monthly_revenue]

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
    import os, pandas as pd
    _csv = os.path.join(os.path.dirname(__file__), "model_files", "store_summary.csv")
    try:
        _df = pd.read_csv(_csv)[["Store", "Actual_Annual", "Predicted_Annual"]].copy()
        _df.columns = ["Store", "Actual", "ML_Predicted"]
    except (FileNotFoundError, KeyError):
        _df = pd.DataFrame({"Store": [], "Actual": [], "ML_Predicted": []})

    # Excel baseline predictions (unchanged reference for comparison)
    _excel = {
        101: 629392, 102: 405601, 103: 365592, 104: 440519, 105: 321790,
        106: 543024, 108: 327904, 109: 411076, 110: 359472, 111: 313408,
        112: 355413, 113: 201598, 114: 277351, 115: 579626, 118: 1265759,
        119: 331651, 120: 267622, 122: 266400, 123: 537817, 124: 1069438,
        125: 542055, 126: 388599, 127: 504811, 128: 567930, 129: 383998,
        130: 370973, 131: 406039, 132: 410552, 133: 238332, 134: 330571,
        135: 854461, 136: 964357, 140: 693961, 141: 304131, 142: 320169,
        144: 427139, 145: 235109,
    }
    _df["Excel_Predicted"] = _df["Store"].map(_excel)

    data = {
        "Store":           _df["Store"].tolist(),
        "Actual":          _df["Actual"].tolist(),
        "ML_Predicted":    _df["ML_Predicted"].tolist(),
        "Excel_Predicted": _df["Excel_Predicted"].tolist(),
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
