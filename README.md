# Cloverkey Gift Shop Revenue Forecasting Dashboard

A machine learning-powered tool that predicts annual revenue for new hospital gift shop locations before they open. Built for the Cloverkey analytics team to replace the legacy Excel regression model with a more accurate, explainable, and scalable forecasting system.

---

## What It Does

Enter basic hospital information (beds, employees, patient volume, gift shop size, and a few layout measurements) and the model predicts:

- **Conservative estimate** — lower bound of expected annual revenue
- **Most likely estimate** — the model's best prediction
- **Optimistic estimate** — upper bound of expected annual revenue

Every prediction comes with a monthly revenue forecast chart and a plain-English explanation of what's driving the prediction up or down.

---

## How Accurate Is It

| Metric | Excel Model (Previous) | ML Model (Current) |
|--------|----------------------|-------------------|
| Average prediction error | 50% | 25% |
| Typical store accuracy | 69% | 82% |
| Stores where model wins | 8 of 37 | 29 of 37 |
| Confidence intervals | None | Yes |
| Prediction explainability | None | Yes |

Tested on 37 stores using GroupKFold cross-validation (every store predicted by a model that never saw that store's data). Validated against 5 holdout stores with real revenue data.

---

## Dashboard Tabs

### Tab 1 — Revenue Forecast
Input form for new hospital prospects. Enter 8 data points, get an annual revenue projection with monthly breakdown and confidence ranges.

**Required inputs:**
| Input | Source | Description |
|-------|--------|-------------|
| Staffed Beds | AHD / CMS | Total staffed beds at the hospital |
| Total Employees (FTE) | AHD / CMS | Full-time equivalent employees |
| Average Daily Census (ADC) | AHD / CMS | Average patients in hospital per day |
| Gift Shop Square Footage | Floor plans | Retail selling floor area |
| Health System Affiliation | Known at signing | Dropdown — select system or "Other" |
| Distance to Elevator Bank | Site visit / floor plans | Walking time in seconds |
| Distance to Cafeteria | Site visit / floor plans | Walking time in seconds |
| Planned Opening Date | Business planning | Date picker |

Everything else is auto-computed: log transforms, occupancy rate, store size classification, seasonal encoding, and partial first month adjustment.

### Tab 2 — Store Performance
Performance dashboard for all 37 existing Cloverkey gift shops showing actual vs predicted revenue, prediction accuracy, and whether each store falls within the forecast range.

### Tab 3 — About This Model
Plain-English explanation of how the model works, what it considers, and how it compares to the previous Excel-based method.

---

## Model Architecture

**Ensemble of 3 models blended:**
- LightGBM (gradient boosted trees) — 25% weight
- Ridge Regression (L2 regularized) — 45% weight
- Elastic Net (L1+L2 regularized) — 30% weight

**11 features:**
1. Store Size Category (auto-classified from sq ft + occupancy rate)
2. Store Maturity (months since opening)
3. Hospital Staff Size (log FTE)
4. Health System Affiliation (target encoded)
5. Hospital Beds (log staffed beds)
6. Proximity to Elevator Bank (seconds)
7. Patient Volume (log ADC)
8. Proximity to Cafeteria (seconds)
9. Seasonal Pattern (month sine)
10. Seasonal Pattern (month cosine)
11. Partial Opening Month (fraction of first month open)

**Training data:** 1,093 monthly revenue observations across 37 stores (January 2021 — April 2026).

**Prediction method:** The model predicts monthly revenue in log-space for 12 consecutive months, then converts back to dollars and sums for the annual total. Confidence intervals are computed from empirical residuals (P25/P75 shifts in log-space), achieving 50% monthly coverage and 68% annual coverage.

---

## Project Structure

```
Annual_Sales_Forecasting_Dashboard/
├── app.py                          # Streamlit entry point — page config and tab routing
├── model.py                        # Prediction logic — zero Streamlit imports, pure Python
├── charts.py                       # Plotly chart functions — revenue chart, SHAP waterfall
├── tabs/
│   ├── __init__.py
│   ├── tab_predictor.py            # Tab 1 — New store forecast form and results
│   ├── tab_performance.py          # Tab 2 — Existing store performance table
│   └── tab_info.py                 # Tab 3 — Model documentation
├── model_files/
│   ├── prod_lgbm.joblib            # Trained LightGBM model
│   ├── prod_ridge.joblib           # Trained Ridge pipeline (Imputer → Scaler → Ridge)
│   ├── prod_enet.joblib            # Trained Elastic Net pipeline
│   ├── flag_classifier.joblib      # Gift Shop Outlier Flag classifier (GBM, 100% accuracy)
│   ├── shap_explainer.joblib       # SHAP TreeExplainer for prediction explanations
│   ├── model_config.json           # Blend weights, CI shifts, features, affiliation lookup
│   ├── store_summary.csv           # Training store data for comparable stores lookup
│   └── existing_predictions.csv    # OOF predictions for all 37 training stores
├── requirements.txt
└── README.md
```

**Design principle:** `model.py` and `charts.py` have zero Streamlit imports. All prediction logic is pure Python/NumPy/Pandas, making it unit-testable and reusable outside the dashboard. Tab files own only their UI layout.

---

## How Predictions Are Computed

```
User enters: Beds, FTE, ADC, Sq Ft, Affiliation, Distances, Opening Date
                                    │
                                    ▼
              ┌─────────────────────────────────────┐
              │  Auto-compute derived features:      │
              │  • Occupancy Rate = ADC / Beds       │
              │  • log(Beds), log(FTE), log(ADC)     │
              │  • Outlier Flag via classifier        │
              │  • Month_Sine, Month_Cosine           │
              │  • Month_Fraction for partial month   │
              │  • Affiliation encoding lookup        │
              └──────────────┬──────────────────────┘
                             │
                    Build 12 monthly rows
                    (one per forecast month)
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          LightGBM        Ridge       Elastic Net
          (25%)           (45%)         (30%)
              │              │              │
              └──────────────┼──────────────┘
                             │
                      Blend in log-space
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
              Conservative  Most    Optimistic
              (P25 shift)  Likely  (P75 shift)
                    │        │        │
                    └────────┼────────┘
                             │
                   expm1() → dollars
                   sum 12 months → annual
```

---

## Development History

This model was developed through a systematic 12-phase process:

| Phase | Description | Key Outcome |
|-------|-------------|-------------|
| 0 | Data Assembly | 1,093 rows, 54 columns, 37 stores |
| 1 | Exploratory Data Analysis | ADC identified as top predictor (r=0.66) |
| 2 | Feature Engineering | 15 log transforms, target encoding, binary flags |
| 3 | Cross-Validation Setup | 5-fold GroupKFold, manually balanced by revenue tier |
| 4 | Baseline Models | Gift Shop Flag alone gets 38% MAPE, R²=0.61 |
| 5 | Hyperparameter Tuning | Optuna (50+30+30 trials), blend optimization |
| 6 | — | Stacking replaced with simple weighted blend |
| 7 | SHAP Analysis | Gift Shop Flag is 4x more important than any other feature |
| 8 | Feature Selection | 39 features pruned to 11 with no accuracy loss |
| 9 | Confidence Intervals | Residual-based P25/P75, 50% monthly coverage |
| 10 | — | Model B (diagnostic) deferred to v2 |
| 11 | Production System | predict_new_store() function, model export |
| 12 | Dashboard | Streamlit app with 3 tabs |

**Key discoveries during development:**
- Hospital cafeteria pricing does NOT suppress gift shop sales (competition score rejected after correlation analysis)
- Revenue is driven by foot traffic (transaction count varies 19x across stores, avg ticket only 2.5x)
- 10 of 39 features had zero SHAP importance and were safely removed
- Partial first months were corrupting the maturity curve — Month_Fraction feature fixed this
- Store size tier (Gift_Shop_Outlier_Flag) can be auto-assigned with 100% accuracy from square footage + occupancy rate

---

## Data Sources

| Source | What It Provides | Update Frequency |
|--------|-----------------|------------------|
| NCR Counterpoint (SSMS) | Monthly gift shop revenue, transaction counts | Real-time |
| American Hospital Directory (AHD) | Hospital beds, FTE, ADC, discharges, births | Annual (CMS cost reports) |
| Cloverkey Store Surveys | Walking distances, cafeteria/elevator proximity | Per store at opening |
| CMS Provider of Services | Hospital characteristics, certification | Annual |

---

## Retraining Schedule

| Timeframe | Action | Expected Impact |
|-----------|--------|-----------------|
| October 2026 | Quick retrain with first BSW stores (3-6 months data) | BSW-specific affiliation encoding |
| April 2027 | Full v2 retrain (60-70 stores, 2025 CMS data) | MAPE improvement to 18-22%, year-over-year features |
| Ongoing | Add each new store after 6+ months of data | Incremental accuracy improvement |

The entire pipeline (Google Colab notebook) carries forward — swap in updated data and re-run from Phase 0.

---

## Retraining Instructions

1. Update `Annual_Sales_Final.xlsx` with new stores and additional months of financial data
2. If new CMS hospital data is available, update hospital features
3. Open the Google Colab notebook and upload the updated Excel file
4. Run all cells from Phase 0 through the export block
5. Download the new `model_export.zip`
6. Replace the contents of `model_files/` in this repo
7. Restart the Streamlit dashboard

No code changes required — the pipeline is parameterized to handle additional stores and features automatically.

---

## Known Limitations

- **37 training stores** — accuracy improves with more stores. Target: 70+ by early 2027.
- **New health systems** (BSW, Methodist, AdventHealth) use a default affiliation encoding until their stores have revenue data.
- **Month 1 predictions** tend to overestimate because opening-month behavior varies by store.
- **Store 149 (BSW Centennial)** is a known outlier — model predicts $172K but actual is $81K. This is an operational issue (low foot traffic), not a model failure.
- **Specialty hospitals** (children's, cancer, rehab) are underrepresented in training data (3 of 37 stores). Predictions for these types carry more uncertainty.
- **Hospital features are a 2024 snapshot.** Year-over-year changes (new beds, growing ADC) aren't captured until 2025 CMS data becomes available.

---

## Technical Requirements

```
Python >= 3.10
streamlit >= 1.30
scikit-learn == 1.6.1    # Must match training version
lightgbm >= 4.0
shap >= 0.45
plotly >= 5.18
pandas >= 2.0
numpy >= 1.24
joblib >= 1.3
```

**Note:** scikit-learn version is pinned to 1.6.1 to match the version used during model training. Using a different version may cause `_fill_dtype` compatibility errors when loading the Ridge and Elastic Net pipelines.

---

## Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

The dashboard runs locally at `http://localhost:8501`.

---

## Authors

- **Houston P.** — Analytics, Cloverkey / Kelli's Gift Shops
- Model development: Google Colab (Python)
- Dashboard: Streamlit + Plotly

---

## License

Internal use only — Cloverkey proprietary.