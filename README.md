# Cloverkey Gift Shop Revenue Forecasting Dashboard

A machine learning–powered tool that predicts annual revenue for new hospital gift shop locations before they open. Built for the Cloverkey analytics team to replace the legacy Excel regression model with a more accurate, explainable, and scalable forecasting system.

## What It Does

Enter basic hospital information (beds, patient volume, gift shop size, hospital type, and a few layout measurements) and the model predicts:

- **Conservative estimate** — lower bound of expected annual revenue
- **Most likely estimate** — the model's best prediction
- **Optimistic estimate** — upper bound of expected annual revenue

Every prediction comes with a monthly revenue forecast chart, plain-English driver explanations, and confidence intervals derived from real prediction residuals.

## How Accurate Is It

| Metric | Excel Model (Previous) | ML Model (Current) |
|---|---|---|
| Average prediction error (MAPE) | 50% | 28% |
| Typical prediction error (MedAPE) | 31% | 25% |
| Stores where model wins | 11 of 37 | 26 of 37 |
| R² (variance explained) | 0.63 | 0.77 |
| Confidence intervals | None | Yes (P25/P75) |
| Prediction explainability | None | Yes (SHAP) |

Tested on 37 stores using GroupKFold cross-validation (every store predicted by a model that never saw that store's data). All metrics shown are out-of-sample.

## Dashboard Tabs

### Tab 1 — Revenue Forecast
Input form for new hospital prospects. Enter 9 data points, get an annual revenue projection with monthly breakdown and confidence ranges.

**Required inputs:**

| Input | Source | Description |
|---|---|---|
| Staffed Beds | AHD / CMS | Total staffed beds at the hospital |
| Average Daily Census (ADC) | AHD / CMS | Average patients in hospital per day |
| Health System Affiliation | Known at signing | Dropdown — select system or "Other / New System" |
| Hospital Type | AHD / CMS | Community, Specialty, or Academic |
| Payroll Deduction Available | Contract terms | Yes/No toggle |
| Gift Shop Square Footage | Floor plans | Retail selling floor area |
| Distance to Elevator Bank | Site visit / floor plans | Walking time in seconds |
| Distance to Cafeteria | Site visit / floor plans | Walking time in seconds |

Everything else is auto-computed: log transforms, occupancy rate, seasonal encoding, and 12-month forecast averaging.

### Tab 2 — Store Performance
Performance dashboard for all 37 existing Cloverkey gift shops showing actual vs. predicted revenue, prediction accuracy per store, and whether each store falls within the forecast confidence range.

### Tab 3 — About This Model
Plain-English explanation of how the model works, what it considers, and how it compares to the previous Excel-based method.

## Model Architecture

**Ensemble of 3 models, blended in log-space:**
- **LightGBM** (gradient boosted trees) — 20% weight
- **Ridge Regression** (L2 regularized) — 65% weight
- **Elastic Net** (L1+L2 regularized) — 15% weight

**13 features (ranked by SHAP importance):**

1. Patient Volume (log ADC)
2. Health System Affiliation (target encoded)
3. Hospital Beds (log staffed beds)
4. Store Maturity (months since opening)
5. Partial Opening Month (set to 1.0 in production)
6. Gift Shop Sq Ft (log)
7. Proximity to Elevator Bank (seconds)
8. Proximity to Cafeteria (seconds)
9. Seasonal Pattern (month cosine)
10. Seasonal Pattern (month sine)
11. Occupancy Rate (ADC / Beds)
12. Hospital Type (target encoded — Community / Specialty / Academic)
13. Payroll Deduction Available (yes/no)

**Training data:** 1,093 monthly revenue observations across 37 stores.

**Prediction method:** The model predicts monthly revenue in log-space for 12 consecutive months (averaged across a typical year of seasonality), then converts back to dollars and sums for the annual total. Confidence intervals are computed from empirical residuals (P25/P75 shifts in log-space), achieving 50% monthly coverage and 49% annual coverage on the training set.

## What Changed in v3

The current model architecture differs from the original prototype in several important ways:

- **Removed Gift_Shop_Outlier_Flag** — the auto-classified store size tier was creating prediction cliffs (e.g., $100K jump between 1,500 and 1,600 sq ft). Square footage is now used as a continuous input directly.
- **Removed log_FTE** — Total Employees was multicollinear with Beds and ADC, causing Ridge regression to assign a spurious negative coefficient. Beds and ADC together capture hospital size more cleanly.
- **Added Hospital Type and Payroll Deduction** as features.
- **Removed the Opening Date input** — the model now forecasts a "typical year" averaging full 12-month seasonality, which is what bidding scenarios actually need.

The new model is structurally simpler (13 features vs. 14), behaves correctly when inputs change (no cliffs, monotonic responses to size variables), and matches v2 accuracy without the architectural problems.

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
│   ├── shap_explainer.joblib       # SHAP TreeExplainer for prediction explanations
│   ├── model_config.json           # Blend weights, residual shifts, feature list
│   └── store_summary.csv           # Per-store actuals and predictions for Tab 2
├── affiliation_lookup.json         # Maps affiliation names to encoded values
├── hospital_type_lookup.json       # Maps hospital type to encoded value
├── requirements.txt
└── README.md
```

**Design principle:** `model.py` and `charts.py` have zero Streamlit imports. All prediction logic is pure Python/NumPy/Pandas, making it unit-testable and reusable outside the dashboard. Tab files own only their UI layout.

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

Model developed through a 12-phase process in Google Colab:

| Phase | Description | Outcome |
|---|---|---|
| 0 | Data Assembly | 1,093 rows, 55 columns, 37 stores |
| 1 | Exploratory Data Analysis | ADC top predictor (r=0.66), multicollinearity surfaced |
| 2 | Feature Engineering | 15 log transforms, target encoding |
| 3 | Cross-Validation Setup | 5-fold GroupKFold, manually balanced by revenue tier |
| 4 | Baseline Models | Single-feature OLS, default ensemble |
| 5 | Hyperparameter Tuning | Optuna (50+30+30 trials), blend optimization |
| 6 | Ensemble Blending | Weighted blend selected over stacking |
| 7 | SHAP Analysis | Identified flag overfit, multicollinearity issues |
| 8 | Feature Selection | Tested flag removal, FTE removal |
| 9 | Confidence Intervals | Residual-based P25/P75 |
| 10 | Architecture Iteration | Three model versions tested (with flag → no flag → no FTE) |
| 11 | Holdout Validation | 5 unseen stores tested |
| 12 | Dashboard | Streamlit app with 3 tabs |

**Key discoveries:**
- The Gift_Shop_Outlier_Flag feature was overfitting (high training accuracy, but caused prediction cliffs at deployment).
- Gradient-boosted classifiers and decision trees showed 100% training accuracy but only 32% leave-one-out accuracy on the flag — clear overfitting.
- Multicollinearity between FTE, Beds, and ADC was assigning a negative coefficient to FTE in linear models. Removing FTE eliminated the bug at no accuracy cost.
- Births and Payroll Deduction adoption rate were tested and rejected as predictors — both effects were confounded with hospital size.
- 28 of 40 candidate features had zero ElasticNet coefficient and were safely removed.

## Data Sources

| Source | Provides | Update Frequency |
|---|---|---|
| NCR Counterpoint (SSMS) | Monthly gift shop revenue, transaction counts | Real-time |
| American Hospital Directory (AHD) | Hospital beds, ADC, discharges, type | Annual (CMS cost reports) |
| Cloverkey Store Surveys | Walking distances, contract terms (PD) | Per store at opening |
| CMS Provider of Services | Hospital characteristics, certification | Annual |

## Retraining Schedule

| Timeframe | Action | Expected Impact |
|---|---|---|
| October 2026 | Quick retrain with first BSW stores (3–6 months data) | BSW-specific affiliation encoding |
| April 2027 | Full retrain (60–70 stores, 2026 CMS data) | MAPE improvement to ~20%, year-over-year features |
| Ongoing | Add each new store after 6+ months of data | Incremental accuracy improvement |

The entire pipeline (Google Colab notebook) carries forward — swap in updated data and re-run from Phase 0.

## Retraining Instructions

1. Update `Annual_Sales_Final.xlsx` with new stores and additional months of financial data
2. If new CMS hospital data is available, update hospital features
3. Open the Google Colab notebook and upload the updated Excel file
4. Run all cells from Phase 0 through the export block
5. Download the new `model_files_v3.zip`
6. Replace the contents of `model_files/` in this repo
7. Restart the Streamlit dashboard

No code changes required — the pipeline is parameterized to handle additional stores and features automatically.

## Known Limitations

- **37 training stores** — accuracy improves with more stores. Target: 70+ by early 2027.
- **New health systems** (BSW, Methodist, AdventHealth, etc.) use a default "Other" affiliation encoding until at least one store from that system has training data.
- **Specialty hospitals** (rehab, psych, children's) are underrepresented in training data. Predictions for these types carry more uncertainty — see Stores 106 and 133 in the Performance tab.
- **Operational quality is invisible to the model.** Stores with unusual layouts, contract restrictions, or in-lobby competition will deviate from predictions. The model assumes typical operational characteristics.
- **Hospital features are a 2024 snapshot.** Year-over-year changes (new beds, growing ADC) aren't captured until 2025 CMS data becomes available.
- **Annual MAPE of 28% means typical predictions are within ±25% of actual revenue.** For a hospital projected at $400K, actual revenue typically lands between $300K and $500K.

## Technical Requirements
Python >= 3.10
streamlit >= 1.30
scikit-learn == 1.5.2     # Must match training version
lightgbm == 4.3.0         # Must match training version
shap == 0.46.0
plotly >= 5.18
pandas >= 2.0
numpy == 1.26.4           # Must match training version
joblib == 1.4.2           # Must match training version

**Note:** Pinned versions must match the versions used during model training to avoid compatibility errors when loading the joblib model files.

## Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

The dashboard runs locally at `http://localhost:8501`.

## Authors

- **Houston P.** — Analytics, Cloverkey / Kelli's Gift Shops
- Model development: Google Colab (Python)
- Dashboard: Streamlit + Plotly

## License

Internal use only — Cloverkey proprietary.
