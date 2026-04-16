# 💧 Smart Water Leakage Detection using Big Data Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartwaterleakageanalysis.streamlit.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rushitkhakhkhar/Smart-Water-Leakage-Dashboard)

A professional-grade smart city analytics platform that detects water leakage, predicts household risk levels using **dual machine learning models** (Random Forest + XGBoost), and provides interactive real-time monitoring through a Streamlit dashboard.

### 🌐 Live Demo — Preview the Dashboard

| Platform | Households | Link |
|----------|-----------|------|
| 🤗 **Hugging Face** (Full) | 1,000 households · 4.32M records | [▶️ Open Full Dashboard](https://rushitkhakhkhar-smart-water-leakage-dashboard.hf.space) |
| ☁️ **Streamlit Cloud** (Lite) | 200 households · 864K records | [▶️ Open Lite Dashboard](https://smartwaterleakageanalysis.streamlit.app/) |

---

## 📋 Project Overview

Water leakage in urban distribution systems leads to significant water loss and infrastructure damage. This project analyzes **4.32M+ smart water meter records** across **1,000 households** to detect abnormal consumption patterns and predict high-risk leakage scenarios.

The system combines big data processing, advanced feature engineering, and ensemble machine learning to generate actionable insights for water management authorities.

---

## 🎯 Project Objectives

- Analyze large-scale smart water meter data (4.32M records)
- Detect abnormal water consumption patterns using spike ratio analysis
- Classify leakage severity (Normal → Critical Leak)
- Rank households by risk level using behavioral profiling
- Train and compare **Random Forest** and **XGBoost** classifiers for predictive risk assessment
- Provide interactive What-If analysis for behavioral simulation
- Build a comprehensive analytics dashboard with 11 pages of insights

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11 |
| **Big Data** | Apache Spark |
| **ML Models** | scikit-learn (Random Forest), XGBoost |
| **Dashboard** | Streamlit, Plotly |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Serialization** | Joblib |

---

## 🔄 Project Pipeline

```
Smart Meter Data → Spark Processing → Feature Engineering → Anomaly Detection
                                            ↓
                              Leakage Severity Classification
                                            ↓
                              Household Risk Scoring (Bayesian)
                                            ↓
                    ┌───────────────────────────────────────┐
                    │     Machine Learning Models           │
                    │  ┌─────────────┐  ┌─────────────┐    │
                    │  │Random Forest│  │   XGBoost   │    │
                    │  │ (Bagging)   │  │ (Boosting)  │    │
                    │  └─────────────┘  └─────────────┘    │
                    │         ↓ Ensemble Verdict ↓          │
                    └───────────────────────────────────────┘
                                            ↓
                         Interactive Streamlit Dashboard
```

---

## 🤖 Machine Learning Models

### Dual-Model Approach
The system trains **two complementary models** on household-level aggregated behavioral features:

| Model | Type | Trees | Key Strength |
|-------|------|-------|-------------|
| **Random Forest** | Bagging (parallel) | 100 | Robust to noise, stable feature importance |
| **XGBoost** | Boosting (sequential) | 150 | Higher accuracy via gradient optimization |

### Features Used (10 behavioral features — no target leakage)
- `avg_usage`, `max_usage`, `std_usage` — Water consumption statistics
- `night_avg_usage` — Night-time usage (12 AM – 5 AM) — key leak indicator
- `avg_spike_ratio`, `max_spike_ratio`, `std_spike_ratio` — Anomaly spike patterns
- `night_day_ratio` — Night-to-day consumption ratio
- `usage_range` — Peak deviation from average
- `cv_usage` — Coefficient of variation (normalized volatility)

### Ensemble Verdict
When both models **agree**, the system proceeds with high confidence.  
When models **disagree**, the case is flagged for manual review — catching borderline risk scenarios.

---

## 📊 Dashboard Pages (11 Pages)

| # | Page | Description |
|---|------|-------------|
| 1 | **System Overview** | KPI cards, risk distribution, spike ratios, daily leak trends |
| 2 | **Leakage Severity Analysis** | Severity breakdown, temporal trends, usage by severity |
| 3 | **Household Risk Intelligence** | Risk profiling, top high-risk households, search |
| 4 | **Water Consumption Behavior** | Hourly/daily patterns, heatmaps, box plots by risk |
| 5 | **Abnormal Pattern Detection** | Spike scatter plots, anomaly thresholds, anomaly table |
| 6 | **Household Explorer** | Deep-dive per household with timeline and leak events |
| 7 | **ML Risk Prediction** | Dual-model gauges (RF + XGBoost), what-if sliders, ensemble verdict |
| 8 | **Model Comparison** | Head-to-head metrics, ROC curves, confusion matrices, feature importance |
| 9 | **Smart Insights Panel** | Auto-generated analytics intelligence |
| 10 | **Data Explorer** | Filtered data table with CSV download |
| 11 | **Methodology & Formulas** | Technical documentation of all algorithms |

---

## 📁 Project Structure

```
water_leakage_bigdata_project_2/
│
├── data/
│   ├── raw/                          # Raw smart meter data
│   └── processed/                    # Processed analytics datasets
│       └── leakage_intelligence_dataset.csv  (4.32M records)
│
├── models/
│   ├── household_risk_model.pkl      # Random Forest model pipeline
│   ├── household_xgboost_model.pkl   # XGBoost model pipeline
│   └── model_comparison_metrics.pkl  # Comparison metrics (ROC, CM, etc.)
│
├── notebooks/
│   └── 05_machine_learning_prediction.ipynb  # ML training notebook
│
├── dashboard/
│   ├── dashboard.py                  # Streamlit dashboard (11 pages)
│   └── requirements.txt             # Dashboard dependencies
│
├── architecture/                     # Architecture diagrams
├── train_xgboost_model.py           # Model training script (RF + XGBoost)
├── create_nb.py                     # Notebook generator
├── requirements.txt                 # Full project dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.11+
pip install -r requirements.txt
```

### Train Models
```bash
python train_xgboost_model.py
```
This trains both Random Forest and XGBoost, saves models to `models/`, and generates comparison metrics.

### Run Dashboard
```bash
cd dashboard
streamlit run dashboard.py
```
Open `http://localhost:8501` in your browser.

---

## 📈 Key Results

- **4,320,000** smart meter records analyzed across **1,000** households
- **247,960** leak events detected (5.7% detection rate)
- **486** high-risk households identified, **232** critical
- Dual ML models provide ensemble risk predictions with behavioral what-if simulation
- Interactive dashboard with 11 analytical pages for comprehensive monitoring

---

## 📊 Model Comparison Highlights

Both models are evaluated on the same 80/20 stratified train-test split:
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score, AUC-ROC
- **Visual Comparisons**: Overlaid ROC curves, side-by-side confusion matrices
- **Feature Importance**: Grouped bar charts showing which behavioral patterns each model prioritizes
- **Training Configuration**: Full hyperparameter transparency for reproducibility

---

## 🏗️ Architecture

The system follows a layered architecture:
1. **Data Layer** — Smart meter data ingestion and Spark processing
2. **Analytics Layer** — Feature engineering, anomaly detection, risk scoring
3. **ML Layer** — Dual-model training, evaluation, and serialization
4. **Presentation Layer** — Interactive Streamlit dashboard with Plotly visualizations

---

## 👤 Author

**Rushit Khakhkhar**  
B.Tech in Information Technology  

---

## 📜 License

This project is for academic and research purposes.
