"""
XGBoost Model Training + Comparison Metrics Generator
=====================================================
Trains an XGBoost classifier on household-level aggregated features,
evaluates against the existing Random Forest model, and saves both
models + comparison metrics for the dashboard.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    print("xgboost not installed. Installing now...")
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "leakage_intelligence_dataset.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = r"C:\Users\khakh\OneDrive\Desktop\water_leakage_bigdata_project\data\processed\leakage_intelligence_dataset.csv"

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Dataset shape: {df.shape}")

# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING (Household Aggregation)
# ──────────────────────────────────────────────
print("\nAggregating household features...")

total_readings_per_hh = df.groupby('household_id').size().reset_index(name='total_readings')

night_df = df[df['hour'].isin([0, 1, 2, 3, 4, 5])]
day_df   = df[df['hour'].isin(range(6, 24))]

night_agg = night_df.groupby('household_id').agg(
    night_avg_usage=('water_usage_liters', 'mean')
).reset_index()

day_agg = day_df.groupby('household_id').agg(
    day_avg_usage=('water_usage_liters', 'mean')
).reset_index()

household_features = df.groupby('household_id').agg(
    avg_usage=('water_usage_liters', 'mean'),
    max_usage=('water_usage_liters', 'max'),
    std_usage=('water_usage_liters', 'std'),
    avg_spike_ratio=('spike_ratio', 'mean'),
    max_spike_ratio=('spike_ratio', 'max'),
    std_spike_ratio=('spike_ratio', 'std'),
    total_leak_events=('leak_flag_detected', 'sum'),
    avg_leak_prob=('leak_probability', 'mean'),
    critical_events=('risk_level', lambda x: (x == 'Critical').sum()),
    high_risk_events=('risk_level', lambda x: (x == 'High Risk').sum()),
    spike_above_2=('spike_ratio', lambda x: (x > 2).sum()),
).reset_index()

household_features = household_features.merge(total_readings_per_hh, on='household_id', how='left')
household_features = household_features.merge(night_agg, on='household_id', how='left')
household_features = household_features.merge(day_agg, on='household_id', how='left')

household_features['night_avg_usage']  = household_features['night_avg_usage'].fillna(0)
household_features['day_avg_usage']    = household_features['day_avg_usage'].fillna(household_features['avg_usage'])
household_features['std_usage']        = household_features['std_usage'].fillna(0)
household_features['std_spike_ratio']  = household_features['std_spike_ratio'].fillna(0)

# ── Derived features (high discriminative power) ──
# Night-to-day ratio: persistent night usage = hidden leaks
household_features['night_day_ratio'] = (
    household_features['night_avg_usage'] / household_features['day_avg_usage'].clip(lower=0.01)
)
# Usage range (max - avg): how extreme the peaks are
household_features['usage_range'] = household_features['max_usage'] - household_features['avg_usage']
# Coefficient of variation: normalized volatility
household_features['cv_usage'] = (
    household_features['std_usage'] / household_features['avg_usage'].clip(lower=0.01)
)

# ── Target variable: based on avg_leak_prob (Bayesian heuristic) ──
# This is independent of the behavioral features — it is computed from
# the original pipeline's probabilistic model (night usage + spike patterns).
# Using 75th percentile as the threshold for "High Risk".
target_threshold = household_features['avg_leak_prob'].quantile(0.75)
household_features['is_high_risk'] = (household_features['avg_leak_prob'] >= target_threshold).astype(int)

# If that didn't create 2 classes, fallback to median
if household_features['is_high_risk'].nunique() < 2:
    target_threshold = household_features['avg_leak_prob'].median()
    household_features['is_high_risk'] = (household_features['avg_leak_prob'] > target_threshold).astype(int)

print(f"Household Profile Shape: {household_features.shape}")
print(f"Avg leak probability threshold (75th pct): {target_threshold:.4f}")
print("\nRisk Target Distribution:")
print(household_features['is_high_risk'].value_counts(normalize=True) * 100)
print(f"  Class 0 (Normal):    {(household_features['is_high_risk'] == 0).sum()}")
print(f"  Class 1 (High Risk): {(household_features['is_high_risk'] == 1).sum()}")

# ──────────────────────────────────────────────
# 3. TRAIN/TEST SPLIT
# ──────────────────────────────────────────────
# IMPORTANT: Only behavioral features. NO leak-derived or risk-derived columns.
# This prevents target leakage.
features = [
    'avg_usage', 'max_usage', 'std_usage', 'night_avg_usage',
    'avg_spike_ratio', 'max_spike_ratio', 'std_spike_ratio',
    'night_day_ratio', 'usage_range', 'cv_usage',
]

X = household_features[features]
y = household_features['is_high_risk']

# Verify both classes exist
if len(np.unique(y)) < 2:
    print("\n⚠️ WARNING: Only one class found. Adjusting threshold to median...")
    threshold = household_features['risk_score'].median()
    household_features['is_high_risk'] = (household_features['risk_score'] > threshold).astype(int)
    y = household_features['is_high_risk']
    print(f"  New distribution: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set:  {X_test.shape}")


def safe_predict_proba(model, X_data):
    """Safely get probability for class 1, handling single-class models."""
    proba = model.predict_proba(X_data)
    if proba.shape[1] == 1:
        # Only one class — return constant
        return np.zeros(len(X_data)) if model.classes_[0] == 0 else np.ones(len(X_data))
    return proba[:, 1]


# ──────────────────────────────────────────────
# 4. TRAIN RANDOM FOREST
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("Training Random Forest Classifier...")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=8,
    random_state=42, class_weight='balanced'
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = safe_predict_proba(rf_model, X_test)

rf_accuracy  = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall    = recall_score(y_test, rf_pred, zero_division=0)
rf_f1        = f1_score(y_test, rf_pred, zero_division=0)
rf_auc       = roc_auc_score(y_test, rf_prob) if len(np.unique(y_test)) > 1 else 0.0
rf_cm        = confusion_matrix(y_test, rf_pred)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")
print(f"AUC-ROC:   {rf_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['Normal', 'High Risk']))

# ──────────────────────────────────────────────
# 5. TRAIN XGBOOST
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("Training XGBoost Classifier...")
print("="*50)

# Calculate scale_pos_weight for imbalanced classes
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric='logloss',
)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = safe_predict_proba(xgb_model, X_test)

xgb_accuracy  = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
xgb_recall    = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1        = f1_score(y_test, xgb_pred, zero_division=0)
xgb_auc       = roc_auc_score(y_test, xgb_prob) if len(np.unique(y_test)) > 1 else 0.0
xgb_cm        = confusion_matrix(y_test, xgb_pred)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

print(f"Accuracy:  {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall:    {xgb_recall:.4f}")
print(f"F1 Score:  {xgb_f1:.4f}")
print(f"AUC-ROC:   {xgb_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred, target_names=['Normal', 'High Risk']))

# ──────────────────────────────────────────────
# 6. SAVE MODELS + COMPARISON METRICS
# ──────────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

# Save RF model (overwrite existing)
rf_pipeline = {
    'model': rf_model,
    'scaler': scaler,
    'features': features,
}
joblib.dump(rf_pipeline, os.path.join(os.path.dirname(__file__), 'models', 'household_risk_model.pkl'))
print("\n✅ Random Forest model saved → models/household_risk_model.pkl")

# Save XGBoost model
xgb_pipeline = {
    'model': xgb_model,
    'scaler': scaler,
    'features': features,
}
joblib.dump(xgb_pipeline, os.path.join(os.path.dirname(__file__), 'models', 'household_xgboost_model.pkl'))
print("✅ XGBoost model saved → models/household_xgboost_model.pkl")

# Save comparison metrics
comparison_metrics = {
    'random_forest': {
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1': rf_f1,
        'auc_roc': rf_auc,
        'confusion_matrix': rf_cm,
        'fpr': rf_fpr,
        'tpr': rf_tpr,
        'feature_importances': dict(zip(features, rf_model.feature_importances_)),
        'model_params': {
            'n_estimators': 100,
            'max_depth': 8,
            'class_weight': 'balanced',
        },
        'training_samples': len(X_train),
        'test_samples': len(X_test),
    },
    'xgboost': {
        'accuracy': xgb_accuracy,
        'precision': xgb_precision,
        'recall': xgb_recall,
        'f1': xgb_f1,
        'auc_roc': xgb_auc,
        'confusion_matrix': xgb_cm,
        'fpr': xgb_fpr,
        'tpr': xgb_tpr,
        'feature_importances': dict(zip(features, xgb_model.feature_importances_)),
        'model_params': {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'training_samples': len(X_train),
        'test_samples': len(X_test),
    },
    'features': features,
    'test_size': 0.2,
    'random_state': 42,
}
joblib.dump(comparison_metrics, os.path.join(os.path.dirname(__file__), 'models', 'model_comparison_metrics.pkl'))
print("✅ Comparison metrics saved → models/model_comparison_metrics.pkl")

print("\n" + "="*50)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY! 🚀")
print("="*50)
