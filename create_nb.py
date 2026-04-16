import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell
text = """# 💧 Smart Water Leakage Detection — Machine Learning Model

In this notebook, we build machine learning models to predict leakage risk at the **household level**.

Instead of detecting leaks per record, we aggregate time-series data per household and predict overall risk.

This makes the system highly realistic and useful for smart city monitoring, enabling proactive maintenance."""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

# Code cell - Imports
code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set visual context
sns.set_theme(style='darkgrid', palette='mako')"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# Markdown
text2 = """## 1. Load Preprocessed Data
We will load the previously generated `leakage_intelligence_dataset.csv` from the data folder."""
nb['cells'].append(nbf.v4.new_markdown_cell(text2))

# Code - Load Data
code2 = """# Define data path
DATA_PATH = '../data/processed/leakage_intelligence_dataset.csv'

# Read full dataset
print("Loading dataset... This might take a moment.")
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Dataset shape: {df.shape}")
df.head()"""
nb['cells'].append(nbf.v4.new_code_cell(code2))


# Markdown
text3 = """## 2. Feature Engineering & Household Aggregation
We need to shift from *record-level* anomalies to *household-level* risk profiles. 
We'll calculate aggregate statistics per household."""
nb['cells'].append(nbf.v4.new_markdown_cell(text3))

# Code - Aggregation
code3 = """# Aggregate data at the household level
household_features = df.groupby('household_id').agg(
    avg_usage=('water_usage_liters', 'mean'),
    max_usage=('water_usage_liters', 'max'),
    avg_spike_ratio=('spike_ratio', 'mean'),
    max_spike_ratio=('spike_ratio', 'max'),
    total_leak_events=('leak_flag_detected', 'sum'),
    critical_events=('risk_level', lambda x: (x == 'Critical').sum()),
    high_risk_events=('risk_level', lambda x: (x == 'High Risk').sum()),
    # We define the household risk logic: if they have critical/high events above a threshold > High Risk
).reset_index()

# Derive target variable (1 = High Risk Household, 0 = Normal)
# We flag a household as high risk if they have > 5 total leak events OR any critical event.
household_features['is_high_risk'] = (
    ((household_features['total_leak_events'] > 20) | (household_features['critical_events'] > 0)).astype(int)
)

print(f"Household Profile Shape: {household_features.shape}")
print("\\nRisk Target Distribution:")
print(household_features['is_high_risk'].value_counts(normalize=True) * 100)

household_features.head()"""
nb['cells'].append(nbf.v4.new_code_cell(code3))


# Markdown
text4 = """## 3. Data Splitting & Preprocessing
Let's prepare our feature set (X) and target variable (y)."""
nb['cells'].append(nbf.v4.new_markdown_cell(text4))

# Code - Split and prep
code4 = """# Define features
features = [
    'avg_usage', 
    'max_usage', 
    'avg_spike_ratio', 
    'max_spike_ratio', 
    'total_leak_events'
]

X = household_features[features]
y = household_features['is_high_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Optional for Random Forest, but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")"""
nb['cells'].append(nbf.v4.new_code_cell(code4))


# Markdown
text5 = """## 4. Train Random Forest Classifier
We use a Random Forest model as it's robust, requires minimal tuning, and provides excellent feature importance out of the box."""
nb['cells'].append(nbf.v4.new_markdown_cell(text5))

# Code - Model training
code5 = """# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')

# Fit the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("Model Training Complete! ✅")"""
nb['cells'].append(nbf.v4.new_code_cell(code5))


# Markdown
text6 = """## 5. Model Evaluation & Metrics
Let's see how well our predictive model performs."""
nb['cells'].append(nbf.v4.new_markdown_cell(text6))

# Code - Metrics
code6 = """# Classification Report
print("Classification Report:\\n")
print(classification_report(y_test, y_pred, target_names=['Normal', 'High Risk']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'High Risk'], yticklabels=['Normal', 'High Risk'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Household Risk Prediction')
plt.show()

# ROC-AUC
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})', color='#ef5350', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()"""
nb['cells'].append(nbf.v4.new_code_cell(code6))


# Markdown
text7 = """## 6. Feature Importance Explanation
Which factors contribute the most to making a household "High Risk"?"""
nb['cells'].append(nbf.v4.new_markdown_cell(text7))

# Code - Feature Importance
code7 = """# Get importance
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

# Plot interactive feature importance
fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
             title='Random Forest - Feature Importance for Household Risk',
             color='Importance', color_continuous_scale='Blues')
fig.update_layout(showlegend=False, height=400, width=800)
fig.show()"""
nb['cells'].append(nbf.v4.new_code_cell(code7))


# Markdown
text8 = """## 7. Export Model Pipeline for Dashboard
Finally, we serialize the trained Random Forest model and scaler so they can be loaded instantly in the Streamlit dashboard app."""
nb['cells'].append(nbf.v4.new_markdown_cell(text8))

# Code - Save Model
code8 = """# Ensure models directory exists
os.makedirs('../models', exist_ok=True)

# Create a pipeline dict holding both scaler and model (although rf doesn't strictly need scaled inputs, it's good practice)
export_pipeline = {
    'model': rf_model,
    'scaler': scaler, # Just in case we need it
    'features': features
}

joblib.dump(export_pipeline, '../models/household_risk_model.pkl')
print("Model Pipeline saved successfully to 'models/household_risk_model.pkl' 🚀")"""
nb['cells'].append(nbf.v4.new_code_cell(code8))

# Save the notebook to disk
with open('notebooks/05_machine_learning_prediction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Notebook successfully generated: notebooks/05_machine_learning_prediction.ipynb")
