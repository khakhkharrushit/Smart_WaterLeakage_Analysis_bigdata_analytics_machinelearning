# 5. Smart Dashboard Integration & Machine Learning Visualization

## 5.1 Overview of the Dashboard Architecture
To translate the processed big data pipelines and machine learning algorithms into actionable insights for stakeholders, a comprehensive, interactive dashboard was developed using **Streamlit** and **Plotly**. The dashboard operates as a unified frontend for the Smart City Water Infrastructure system, providing predictive alerts, anomaly detection, and granular geographical or household-level data exploration. 

The dashboard provides eight specialized analytical views designed to isolate specific infrastructural failure points.

*[Insert Screenshot 1: The "System Overview" Dashboard Page showing the main high-level metrics]*

## 5.2 Methodologies & Mathematical Thresholds
The underlying logic powering the visual anomalies is driven by rigid statistical heuristics and behavioral clustering. The core metrics displayed within the dashboard are calculated using the following methodologies:

### 5.2.1 Spike Ratio (Anomaly Detection)
The Spike Ratio serves as the primary statistical test for sudden pipe bursts.
* **Formula:** `Spike Ratio = Current Hourly Usage / 7-Day Rolling Average`
* **Statistical Ranges:**
  * `< 1.0` – Normal, below-average consumption.
  * `1.0 - 1.5` – Expected variance.
  * `1.5 - 3.0` – **Moderate Anomaly**: High standard deviation from historical flow.
  * `> 3.0` – **Severe Spike**: Indicates a potential burst pipe or unauthorized massive water discharge.

### 5.2.2 Household Risk Profiling
Each localized household is categorized into one of five tier-based risk levels, dictated by the frequency of flow spikes and sustained probability of continuous leakage:
1. **Normal:** Smooth utilization (Spike ratio generally `~1.0`).
2. **Low/Moderate Risk:** Sporadic elevated usage (Spike ranges `1.5-3.0`).
3. **High Risk / Critical:** Sustained severe anomalies, continuous night-usage, or high predictive probability of an active major leak burst.

*[Insert Screenshot 2: The "Household Explorer" or "Abnormal Pattern Detection" page showing the Risk Profiling logic applied to individual records]*

## 5.3 Predictive Maintenance: Machine Learning Subsystem
In addition to heuristics, a proactive **Random Forest Classifier** was integrated into the pipeline to act as an Early-Warning System (EWS).

### 5.3.1 Resolving Target Leakage & Enhancing Behavioral Analysis
During initial model iterations, utilizing direct sensory leak flags (`total_leak_events`) resulted in "target leakage," where the model simply learned the flagging threshold rather than predicting future risk. To circumvent this, the model's feature space was strictly limited to **behavioral indicators**:
* `avg_usage` and `max_usage` (Highest volume spikes)
* `std_usage` (Standard Deviation; a measure of water volatility)
* `avg_spike_ratio` and `max_spike_ratio`
* `night_avg_usage` (Usage specifically bracketed between 12:00 AM - 05:00 AM)

By isolating `night_avg_usage`, the Random Forest is highly sensitized to continuous trickle leaks—the most ubiquitous cause of unseen infrastructure decay.

*[Insert Screenshot 3: The "ML Risk Prediction" Dashboard Page showing the probability gauge and key risk drivers]*

### 5.3.2 What-If Analysis and Simulator
The frontend incorporates a real-time behavioral simulator designed for infrastructural stress testing. Engineers can isolate a specific `HOUSE_ID` and artificially modify the historical variance, night usage, and maximum spikes using interconnected sliders. The Random Forest model evaluates this theoretical data array on the fly and returns a dynamic **High Risk Probability (%)**, allowing city planners to validate the stability of arbitrary infrastructural zones under changing behavioral patterns.

*[Insert Screenshot 4: A zoomed-in capture of the "Tweak Household Behaviors" panel and the reactive Plotly Gauge Chart]*

## 5.4 Conclusion & Deployment Success
The integration of a visually dynamic, mathematically robust dashboard transitions the project from a raw big data pipeline into an enterprise-ready software solution. The seamless pairing of hard-coded Bayesian heuristics alongside the Random Forest ML backend guarantees a multi-layered detection net for water abnormalities, preventing false positives while alerting personnel to critical infrastructural failures accurately.
