"""
Create a smaller dataset for Streamlit Cloud deployment.
Samples 200 households (instead of 1000) to stay under 1GB RAM.
"""
import pandas as pd
import os
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "leakage_intelligence_dataset.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "leakage_intelligence_dataset.parquet")

print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Full dataset: {df.shape}")

# Sample 200 households to keep memory under 1GB on Streamlit Cloud
np.random.seed(42)
all_households = df['household_id'].unique()
sampled_households = np.random.choice(all_households, size=200, replace=False)
df_sampled = df[df['household_id'].isin(sampled_households)].copy()

print(f"Sampled dataset: {df_sampled.shape} ({len(sampled_households)} households)")

df_sampled.to_parquet(OUTPUT_PATH, engine='pyarrow', compression='snappy', index=False)

parquet_size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
print(f"Parquet size: {parquet_size:.1f} MB")
print("\n✅ Sampled parquet saved!")
