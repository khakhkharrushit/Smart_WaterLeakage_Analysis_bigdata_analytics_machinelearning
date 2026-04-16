"""
Convert CSV dataset to Parquet format for GitHub + Streamlit Cloud deployment.
Parquet is ~10x smaller than CSV and loads much faster.
"""
import pandas as pd
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "leakage_intelligence_dataset.csv")
PARQUET_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "leakage_intelligence_dataset.parquet")

print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")
print(f"CSV size: {os.path.getsize(CSV_PATH) / 1024 / 1024:.1f} MB")

print(f"\nSaving Parquet: {PARQUET_PATH}")
df.to_parquet(PARQUET_PATH, engine='pyarrow', compression='snappy', index=False)

parquet_size = os.path.getsize(PARQUET_PATH) / 1024 / 1024
print(f"Parquet size: {parquet_size:.1f} MB")
print(f"Compression ratio: {os.path.getsize(CSV_PATH) / os.path.getsize(PARQUET_PATH):.1f}x smaller")
print("\n✅ Done! Parquet file saved successfully.")
