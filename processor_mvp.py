import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import base64
import re
import os

# Create processed directory if not exists
os.makedirs("data/processed", exist_ok=True)

# 1. LOAD & SAMPLE
print("[Step 1/5] Loading data and sampling...")
try:
    # Reading CSV instead of Parquet for the current environment
    df_raw = pd.read_csv("data/raw/moltbook_raw.csv")
    # Take a 50k sample for the MVP (or total if less than 50k)
    sample_size = min(50000, len(df_raw))
    df = df_raw.sample(n=sample_size, random_state=42)
    print(f"Successfully loaded and sampled {sample_size} records.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# 2. BASE64 & CLEANING
print("[Step 2/5] Language and encryption analysis...")
def is_base64(s):
    if not isinstance(s, str): return False
    try:
        # Fast regex check for Base64 pattern
        if re.match(r'^[A-Za-z0-9+/]+={0,2}$', s) and len(s) > 10:
            return True
        return False
    except: return False

df['is_encrypted'] = df['content_body'].apply(is_base64)

# Summary for Pie Chart
dil_analizi = pd.DataFrame({
    'type': ['Normal', 'Encrypted'],
    'count': [len(df[~df['is_encrypted']]), len(df[df['is_encrypted']])]
})
dil_analizi.to_csv("data/processed/dil_analizi.csv", index=False)
print("Base64 analysis saved to dil_analizi.csv")

# 3. TOXICITY (Using existing toxic_level if available, or generating score)
print("[Step 3/5] Processing toxicity scores...")
# Since toxic_level is 0/1 in the data, we use it directly or map to floats
df['toxicity_score'] = df['toxic_level'].astype(float)

# 4. CLUSTERING & PCA (Identity Page)
print("[Step 4/5] Running behavioral clustering (Post-based)...")
# Since most agents have "ID Bulunamadı", we cluster individual posts to show density
cluster_df = df[['id', 'toxicity_score', 'is_encrypted', 'upvotes_count']].copy()

features = ['toxicity_score', 'is_encrypted', 'upvotes_count']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df[features].fillna(0))

pca = PCA(n_components=2)
coords = pca.fit_transform(scaled_data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_df['cluster'] = kmeans.fit_predict(scaled_data)
cluster_df['pca_x'] = coords[:, 0]
cluster_df['pca_y'] = coords[:, 1]

# Rename columns for clarity in frontend
cluster_df.columns = ['id', 'toxicity_score', 'is_encrypted', 'upvotes_count', 'cluster', 'pca_x', 'pca_y']

# Save a smaller sample for the scatter plot to keep it fast (e.g., 5k points)
cluster_df.sample(n=min(5000, len(cluster_df))).to_csv("data/processed/ajan_kumeleri.csv", index=False)
print("Cluster data saved to ajan_kumeleri.csv (Post-based)")

# 5. PROPHET FORECAST (Forecast Page)
print("[Step 5/5] Simulating future toxicity trends (Prophet)...")
try:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Resample by Day
    forecast_data = df.resample('D', on='timestamp')['toxicity_score'].mean().reset_index()
    forecast_data.columns = ['ds', 'y']
    forecast_data['ds'] = forecast_data['ds'].dt.tz_localize(None) # Remove timezone for Prophet

    m = Prophet(interval_width=0.95)
    m.fit(forecast_data)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("data/processed/prophet_tahmin.csv", index=False)
    print("30-day forecast saved to prophet_tahmin.csv")
except Exception as e:
    print(f"Prophet forecasting skipped or failed: {e}")

print("\n[MISSION COMPLETE] All 3 dynamic files are now in data/processed/")
