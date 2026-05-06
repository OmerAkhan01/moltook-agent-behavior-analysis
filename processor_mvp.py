import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import base64
import re
import os

os.makedirs("data/processed", exist_ok=True)

# ─────────────────────────────────────────
# 1. LOAD & SAMPLE
# ─────────────────────────────────────────
print("[Step 1/5] Loading and sampling...")
try:
    df_raw = pd.read_csv("moltbook_temiz_final.csv")
    sample_size = min(50000, len(df_raw))
    df = df_raw.sample(n=sample_size, random_state=42).copy()
    print(f"  {sample_size} kayıt yüklendi.")
except Exception as e:
    print(f"Hata: {e}")
    exit(1)

# ─────────────────────────────────────────
# 2. ŞİFRELEME & DİL ANALİZİ
# ─────────────────────────────────────────
print("[Step 2/5] Şifreleme analizi...")

def is_base64(s):
    """
    Güvenilir Base64 tespiti.
    - Sadece Base64 karakteri içermeli
    - Minimum 20 karakter (kısa tokenlar false-positive verir)
    - Uzunluk 4'ün katı olmalı (padding kontrolü)
    - Kod bloğu veya boşluk içermemeli
    """
    if not isinstance(s, str):
        return False
    s = s.strip()
    # Çok kısa veya boşluk/newline içeriyorsa kesinlikle değil
    if len(s) < 20 or '\n' in s or ' ' in s:
        return False
    # Base64 pattern + uzunluk 4'ün katı
    if re.fullmatch(r'[A-Za-z0-9+/]*={0,2}', s) and len(s) % 4 == 0:
        try:
            decoded = base64.b64decode(s)
            # Gerçek Base64: decode edilince anlamlı bytes üretmeli
            return len(decoded) > 10
        except Exception:
            return False
    return False

df['is_encrypted'] = df['content_body'].apply(is_base64)

dil_analizi = pd.DataFrame({
    'type': ['Normal', 'Encrypted'],
    'count': [int((~df['is_encrypted']).sum()), int(df['is_encrypted'].sum())]
})
dil_analizi.to_json("data/processed/dil_analizi.json", orient="records")
print(f"  Şifreli: {df['is_encrypted'].sum()} | Normal: {(~df['is_encrypted']).sum()}")

# ─────────────────────────────────────────
# 2.5 HEATMAP (Toksisite × Şifreleme)
# ─────────────────────────────────────────
print("[Step 2.5/5] Heatmap matrisi...")
heatmap_matrix = pd.crosstab(df['toxic_level'], df['is_encrypted']).reset_index()
# Sütun isimlerini string yap (JSON uyumluluğu)
heatmap_matrix.columns = ['toxic_level'] + [str(c) for c in heatmap_matrix.columns[1:]]
heatmap_matrix.to_json("data/processed/heatmap_analizi.json", orient="records")

# ─────────────────────────────────────────
# 3. TOKSİSİTE SKORU
# ─────────────────────────────────────────
print("[Step 3/5] Toksisite skorları...")
df['toxicity_score'] = df['toxic_level'].astype(float)

# ─────────────────────────────────────────
# 4. KÜMELEME (t-SNE + KMeans)
# ─────────────────────────────────────────
print("[Step 4/5] Kümeleme (t-SNE)...")

df['content_len'] = df['content_body'].fillna("").apply(len)
# FIX: content_len log-normalize edilmeli, aksi halde clustering'i domine eder
df['content_len_log'] = np.log1p(df['content_len'])

features = ['toxicity_score', 'is_encrypted', 'upvotes_count', 'content_len_log']

cluster_df = df[['id', 'toxicity_score', 'is_encrypted', 'upvotes_count']].copy()
sample_cluster = cluster_df.sample(n=min(3000, len(cluster_df)), random_state=42).copy()

# Feature matrix
feature_matrix = df.loc[sample_cluster.index, features].copy()
feature_matrix['is_encrypted'] = feature_matrix['is_encrypted'].astype(int)
feature_matrix = feature_matrix.fillna(0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(feature_matrix)

# t-SNE — init parametresi versiyon kontrolüyle
import sklearn
sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
tsne_init = 'pca' if sklearn_version >= (1, 1) else 'random'

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init=tsne_init,
    learning_rate='auto'
)
coords = tsne.fit_transform(scaled_data)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
sample_cluster = sample_cluster.copy()
sample_cluster['cluster'] = kmeans.fit_predict(scaled_data)
sample_cluster['pca_x'] = coords[:, 0] + np.random.normal(0, 0.5, size=len(sample_cluster))
sample_cluster['pca_y'] = coords[:, 1] + np.random.normal(0, 0.5, size=len(sample_cluster))

sample_cluster[['id', 'toxicity_score', 'is_encrypted', 'upvotes_count', 'cluster', 'pca_x', 'pca_y']].to_json(
    "data/processed/ajan_kumeleri.json", orient="records"
)
print(f"  Küme dağılımı: {sample_cluster['cluster'].value_counts().to_dict()}")

# ─────────────────────────────────────────
# 5. PROPHET TAHMİNİ
# ─────────────────────────────────────────
print("[Step 5/5] Prophet tahmini...")
try:
    # FIX: timestamp parse + timezone kaldırma
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df['timestamp'] = df['timestamp'].dt.tz_convert(None)  # UTC naive
    df = df.dropna(subset=['timestamp'])

    forecast_data = (
        df.resample('D', on='timestamp')['toxicity_score']
        .mean()
        .reset_index()
    )
    forecast_data.columns = ['ds', 'y']
    forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])  # kesin datetime
    forecast_data['cap'] = 4.0
    forecast_data['floor'] = 0.0

    # En az 2 veri noktası şart
    if len(forecast_data) < 2:
        raise ValueError("Yetersiz zaman serisi verisi (min 2 gün gerekli)")

    m = Prophet(
        growth='logistic',
        interval_width=0.95,
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=False
    )
    m.fit(forecast_data)

    future = m.make_future_dataframe(periods=30)
    future['cap'] = 4.0
    future['floor'] = 0.0

    forecast = m.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.strftime('%Y-%m-%d')
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json(
        "data/processed/prophet_tahmin.json", orient="records"
    )
    print(f"  {len(forecast)} günlük tahmin kaydedildi.")

except Exception as e:
    print(f"  Prophet atlandı: {e}")

print("\n[TAMAMLANDI] Tüm çıktılar data/processed/ altında.")