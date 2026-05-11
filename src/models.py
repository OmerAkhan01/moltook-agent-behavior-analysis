"""
src/models.py
────────────────
Streamlit üzerinde model yükleme/çalıştırma katmanı.

- Model dosyaları opsiyoneldir. Bulunamazsa uygulama yine çalışır.
- Cache stratejisi:
  - @st.cache_resource: Diskten model yükleme (tek sefer)
  - @st.cache_data: Veri + parametreye göre forecast üretme (tekrar kullan)
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import pandas as pd
import streamlit as st


_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _ROOT / "models"


@st.cache_resource
def load_kmeans_bundle() -> dict[str, Any] | None:
    """
    Beklenen dosyalar (varsa):
    - models/kmeans.joblib  (veya pkl)
    - models/scaler.joblib  (opsiyonel)
    """
    # joblib opsiyonel dependency; yoksa pickle fallback yapıyoruz
    km_path_joblib = _MODELS_DIR / "kmeans.joblib"
    km_path_pkl = _MODELS_DIR / "kmeans.pkl"
    scaler_path_joblib = _MODELS_DIR / "scaler.joblib"
    scaler_path_pkl = _MODELS_DIR / "scaler.pkl"

    if not km_path_joblib.exists() and not km_path_pkl.exists():
        return None

    km = None
    scaler = None
    try:
        if km_path_joblib.exists():
            import joblib  # type: ignore

            km = joblib.load(km_path_joblib)
        else:
            with open(km_path_pkl, "rb") as f:
                km = pickle.load(f)

        if scaler_path_joblib.exists():
            import joblib  # type: ignore

            scaler = joblib.load(scaler_path_joblib)
        elif scaler_path_pkl.exists():
            with open(scaler_path_pkl, "rb") as f:
                scaler = pickle.load(f)
    except Exception:
        # Model bozuksa UI’ı düşürmeyelim
        return None

    return {"kmeans": km, "scaler": scaler}


@st.cache_resource
def load_prophet_model() -> Any | None:
    """
    Beklenen dosya (varsa):
    - models/prophet.pkl
    """
    path = _MODELS_DIR / "prophet.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def forecast_toxicity_from_trend(df_trend: pd.DataFrame, periods: int = 15) -> pd.DataFrame:
    """
    df_trend: ds,y formatında (günlük) trend.
    Çıktı: ds, yhat, yhat_lower, yhat_upper
    """
    if df_trend is None or df_trend.empty:
        return pd.DataFrame()
    if not {"ds", "y"}.issubset(df_trend.columns):
        return pd.DataFrame()

    data = df_trend[["ds", "y"]].copy()
    data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["ds", "y"]).sort_values("ds")
    if data.empty:
        return pd.DataFrame()

    model = load_prophet_model()
    if model is None:
        from prophet import Prophet  # local import: cold start daha iyi

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.fit(data)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return forecast

