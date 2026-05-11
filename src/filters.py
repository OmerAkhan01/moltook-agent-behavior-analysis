"""
src/filters.py
────────────────
UI katmanının doğrudan dosya okumasını engelleyen servis katmanı.

Not: Repo içinde halihazırda `src/data_loader.py` cache’li loader + filtre sunuyor.
Bu modül, diğer branch/denemelerdeki import’larla uyumluluk ve tek API yüzeyi sağlar.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import (
    get_filtered_agents,
    get_filtered_toxicity_trend,
    load_toxicity_forecast,
)
from src.models import forecast_toxicity_from_trend


def get_filtered_toxicity(start_date, end_date) -> pd.DataFrame:
    """
    Tahmin serisini verilen tarih aralığına göre döndürür.
    Öncelik: `data/processed/toxicity_forecast.parquet` (varsa)
    Fallback: trend’den runtime forecast (cache’li)
    """
    cached = load_toxicity_forecast()
    if cached is not None and not cached.empty:
        df = cached.copy()
        if "toxicity_forecast" in df.columns:
            df = df.rename(columns={"toxicity_forecast": "yhat"})
    else:
        trend = get_filtered_toxicity_trend()
        df = forecast_toxicity_from_trend(trend, periods=30)

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    start_ts = pd.to_datetime(start_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    mask = (df["ds"] >= start_ts) & (df["ds"] <= end_ts)
    return df.loc[mask].reset_index(drop=True)


def get_cluster_data(cluster_labels: list[str] | None = None) -> pd.DataFrame:
    """
    Küme etiketlerine göre ajanları filtreler.
    (Eski 'cluster id' yaklaşımı yerine bizim `cluster_label` kullanıyoruz.)
    """
    return get_filtered_agents(clusters=cluster_labels).reset_index(drop=True)


def get_toxicity_stats(cluster_label: str | None = None) -> dict:
    """
    Basit toksisite istatistikleri.
    """
    df = get_filtered_agents(clusters=[cluster_label] if cluster_label else None)
    if df is None or df.empty or "toxicity_score" not in df.columns:
        return {"mean": 0.0, "max": 0.0, "high_risk_count": 0, "total": 0}

    tox = pd.to_numeric(df["toxicity_score"], errors="coerce").fillna(0.0)
    return {
        "mean": float(tox.mean()),
        "max": float(tox.max()),
        "high_risk_count": int((tox >= 0.8).sum()),  # proxy skor 0-1 aralığında
        "total": int(len(df)),
    }

