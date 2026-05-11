"""
src/data_loader.py
──────────────────
Parquet dosyalarını Streamlit önbelleğinde tutar.

clustered_agents.parquet sütunları:
  agent_id, ds, cluster_id, cluster_label,
  karma, follower_count, following_count, is_claimed,
  post_count, comment_count,
  avg_post_score, avg_comment_score, max_post_score, max_comment_score,
  toxicity_score (proxy; verilen verideki score metriklerinden türetilir)

toxicity_daily_trend.parquet sütunları:
  ds (datetime), y (float32 — günlük ortalama skor; proxy)

toxicity_forecast.parquet sütunları:
  ds (datetime), toxicity_forecast / yhat (float),
  yhat_lower (float), yhat_upper (float)
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

_BASE = Path(__file__).resolve().parents[1] / "data" / "processed"
_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

CLUSTER_COLORS = {
    "Kaos Elçileri":        "#e74c3c",
    "İçerik Üreticiler":    "#3498db",
    "Pasif Gözlemciler":    "#2ecc71",
    "Yankı Odası Üyeleri":  "#f39c12",
}


@st.cache_data(ttl=3600)
def load_clustered_agents() -> pd.DataFrame:
    """
    200.000 satır ajan verisi.
    Sütunlar: toxicity_score, severe_toxicity, obscene, threat, insult,
              identity_attack, sexual_explicit, agent_id, ds,
              cluster_id, cluster_label
    """
    path = _BASE / "clustered_agents.parquet"
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        st.error(f"Dosya bulunamadı: `{path}`")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Okuma hatası: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_raw_data() -> pd.DataFrame:
    """
    (Opsiyonel) Ham veri loader'ı.
    Öncelik: data/raw/raw_data.parquet, fallback: data/raw/moltbook_raw.csv
    """
    parquet_path = _RAW / "raw_data.parquet"
    csv_path = _RAW / "moltbook_raw.csv"
    try:
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ham veri okuma hatası: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_toxicity_trend() -> pd.DataFrame:
    """
    Günlük ortalama toksisite trendi (139 gün).
    Sütunlar: ds (datetime), y (float32)
    """
    path = _BASE / "toxicity_daily_trend.parquet"
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        st.error(f"Dosya bulunamadı: `{path}`")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Okuma hatası: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_toxicity_forecast() -> pd.DataFrame:
    """
    Prophet çıktısı (opsiyonel).
    Sütunlar: ds, toxicity_forecast (veya yhat), yhat_lower, yhat_upper
    """
    path = _BASE / "toxicity_forecast.parquet"
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return df
        if "toxicity_forecast" not in df.columns and "yhat" in df.columns:
            df = df.rename(columns={"yhat": "toxicity_forecast"})
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        return df
    except FileNotFoundError:
        # Forecast dosyası opsiyonel: UI fallback yapabilir
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Okuma hatası: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _read_parquet_with_mtime(path_str: str, mtime_ns: int) -> pd.DataFrame:
    # mtime_ns is part of the cache key -> cache invalidates when file changes/appears
    return pd.read_parquet(path_str)


def load_topic_toxicity_heatmap() -> pd.DataFrame:
    """submolt × toksisite seviyesi yüzde dağılımı (post-level proxy)."""
    path = _BASE / "topic_toxicity_heatmap.parquet"
    try:
        if not path.exists():
            return pd.DataFrame()
        return _read_parquet_with_mtime(str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Okuma hatası: {e}")
        return pd.DataFrame()


def load_cluster_profiles() -> pd.DataFrame:
    """Küme bazlı ort. post_toxicity ve konu entropisi."""
    path = _BASE / "cluster_profiles.parquet"
    try:
        if not path.exists():
            return pd.DataFrame()
        return _read_parquet_with_mtime(str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Okuma hatası: {e}")
        return pd.DataFrame()


def load_topic_toxicity() -> pd.DataFrame:
    """submolt bazında ort. toksisite (post-level proxy) ve adet."""
    path = _BASE / "topic_toxicity.parquet"
    try:
        if not path.exists():
            return pd.DataFrame()
        return _read_parquet_with_mtime(str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Okuma hatası: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_filtered_agents(
    clusters: Optional[List[str]] = None,
    start_date: Optional["str | pd.Timestamp"] = None,
    end_date: Optional["str | pd.Timestamp"] = None,
) -> pd.DataFrame:
    """
    Küme ve tarih aralığına göre ajanları filtreler.
    Not: ds sütunu yoksa tarih filtresi uygulanmaz.
    """
    df = load_clustered_agents()
    if df.empty:
        return df

    dff = df
    if clusters:
        if "cluster_label" in dff.columns:
            dff = dff[dff["cluster_label"].isin(clusters)]
        else:
            return pd.DataFrame()

    if start_date is not None or end_date is not None:
        if "ds" not in dff.columns:
            return dff
        ds = pd.to_datetime(dff["ds"], errors="coerce")
        dff = dff.assign(ds=ds).dropna(subset=["ds"])
        if start_date is not None:
            start_ts = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_ts):
                dff = dff[dff["ds"] >= start_ts]
        if end_date is not None:
            end_ts = pd.to_datetime(end_date, errors="coerce")
            if pd.notna(end_ts):
                dff = dff[dff["ds"] <= end_ts]

    return dff


@st.cache_data(ttl=3600)
def get_filtered_toxicity_trend(
    start_date: Optional["str | pd.Timestamp"] = None,
    end_date: Optional["str | pd.Timestamp"] = None,
) -> pd.DataFrame:
    """
    Trend verisini (ds,y) tarih aralığına göre filtreler.
    """
    df = load_toxicity_trend()
    if df.empty:
        return df

    dff = df.copy()
    if "ds" in dff.columns:
        dff["ds"] = pd.to_datetime(dff["ds"], errors="coerce")
        dff = dff.dropna(subset=["ds"])

    if start_date is not None:
        start_ts = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_ts):
            dff = dff[dff["ds"] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_ts):
            dff = dff[dff["ds"] <= end_ts]

    return dff.sort_values("ds")
