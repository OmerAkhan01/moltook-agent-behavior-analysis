import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    """
    Veriyi bir kere okur ve RAM'de (cache) tutarak uygulamanın performanslı çalışmasını sağlar.
    """
    try:
        return pd.read_parquet("data/raw/raw_data.parquet")
    except FileNotFoundError:
        import warnings
        warnings.warn("Veri dosyası bulunamadı. Lütfen fetch_data.py aracılığıyla veriyi indirin.")
        return pd.DataFrame()
