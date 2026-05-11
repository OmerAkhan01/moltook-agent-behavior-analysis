"""
app.py — MoltAnalytics Ana Kapı
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="MoltAnalytics",
    page_icon="🌌",
    layout="wide",
)

PLOTS_DIR = Path(__file__).resolve().parent / "assets" / "plots"

st.title("🌌 MoltAnalytics")
st.caption("114.000 Otonom Ajan Davranış Analizi Platformu")
st.divider()

col1, col2, col3 = st.columns(3)
col1.page_link("pages/1_Davranis_Analizi.py",  label="🔬 Davranış Analizi",  icon="1️⃣")
col2.page_link("pages/2_Ajan_Kimlikleri.py",   label="🤖 Ajan Kimlikleri",   icon="2️⃣")
col3.page_link("pages/3_Risk_Tahmini.py",       label="⚠️ Risk Tahmini",      icon="3️⃣")

st.divider()
st.markdown("""
**MoltAnalytics**, 114.000 otonom ajanın yankı odası oluşumlarını ve toksisite eğilimlerini
harici API olmadan, tamamen yerel makine öğrenmesi algoritmalarıyla (KMeans + Prophet) analiz eder.
Veriler Parquet formatında saklanır; 1 GB RAM kısıtı altında yüksek performansla çalışır.
""")

st.subheader("🖼️ Rapor Görselleri (Assets)")
st.caption("Notebook/rapor çıktıları projeye eklendi; sayfalarda ilgili yerlere gömülüdür.")

gallery = [
    ("Davranış", "01_daily_activity.png"),
    ("Kümeler (PCA)", "04_clustering_pca.png"),
    ("Risk (Forecast)", "06_toxicity_forecast.png"),
]
cols = st.columns(3)
for i, (label, fname) in enumerate(gallery):
    p = PLOTS_DIR / fname
    with cols[i]:
        if p.exists():
            st.image(str(p), use_container_width=True, caption=f"{label} · {fname}")
        else:
            st.warning(f"Görsel bulunamadı: {fname}")

with st.sidebar:
    st.header("🌌 MoltAnalytics")
    st.caption("v1.0 · Yerel ML")
    st.divider()
    st.info("Sol menüden bir analiz sayfası seçin.")
