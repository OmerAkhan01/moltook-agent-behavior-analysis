import streamlit as st

# Sayfa yapılandırması
st.set_page_config(page_title="MoltAnalytics Paneli", layout="wide")

# Kenar Çubuğu (Sidebar) Navigasyonu
st.sidebar.title("MoltAnalytics Navigasyon")
sayfa = st.sidebar.radio("Sayfa Seçin:", ["Dashboard", "Kümeleme", "Tahmin"])

st.sidebar.markdown("---")
st.sidebar.info("Proje: Otonom Yapay Zeka Ajanlarında Davranış Analizi ve Toksisite Tahmini")

# 1. Sayfa: Dashboard
if sayfa == "Dashboard":
    st.title("📊 Sistem Dashboard")
    st.subheader("Temel Metrikler ve Günlük Aktivite")
    st.info("Yakında...")

# 2. Sayfa: Kümeleme
elif sayfa == "Kümeleme":
    st.title("🎯 Ajan Kimlik Kümelemesi")
    st.subheader("Riskli Profiller ve Küme Analizi")
    st.info("Yakında...")

# 3. Sayfa: Tahmin
elif sayfa == "Tahmin":
    st.title("📈 30 Günlük Toksisite Tahmini")
    st.subheader("Zaman Serisi ve Erken Uyarı Trendleri")
    st.info("Yakında...")
