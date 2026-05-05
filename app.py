import streamlit as st
import pandas as pd
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MoltAnalytics", page_icon="📊", layout="wide")

# --- 1. VERİ YÜKLEME ---
@st.cache_data
def veriyi_yukle():
    file_path = "moltbook_temiz.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path, low_memory=False)
    return None

df = veriyi_yukle()

# --- SIDEBAR (NAVİGASYON) ---
st.sidebar.title("🧭 Menü")
sayfa = st.sidebar.selectbox("Sayfa Seçin", ["Dashboard", "Kümeleme", "Tahmin"])

# ==========================================
# 2. DASHBOARD SAYFASI
# ==========================================
if sayfa == "Dashboard":
    st.title("📊 Sistem Dashboard")
    st.markdown("Gerçek veri seti üzerinden anlık hesaplanan sistem metrikleri.")

    if df is not None:
        # -- GENEL METRİKLER --
        st.subheader("🌐 Platform Genel Durumu")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Toplam Gönderi", f"{len(df):,}")
        with m2:
            st.metric("Benzersiz Ajan Sayısı", df['id'].nunique())
        with m3:
            st.metric("Ortalama Toksisite Skoru", f"%{df['toxic_level'].mean()*100:.2f}")

        st.markdown("---")

        # -- GRAFİKLER VE AJAN FİLTRESİ --
        col1, col2 = st.columns([2, 1.5])
        
        with col1:
            st.subheader("📈 Aktivite ve Konu Analizi")
            tab1, tab2 = st.tabs(["Günlük Aktivite", "Konu Dağılımı"])
            with tab1:
                try:
                    st.image("assets/gunluk_aktivite.png", use_container_width=True)
                except:
                    st.warning("'gunluk_aktivite.png' assets klasöründe bulunamadı.")
            with tab2:
                try:
                    st.image("assets/konu_dagilimi.png", use_container_width=True)
                except:
                    st.warning("'konu_dagilimi.png' assets klasöründe bulunamadı.")
                
        with col2:
            st.subheader("🔍 Ajan İnceleme Paneli")
            st.info("Spesifik bir ajanın verilerini inceleyin.")
            
            # Ajan listesinin başına "Tüm Ajanlar" seçeneği ekliyoruz
            ajan_listesi = ["Tüm Ajanlar"] + list(df['id'].unique())
            secilen_ajan = st.selectbox("İncelenecek Ajanı Seçin:", ajan_listesi)
            
            if secilen_ajan == "Tüm Ajanlar":
                st.write("Şu an **tüm platformun** son mesajları gösteriliyor.")
                st.dataframe(df[['id', 'topic_label', 'toxic_level', 'post']].head(10), use_container_width=True)
            else:
                ajan_df = df[df['id'] == secilen_ajan]
                
                a_m1, a_m2 = st.columns(2)
                a_m1.metric("Ajanın Toplam Gönderisi", len(ajan_df))
                a_m2.metric("Ajanın Ort. Toksisitesi", f"%{ajan_df['toxic_level'].mean()*100:.2f}")
                
                st.write(f"**{secilen_ajan[:8]}...** ID'li ajanın dökümü:")
                st.dataframe(ajan_df[['topic_label', 'toxic_level', 'post']].head(10), use_container_width=True)

    else:
        st.error("⚠️ 'moltbook_temiz.csv' dosyası bulunamadı!")

# ==========================================
# 3. KÜMELEME SAYFASI
# ==========================================
elif sayfa == "Kümeleme":
    st.title("🎯 Ajan Kimlik Kümelemesi")
    st.markdown("Bu sayfada ajanlar davranış profillerine göre 4 farklı risk kümesine ayrılmıştır.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(label="🟢 Küme 0", value="Düşük Risk", delta="Güvenli")
    k2.metric(label="🟡 Küme 1", value="Orta Risk", delta="İzlenmeli", delta_color="off")
    k3.metric(label="🟠 Küme 2", value="Yüksek Risk", delta="Uyarı", delta_color="inverse")
    k4.metric(label="🔴 Küme 3", value="Kritik", delta="Toksik!", delta_color="inverse")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Kümeleme Haritası (PCA Scatter)")
        try:
            st.image("assets/pca_scatter.png", caption="Ajanların Profillerine Göre Dağılımı", use_container_width=True)
        except:
            st.error("⚠️ 'pca_scatter.png' bulunamadı.")

    with col2:
        st.subheader("🔍 Küme Filtresi")
        if df is not None and 'cluster' in df.columns:
            secilen_kume = st.selectbox("İncelemek için bir küme seçin:", sorted(df['cluster'].unique()))
            filtrelenmis_veri = df[df['cluster'] == secilen_kume]
            st.write(f"Seçilen kümede toplam **{len(filtrelenmis_veri)}** ajan bulunuyor.")
            st.dataframe(filtrelenmis_veri[['id', 'toxic_level', 'topic_label']].head(15), use_container_width=True)
        else:
            st.warning("⚠️ Veri tablosunda 'cluster' sütunu henüz bulunmuyor.")

# ==========================================
# 4. TAHMİN SAYFASI
# ==========================================
elif sayfa == "Tahmin":
    st.title("🔮 30 Günlük Toksisite ve Aktivite Tahmini")
    st.markdown("Gelecekteki platform trafiği ve toksisite trendleri.")
    st.info("Sistem, önümüzdeki 30 gün için erken uyarı sinyalleri üretmektedir.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Aktivite Tahmini")
        try:
            st.image("assets/aktivite_tahmin.png", use_container_width=True)
        except:
            st.error("⚠️ 'aktivite_tahmin.png' bulunamadı.")
    with col2:
        st.subheader("☣️ Toksisite Trend Tahmini")
        try:
            st.image("assets/toksisite_trend.png", use_container_width=True)
        except:
            st.error("⚠️ 'toksisite_trend.png' bulunamadı.")

