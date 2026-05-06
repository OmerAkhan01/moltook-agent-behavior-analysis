import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. SAYFA AYARLARI (Sunum için geniş ve temiz görünüm)
st.set_page_config(page_title="Moltook Analiz Paneli", layout="wide", initial_sidebar_state="expanded")

# 2. VERİ YÜKLEME
@st.cache_data
def load_data(file):
    path = os.path.join("data", "processed", file)
    return pd.read_csv(path) if os.path.exists(path) else None

df_ajan = load_data("ajan_kumeleri.csv")
df_prophet = load_data("prophet_tahmin.csv")
df_dil = load_data("dil_analizi.csv")
df_heat = load_data("heatmap_analizi.csv")

# Ortak Tema Ayarı (Sunum için beyaz/aydınlık, net tema)
CHART_TEMPLATE = "plotly_white"

# 3. YAN MENÜ (Tamamen Profesyonel)
st.sidebar.title("🛡️ Moltook Analiz")
st.sidebar.markdown("---")
modul = st.sidebar.radio("📌 İnceleme Modülleri:", 
                         ["Genel Bakış (Dashboard)", "Ajan Kümeleme Analizi", "Risk Tahmini (Prophet)"])

# --- MODÜL 1: GENEL BAKIŞ ---
if modul == "Genel Bakış (Dashboard)":
    st.title("📊 Genel Bakış")
    st.markdown("Ajan Davranışları ve İletişim Analizi Özet Metrikleri")
    
    # Sena'nın HTML sayfasındaki üst metrik kartları
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Kayıt", "44.376")
    m2.metric("Küme 0 (Normal)", "33.239")
    m3.metric("Küme 1 (Toksik)", "11.137")
    m4.metric("Şifreli Mesaj", "12")
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("İletişim Tipi Dağılımı")
        if df_dil is not None:
            fig_pie = px.pie(df_dil, names=df_dil.columns[0], values=df_dil.columns[1], hole=0.5,
                             color_discrete_sequence=['#3498db', '#e74c3c'])
            fig_pie.update_layout(template=CHART_TEMPLATE, margin=dict(t=30, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Toksisite Yoğunluk Dağılımı")
        if df_heat is not None:
            fig_heat = px.bar(df_heat, x='toxic_level', y=[c for c in df_heat.columns if c != 'toxic_level'], 
                             barmode='group', color_discrete_sequence=['#3498db', '#e74c3c'])
            fig_heat.update_layout(template=CHART_TEMPLATE, xaxis_title="Toksisite Seviyesi", 
                                   yaxis_title="Kayıt Sayısı", legend_title="Şifreleme (False/True)", 
                                   margin=dict(t=30, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)

# --- MODÜL 2: AJAN KÜMELEME ANALİZİ ---
elif modul == "Ajan Kümeleme Analizi":
    st.title("🎯 Ajan Kümeleme Analizi (PCA)")
    st.markdown("Temel Bileşenler Analizi (PCA) ile Ajan Davranış Kümelerinin Dağılımı")
    st.markdown("---")
    
    if df_ajan is not None:
        cluster_col = [col for col in df_ajan.columns if 'cluster' in col.lower()][0]
        df_ajan[cluster_col] = "Küme " + df_ajan[cluster_col].astype(str) 
        
        # Sunum için pastel ve net ayrıştırılabilir renkler, noktalar daha estetik
        fig_scatter = px.scatter(df_ajan, x='pca_x', y='pca_y', color=cluster_col,
                                 color_discrete_sequence=px.colors.qualitative.Set1,
                                 opacity=0.7)
        fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
        fig_scatter.update_layout(template=CHART_TEMPLATE, xaxis_title="PCA X", yaxis_title="PCA Y", 
                                  legend_title="Davranış Kümeleri", margin=dict(t=30, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- MODÜL 3: RİSK TAHMİNİ ---
else:
    st.title("📈 Risk Tahmini ve Projeksiyon")
    st.markdown("Prophet Modeli ile Toksisite Zaman Serisi Analizi ve Gelecek Tahmini")
    st.markdown("---")
    
    if df_prophet is not None:
        fig_line = go.Figure()
        
        if 'yhat_upper' in df_prophet.columns and 'yhat_lower' in df_prophet.columns:
            # Güven aralığı çizgileri (Sena'nın grafiğindeki gibi açık mavi)
            fig_line.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig_line.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['yhat_lower'], mode='lines', line=dict(width=0), 
                                         fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', name='Güven Aralığı'))
            
        y_col = 'yhat' if 'yhat' in df_prophet.columns else df_prophet.columns[1]
        fig_line.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet[y_col], mode='lines+markers', 
                                     name='Tahmin Edilen Toksisite (yhat)', line=dict(color='#2980b9', width=2)))
        
        fig_line.update_layout(template=CHART_TEMPLATE, xaxis_title="Tarih", yaxis_title="Tahmin Değeri", 
                               hovermode="x unified", margin=dict(t=30, b=10))
        st.plotly_chart(fig_line, use_container_width=True)