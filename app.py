import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MoltAnalytics | İstihbarat Paneli", page_icon="🛡️", layout="wide")

# --- ÖZEL CSS (MODERN & PROFESYONEL) ---
st.markdown("""
<style>
    .stApp { background-color: #0b0d11; color: #e0e6ed; }
    div[data-testid="metric-container"] {
        background-color: #161920;
        border: 1px solid #2d3139;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
    }
    div[data-testid="stMetricValue"] {
        color: #00d2ff;
        font-size: 2.5rem !important;
        font-weight: bold;
    }
    [data-testid="stSidebar"] { background-color: #101216; border-right: 1px solid #2d3139; }
</style>
""", unsafe_allow_html=True)

# --- VERİ YÜKLEME ---
@st.cache_data
def load_static_data():
    path_cluster = "data/processed/ajan_kumeleri.csv"
    path_prophet = "data/processed/prophet_tahmin.csv"
    path_dil = "data/processed/dil_analizi.csv"
    
    df_c = pd.read_csv(path_cluster) if os.path.exists(path_cluster) else pd.DataFrame()
    df_p = pd.read_csv(path_prophet) if os.path.exists(path_prophet) else pd.DataFrame()
    df_d = pd.read_csv(path_dil) if os.path.exists(path_dil) else pd.DataFrame()

    if not df_c.empty and 'cluster' in df_c.columns:
        df_c['cluster'] = df_c['cluster'].astype(str)
    return df_c, df_p, df_d

df_c, df_p, df_d = load_static_data()

# --- SOL MENÜ ---
st.sidebar.markdown("## 🛡️ MoltAnalytics Pro")
st.sidebar.caption("Gelişmiş Tehdit İstihbarat Paneli")
page = st.sidebar.selectbox("📌 Modül Seçin", ["Genel Bakış (Dashboard)", "Ajan Kümeleme", "Risk Tahmini"])

# =========================================================
# SAYFA İÇERİKLERİ (STABİL & GERÇEK VERİ)
# =========================================================

if page == "Genel Bakış (Dashboard)":
    st.markdown("## 📊 Merkezi İstihbarat Dashboard'u")
    st.markdown("<hr style='border-color: #2d3139;'>", unsafe_allow_html=True)
    
    if not df_c.empty:
        toplam_kayit = len(df_c)
        avg_tox = df_c['toxicity_score'].mean()
        sifreli_sayisi = df_d[df_d['type'] == 'Encrypted']['count'].values[0] if not df_d.empty else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Toplam Analiz Edilen Ajan", f"{toplam_kayit:,}")
        m2.metric("Sistem Ort. Toksisitesi", f"{avg_tox:.3f}")
        m3.metric("Tespit Edilen Şifreli İletişim", f"{sifreli_sayisi}")
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📈 Aktivite Trendi")
            if not df_p.empty:
                fig_trend = px.line(df_p.head(30), x='ds', y='yhat', template="plotly_dark", color_discrete_sequence=['#00d2ff'])
                fig_trend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_trend, use_container_width=True, key="d_trend")

        with c2:
            st.markdown("### 🏷️ Karakter (Küme) Dağılımı")
            topic_data = df_c['cluster'].value_counts().reset_index()
            topic_data.columns = ['Küme', 'Ajan Sayısı']
            fig_topic = px.bar(topic_data, x='Küme', y='Ajan Sayısı', template="plotly_dark", color='Küme', color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig_topic.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_topic, use_container_width=True, key="d_topic")
            
        st.markdown("<br>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### 🌡️ Toksisite Yoğunluk Haritası")
            fig_heat = px.density_heatmap(df_c, x="pca_x", y="pca_y", z="toxicity_score", template="plotly_dark", color_continuous_scale="Inferno")
            fig_heat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_heat, use_container_width=True, key="d_heat")

        with c4:
            st.markdown("### 🔐 İletişim Tipi Dağılımı")
            if not df_d.empty:
                fig_pie = px.pie(df_d, values='count', names='type', hole=0.6, template="plotly_dark", color='type', color_discrete_map={'Normal':'#1f77b4', 'Encrypted':'#e74c3c'})
                fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True, key="d_pie")

elif page == "Ajan Kümeleme":
    st.markdown("## 🧠 Davranışsal Kümeleme Analizi")
    st.markdown("<hr style='border-color: #2d3139;'>", unsafe_allow_html=True)
    if not df_c.empty:
        fig_scatter = px.scatter(df_c, x='pca_x', y='pca_y', color='cluster', hover_data=['id', 'toxicity_score'], template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Safe)
        fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_scatter.update_traces(marker=dict(size=9, opacity=0.7, line=dict(width=1, color='#161920')))
        st.plotly_chart(fig_scatter, use_container_width=True, key="k_scatter")

elif page == "Risk Tahmini":
    st.markdown("## 🔮 Gelecek Projeksiyonu ve Risk Kontrolü")
    st.markdown("<hr style='border-color: #2d3139;'>", unsafe_allow_html=True)
    if not df_p.empty:
        col_ctrl, col_graph = st.columns([1, 2])
        with col_ctrl:
            st.markdown("### ⚙️ Sistem Parametreleri")
            thresh = st.slider("Güvenlik Alarm Eşiği", 0.0, 5.0, 0.5, 0.1, key="t_slider")
            curr_max = df_p['yhat'].max()
            
            st.metric("Modelin Öngördüğü Maks. Risk", f"{curr_max:.3f}")
            if curr_max > thresh:
                st.error(f"🚨 DİKKAT: Seçilen risk eşiği, tahmin edilen maksimum değeri aşıyor!")
            else:
                st.success("✅ Seçilen eşik değerine göre sistem stabil.")
        
        with col_graph:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=df_p['ds'].tolist() + df_p['ds'].tolist()[::-1], y=df_p['yhat_upper'].tolist() + df_p['yhat_lower'].tolist()[::-1], fill='toself', fillcolor='rgba(231, 76, 60, 0.15)', line=dict(color='rgba(0,0,0,0)'), name="Güven Aralığı"))
            fig_p.add_trace(go.Scatter(x=df_p['ds'], y=df_p['yhat'], line=dict(color='#e74c3c', width=3), name="Tahmin Edilen Eğilim"))
            fig_p.add_hline(y=thresh, line_dash="dash", line_color="#00d2ff", annotation_text="Eşik Değeri")
            fig_p.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_p, use_container_width=True, key="t_prophet")