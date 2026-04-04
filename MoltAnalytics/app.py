import streamlit as st

st.set_page_config(
    page_title="MoltAnalytics",
    page_icon="🔴",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] h1, h2, h3 {
        color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("MoltAnalytics Dashboard")

st.sidebar.title("MoltAnalytics")
st.sidebar.markdown("""
Hoşgeldiniz! 
Moltbook platformundaki otonom yapay zeka ajanlarının davranış profil analizleri bu sistem üzerinden izlenebilir.

Sprint 1: Modüler altyapı iskeleti kurulumu.
""")
