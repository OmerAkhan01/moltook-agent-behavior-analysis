"""
pages/2_Ajan_Kimlikleri.py
──────────────────────────
Küme başına ajan profilleri ve radar analizi.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.data_loader import load_clustered_agents, CLUSTER_COLORS

st.set_page_config(page_title="Ajan Kimlikleri | MoltAnalytics", page_icon="🤖", layout="wide")

PLOTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "plots"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌌 MoltAnalytics")
    st.caption("v1.0 · Yerel ML")
    st.divider()
    st.markdown("**🤖 Ajan Kimlikleri**")
    with st.expander("📷 Rapor Görselleri", expanded=False):
        for name in [
            "08_toxicity_violin_analysis.png",
            "09_cluster_topic_composition.png",
        ]:
            p = PLOTS_DIR / name
            if p.exists():
                st.image(str(p), use_container_width=True, caption=name)

# ── Veri ─────────────────────────────────────────────────────────────────────
df = load_clustered_agents()
if df.empty:
    st.stop()

PREFERRED_DIMS = ["toxicity_score","severe_toxicity","obscene","threat","insult","identity_attack","sexual_explicit"]
DIMS = [c for c in PREFERRED_DIMS if c in df.columns]

# Bu dataset (observatory archive) toksisite alt boyutlarını içermez.
# Bu durumda "profil" için ajan metrikleri üzerinden boyutlar seçilir.
FALLBACK_PROFILE_DIMS = [
    "toxicity_score",  # bu projede score'lardan normalize türetilen proxy
    "karma",
    "follower_count",
    "following_count",
    "post_count",
    "comment_count",
    "avg_post_score",
    "avg_comment_score",
]
PROFILE_DIMS = DIMS if len(DIMS) >= 2 else [c for c in FALLBACK_PROFILE_DIMS if c in df.columns]


def _robust_scale_series(s: pd.Series) -> pd.Series:
    """Clip to p01-p99 and min-max scale to 0-1 (keeps outliers from dominating)."""
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.clip(lower=0.0)
    lo = float(x.quantile(0.01))
    hi = float(x.quantile(0.99))
    if hi <= lo:
        return pd.Series(0.0, index=x.index)
    x = x.clip(lo, hi)
    return (x - lo) / (hi - lo)


def _transform_profile(df_in: pd.DataFrame, cols: list[str], mode: str) -> pd.DataFrame:
    """
    mode:
      - 'raw': no transform
      - 'log_scaled': log1p for heavy-tail + robust scale to 0-1
    """
    out = df_in[cols].copy()
    if mode == "raw":
        return out

    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x = x.clip(lower=0.0)
        # log1p for count/score heavy tails
        out[c] = np.log1p(x)
        out[c] = _robust_scale_series(out[c])
    return out

# ── Başlık ───────────────────────────────────────────────────────────────────
st.title("🤖 Ajan Kimlikleri")
st.caption("KMeans kümelerinin davranış profilleri ve ajan dağılımları")
st.caption("Not: Bu veri setinde toksisite alt-boyutları yoktur; profil boyutları ajan metriklerinden türetilir.")
st.divider()

tab_viz, tab_assets = st.tabs(["📊 Etkileşimli Analiz", "🖼️ Rapor Görselleri"])

# ── Küme Seçici ──────────────────────────────────────────────────────────────
with tab_viz:
    cluster_options = sorted(df["cluster_label"].unique().tolist())
    selected = st.selectbox("Profil İncelenecek Küme", cluster_options)
    dfc = df[df["cluster_label"] == selected]
    color = CLUSTER_COLORS.get(selected, "#888")

# ── KPI ──────────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ajan Sayısı",           f"{len(dfc):,}")
    k2.metric("Ort. Skor",        f"{dfc['toxicity_score'].mean():.4f}" if "toxicity_score" in dfc.columns else "—")
    k3.metric("Ort. Post Skoru",       f"{dfc['avg_post_score'].mean():.2f}" if "avg_post_score" in dfc.columns else "—")
    k4.metric("Ort. Yorum Skoru",      f"{dfc['avg_comment_score'].mean():.2f}" if "avg_comment_score" in dfc.columns else "—")
    st.divider()

# ── Radar + Box ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"🕸️ {selected} — Radar Profili")
        if PROFILE_DIMS and len(PROFILE_DIMS) >= 2:
            mode = st.radio(
                "Radar ölçeği",
                options=["Ölçekli (log + robust)", "Ham (raw)"],
                horizontal=True,
            )
            mode_key = "log_scaled" if mode.startswith("Ölçekli") else "raw"

            # Radar'ı küme ortalamaları üstünden çiz
            prof = df[["cluster_label"] + PROFILE_DIMS].copy()
            prof[PROFILE_DIMS] = _transform_profile(prof, PROFILE_DIMS, mode_key)
            all_means = prof.groupby("cluster_label")[PROFILE_DIMS].mean()

            fig_radar = go.Figure()
            for label, row in all_means.iterrows():
                is_selected = label == selected
                fig_radar.add_trace(go.Scatterpolar(
                    r=row.values.tolist() + [row.values[0]],
                    theta=PROFILE_DIMS + [PROFILE_DIMS[0]],
                    fill="toself" if is_selected else "none",
                    name=label,
                    line=dict(color=CLUSTER_COLORS.get(label, "#888"), width=3 if is_selected else 1),
                    opacity=1.0 if is_selected else 0.35,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1] if mode_key == "log_scaled" else None)),
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                margin=dict(t=40, b=80),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Radar için yeterli profil boyutu bulunamadı.")

    with c2:
        st.subheader(f"📦 {selected} — Boyut Dağılımları")
        if PROFILE_DIMS:
            # Dağılımda ölçekli görünüm kullan (büyük veride daha doğru kıyas)
            dfc_plot = dfc.copy()
            dfc_plot[PROFILE_DIMS] = _transform_profile(dfc_plot, PROFILE_DIMS, "log_scaled")
            dfc_long = dfc_plot[PROFILE_DIMS].melt(var_name="boyut", value_name="deger")
            fig_box = px.box(
                dfc_long, x="boyut", y="deger",
                color="boyut",
                points=False,
                labels={"boyut": "", "deger": "Değer"},
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Bu veri setinde profil boyutları bulunamadı.")

# ── Tüm Kümeler Karşılaştırması ───────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Tüm Kümeler — Skor Dağılımı")
    if "toxicity_score" in df.columns:
        fig_violin = px.violin(
            df, y="toxicity_score", x="cluster_label",
            color="cluster_label",
            color_discrete_map=CLUSTER_COLORS,
            box=True, points=False,
            labels={"cluster_label": "Küme", "toxicity_score": "Skor"},
        )
        fig_violin.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.info("Skor sütunu bulunamadı.")

# ── Özet Tablo ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Küme Özet İstatistikleri")
    if PROFILE_DIMS:
        # Median is more reliable than mean on heavy-tailed metrics
        summary = df.groupby("cluster_label")[PROFILE_DIMS].median().round(4).reset_index()
        summary.columns = ["Küme"] + PROFILE_DIMS
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Özet istatistik için uygun boyut sütunu bulunamadı.")

with tab_assets:
    st.subheader("🖼️ Ajan Kimlikleri — Görsel Kanıtlar")
    st.caption("Küme kompozisyonu ve dağılımlar (assets).")
    imgs = [
        ("Kümeleme (PCA)", "04_clustering_pca.png"),
        ("K-Ortalamalar küme profilleri (sunum figürü)", "11_fig2_cluster_profiles.png"),
        ("Konu dağılımı", "02_topic_distribution.png"),
        ("Küme–Konu kompozisyonu", "09_cluster_topic_composition.png"),
        ("Violin (skor)", "08_toxicity_violin_analysis.png"),
    ]
    cols = st.columns(2)
    for i, (title, fname) in enumerate(imgs):
        p = PLOTS_DIR / fname
        with cols[i % 2]:
            if p.exists():
                st.image(str(p), use_container_width=True, caption=title)
            else:
                st.warning(f"Görsel bulunamadı: {fname}")

    st.divider()
    st.subheader("🔁 Rapor Figürü ile Eşleştirme (Bu Çalışmanın Verisi)")
    st.caption(
        "Rapor figürleri gönderi (post) bazlı bir örneklem üzerinden üretildiği için değerler birebir aynı olmayabilir. "
        "Aşağıdaki grafik, **mevcut dashboard verisinden** (ajan bazlı) küme ortalama `toxicity_score` değerlerini gösterir."
    )

    if "toxicity_score" in df.columns and "cluster_label" in df.columns:
        byc = (
            df.groupby("cluster_label")["toxicity_score"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"toxicity_score": "ort_skor"})
        )
        platform_mean = float(df["toxicity_score"].mean())

        fig_match = px.bar(
            byc,
            x="cluster_label",
            y="ort_skor",
            color="cluster_label",
            color_discrete_map=CLUSTER_COLORS,
            text="ort_skor",
            labels={"cluster_label": "Küme", "ort_skor": "Ort. Skor"},
        )
        fig_match.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_match.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=platform_mean,
            y1=platform_mean,
            xref="paper",
            yref="y",
            line=dict(color="#7f8c8d", width=2, dash="dash"),
        )
        fig_match.add_annotation(
            x=1,
            y=platform_mean,
            xref="paper",
            yref="y",
            text=f"Platform Ort. ({platform_mean:.3f})",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color="#7f8c8d"),
        )
        fig_match.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_match, use_container_width=True)
    else:
        st.info("Eşleştirme grafiği için gerekli sütunlar bulunamadı (`cluster_label`, `toxicity_score`).")
