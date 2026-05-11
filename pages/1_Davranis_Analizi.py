"""
pages/1_Davranis_Analizi.py
───────────────────────────
KMeans küme dağılımları ve davranış örüntüleri.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.data_loader import (
    load_clustered_agents,
    load_topic_toxicity,
    load_topic_toxicity_heatmap,
    CLUSTER_COLORS,
)

st.set_page_config(page_title="Davranış Analizi | MoltAnalytics", page_icon="🔬", layout="wide")

PLOTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "plots"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌌 MoltAnalytics")
    st.caption("v1.0 · Yerel ML")
    st.divider()
    st.markdown("**🔬 Davranış Analizi**")
    with st.expander("📷 Rapor Görselleri", expanded=False):
        for name in [
            "01_daily_activity.png",
            "03_toxicity_distribution.png",
            "04_clustering_pca.png",
        ]:
            p = PLOTS_DIR / name
            if p.exists():
                st.image(str(p), use_container_width=True, caption=name)

# ── Veri ─────────────────────────────────────────────────────────────────────
df = load_clustered_agents()
if df.empty:
    st.stop()

PREFERRED_DIMS = ["toxicity_score","severe_toxicity","obscene","threat","insult","identity_attack","sexual_explicit"]
TOXICITY_DIMS = [c for c in PREFERRED_DIMS if c in df.columns]

# Observatory archive veri setinde alt toksisite boyutları yoksa, ajan metriklerini göster
FALLBACK_DIMS = [
    "toxicity_score",
    "karma",
    "follower_count",
    "following_count",
    "post_count",
    "comment_count",
    "avg_post_score",
    "avg_comment_score",
]
DISPLAY_DIMS = TOXICITY_DIMS if len(TOXICITY_DIMS) >= 2 else [c for c in FALLBACK_DIMS if c in df.columns]

# ── Başlık ───────────────────────────────────────────────────────────────────
st.title("🔬 Davranış Analizi")
st.caption(f"{len(df):,} ajan · 4 davranış kümesi · KMeans (scikit-learn)")
st.caption("Not: Bu projede **Skor** alanı, verilen arşiv verisindeki etkileşim skorlarından türetilen bir proxy’dir.")
st.divider()

tab_viz, tab_assets = st.tabs(["📊 Etkileşimli Analiz", "🖼️ Rapor Görselleri"])

# ── KPI'lar ──────────────────────────────────────────────────────────────────
with tab_viz:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Toplam Ajan",        f"{len(df):,}")
    k2.metric("Ortalama Skor", f"{df['toxicity_score'].mean():.4f}" if "toxicity_score" in df.columns else "—")
    k3.metric("Maks Skor",     f"{df['toxicity_score'].max():.4f}" if "toxicity_score" in df.columns else "—")
    k4.metric("Küme Sayısı",        "4")
    k5.metric("Boyut (özellik)",    str(len(TOXICITY_DIMS)) if TOXICITY_DIMS else "—")
    st.divider()

    # ── Filtre ───────────────────────────────────────────────────────────────
    selected_clusters = st.multiselect(
        "Küme Filtresi",
        options=df["cluster_label"].unique().tolist(),
        default=df["cluster_label"].unique().tolist(),
    )
    dff = df[df["cluster_label"].isin(selected_clusters)]

    # ── Satır 1: Donut + Bar ────────────────────────────────────────────────
    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Küme Dağılımı")
        dist = dff["cluster_label"].value_counts().reset_index()
        dist.columns = ["cluster_label", "count"]
        fig_donut = px.pie(
            dist, values="count", names="cluster_label",
            hole=0.55,
            color="cluster_label",
            color_discrete_map=CLUSTER_COLORS,
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        fig_donut.update_layout(
            showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.subheader("Küme Başına Ortalama Boyutlar / Skor")
        if DISPLAY_DIMS:
            agg = dff.groupby("cluster_label")[DISPLAY_DIMS].mean().reset_index()
            agg_long = agg.melt(id_vars="cluster_label", var_name="boyut", value_name="ortalama")
            fig_bar = px.bar(
                agg_long, x="boyut", y="ortalama", color="cluster_label",
                barmode="group",
                color_discrete_map=CLUSTER_COLORS,
                labels={"boyut": "Boyut", "ortalama": "Ortalama", "cluster_label": "Küme"},
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Gösterilecek boyut bulunamadı.")

    # ── Satır 2: Scatter ─────────────────────────────────────────────────────
    st.subheader("Skor vs Bileşik Skor")
    dff = dff.copy()
    if all(c in dff.columns for c in ["severe_toxicity","threat","identity_attack","insult"]):
        dff["composite"] = (
            dff["severe_toxicity"] * 0.3 +
            dff["threat"]          * 0.25 +
            dff["identity_attack"] * 0.25 +
            dff["insult"]          * 0.2
        )
    else:
        # Proxy composite from available engagement/profile metrics (derived from given data)
        # Scale each component within the filtered set to 0-1 before weighting.
        comps = {}
        for col in ["avg_post_score", "avg_comment_score", "post_count", "comment_count", "karma"]:
            if col in dff.columns:
                s = pd.to_numeric(dff[col], errors="coerce").fillna(0.0)
                s = s.clip(lower=0.0)
                lo = float(s.quantile(0.01))
                hi = float(s.quantile(0.99))
                denom = (hi - lo) if (hi - lo) != 0 else 1.0
                comps[col] = ((s.clip(lo, hi) - lo) / denom)
        if comps:
            dff["composite"] = (
                comps.get("avg_post_score", 0) * 0.35 +
                comps.get("avg_comment_score", 0) * 0.25 +
                comps.get("post_count", 0) * 0.2 +
                comps.get("comment_count", 0) * 0.15 +
                comps.get("karma", 0) * 0.05
            )
        else:
            dff["composite"] = dff["toxicity_score"] if "toxicity_score" in dff.columns else 0.0

    sample = dff.sample(min(8000, len(dff)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="toxicity_score", y="composite",
        color="cluster_label",
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.45,
        labels={
            "toxicity_score": "Skor",
            "composite":      "Bileşik Skor",
            "cluster_label":  "Küme",
        },
        hover_data={"agent_id": True},
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.subheader("🧩 Konu (submolt) Bazlı Toksisite — Rapor ile Uyumlu Tablo")
    st.caption("Bu bölüm post verisinden türetilir (rapor figürlerinin mantığıyla uyumlu).")

    topic = load_topic_toxicity()
    heat = load_topic_toxicity_heatmap()

    if not topic.empty:
        min_n = int(st.slider("Minimum post sayısı (kayıtları stabilize eder)", 50, 5000, 500, step=50))
        topic_f = topic[topic["n_posts"] >= min_n].copy()
        if topic_f.empty:
            topic_f = topic.copy()

        topn = int(st.slider("Gösterilecek konu sayısı", 10, 60, 20, step=5))
        view = topic_f.sort_values("mean_toxicity", ascending=False).head(topn)
        # weighted platform mean (post-weighted)
        platform_mean = float((topic_f["mean_toxicity"] * topic_f["n_posts"]).sum() / max(float(topic_f["n_posts"].sum()), 1.0))

        fig_topic = px.bar(
            view,
            x="submolt",
            y="mean_toxicity",
            labels={"submolt": "Konu (submolt)", "mean_toxicity": "Ort. Toksisite (proxy)"},
        )
        fig_topic.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=platform_mean,
            y1=platform_mean,
            xref="paper",
            yref="y",
            line=dict(color="#7f8c8d", width=2, dash="dash"),
        )
        fig_topic.add_annotation(
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
        fig_topic.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_topic, use_container_width=True)
    else:
        st.info("Konu bazlı özet bulunamadı. `python3 src/process_data.py` ile tabloları üret.")

    if not heat.empty:
        st.subheader("🌡️ Konu × Toksisite Seviyesi Isı Haritası")
        # pick most frequent topics for a stable heatmap
        totals = heat.groupby("submolt")["total"].max().sort_values(ascending=False)
        top_topics = totals.head(25).index.tolist()
        h = heat[heat["submolt"].isin(top_topics)].copy()
        # pivot for heatmap
        pivot = h.pivot_table(index="submolt", columns="tox_level", values="pct", fill_value=0.0)
        pivot = pivot.reindex(index=top_topics)
        # keep level ordering
        cols_order = ["0_Nötr", "1_Hafif", "2_Orta", "3_Yüksek", "4_ÇokY"]
        pivot = pivot[[c for c in cols_order if c in pivot.columns]]

        fig_h = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="YlGn",
            labels=dict(x="Toksisite Seviyesi", y="Konu (submolt)", color="%"),
        )
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_h, use_container_width=True)

        st.subheader("🍩 Risk Bandı Dağılımı (Post Bazlı)")
        # aggregate over topics: overall distribution
        overall = (
            heat.groupby("tox_level")["count"]
            .sum()
            .reset_index()
            .rename(columns={"tox_level": "Risk Bandı", "count": "Post Sayısı"})
        )
        order = ["0_Nötr", "1_Hafif", "2_Orta", "3_Yüksek", "4_ÇokY"]
        overall["Risk Bandı"] = overall["Risk Bandı"].astype(str)
        overall["Risk Bandı"] = pd.Categorical(overall["Risk Bandı"], categories=order, ordered=True)
        overall = overall.sort_values("Risk Bandı")
        fig_pie = px.pie(overall, values="Post Sayısı", names="Risk Bandı", hole=0.55)
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Satır 3: Ham Tablo ───────────────────────────────────────────────────
    with st.expander("📋 Ham Veri Önizleme (ilk 500 satır)"):
        st.dataframe(dff.head(500), use_container_width=True)

with tab_assets:
    st.subheader("🖼️ Davranış Analizi — Görsel Kanıtlar")
    st.caption("Notebook/rapor çıktıları (assets) — sayfa ile ilişkili seçilmiş görseller.")
    imgs = [
        ("Günlük aktivite", "01_daily_activity.png"),
        ("Toksisite dağılımı", "03_toxicity_distribution.png"),
        ("Kümeleme (PCA)", "04_clustering_pca.png"),
        ("Konu bazlı ort. toksisite (sunum figürü)", "10_fig1_konu_toksisite.png"),
        ("Konu × toksisite ısı haritası (sunum figürü)", "13_fig4_heatmap_topic_toxicity.png"),
    ]
    cols = st.columns(2)
    for i, (title, fname) in enumerate(imgs):
        p = PLOTS_DIR / fname
        with cols[i % 2]:
            if p.exists():
                st.image(str(p), use_container_width=True, caption=title)
            else:
                st.warning(f"Görsel bulunamadı: {fname}")
