"""
pages/3_Risk_Tahmini.py
───────────────────────
Günlük toksisite trendi + Prophet 30 günlük tahmin.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.data_loader import (
    load_clustered_agents,
    load_toxicity_trend,
    load_toxicity_forecast,
    CLUSTER_COLORS,
)

st.set_page_config(page_title="Risk Tahmini | MoltAnalytics", page_icon="⚠️", layout="wide")

PLOTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "plots"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌌 MoltAnalytics")
    st.caption("v1.0 · Yerel ML")
    st.divider()
    st.markdown("**⚠️ Risk Tahmini**")
    with st.expander("📷 Kanıt Görselleri", expanded=False):
        st.caption("Rapor/Colab çıktıları (özet görünüm).")
        thumbs = [
            ("Trend", "toksisite_trend.png"),
            ("Forecast", "06_toxicity_forecast.png"),
            ("Prophet hacim projeksiyonu (sunum figürü)", "12_fig3_prophet_volume.png"),
            ("Yoğunluk", "toksisite_yogunluk_keman.png"),
            ("Kümülatif", "kumulatif_buyume_tahmin.png"),
            ("Aktivite", "05_activity_forecast.png"),
        ]
        for i in range(0, len(thumbs), 2):
            c1, c2 = st.columns(2)
            for col, (label, fname) in zip([c1, c2], thumbs[i:i+2]):
                with col:
                    p = PLOTS_DIR / fname
                    if p.exists():
                        st.image(str(p), use_container_width=True, caption=label)
                    else:
                        st.caption(f"Eksik: {fname}")

# ── Veri ─────────────────────────────────────────────────────────────────────
df_agents = load_clustered_agents()
df_trend  = load_toxicity_trend()
df_forecast_cached = load_toxicity_forecast()

if df_agents.empty or df_trend.empty:
    st.stop()

# ── Başlık ───────────────────────────────────────────────────────────────────
st.title("⚠️ Risk Tahmini")
st.caption("Günlük toksisite trendi · Prophet ile 30 günlük öngörü")
st.caption("Not: Bu sayfadaki skorlar, observatory arşivindeki etkileşim skorlarından türetilen proxy’dir.")
st.divider()

tab_viz, tab_assets = st.tabs(["📈 Trend & Tahmin", "🖼️ Kanıtlar"])

# Rapor figürleriyle uyumlu risk eşikleri (Şekil 1)
RISK_LOW_MAX = 0.50
RISK_MED_MAX = 0.65

# ── KPI'lar ──────────────────────────────────────────────────────────────────
last_val   = float(df_trend["y"].iloc[-1])
peak_val   = float(df_trend["y"].max())
mean_val   = float(df_trend["y"].mean())
peak_date  = df_trend.loc[df_trend["y"].idxmax(), "ds"].strftime("%d %b %Y")

with tab_viz:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Son Günlük Skor", f"{last_val:.4f}")
    k2.metric("Zirve Skor",      f"{peak_val:.4f}", delta=f"📅 {peak_date}", delta_color="off")
    k3.metric("Dönem Ortalaması",     f"{mean_val:.4f}")
    k4.metric("Gözlem Günü",          f"{len(df_trend)}")
    st.divider()

    # ── Prophet Tahmini ─────────────────────────────────────────────────────
    st.subheader("🔮 30 Günlük Tahmin (Prophet)")

    # Colab çıktısı (CSV) varsa onu tercih et; yoksa parquet forecast'ı kullan.
    forecast = pd.DataFrame()
    prophet_ok = False

    colab_fc = PROCESSED_DIR / "prophet_tahmin.csv"
    if colab_fc.exists():
        try:
            forecast = pd.read_csv(colab_fc)
        except Exception as e:
            st.warning(f"Colab forecast okunamadı: {e}")

    if forecast.empty and df_forecast_cached is not None and not df_forecast_cached.empty:
        forecast = df_forecast_cached.copy()

    # normalize expected columns
    if not forecast.empty:
        forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
        forecast = forecast.dropna(subset=["ds"]).sort_values("ds")
        if "toxicity_forecast" in forecast.columns and "yhat" not in forecast.columns:
            forecast = forecast.rename(columns={"toxicity_forecast": "yhat"})
        prophet_ok = {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(forecast.columns)

    prophet_data = df_trend.rename(columns={"ds": "ds", "y": "y"}).copy()
    prophet_data["ds"] = pd.to_datetime(prophet_data["ds"], errors="coerce")
    prophet_data["y"] = pd.to_numeric(prophet_data["y"], errors="coerce")
    prophet_data = prophet_data.dropna(subset=["ds", "y"]).sort_values("ds")

    if prophet_ok:
        fc = forecast
        hist_end = prophet_data["ds"].max()
        hist_end_x = hist_end.to_pydatetime() if hasattr(hist_end, "to_pydatetime") else hist_end
        fc_future = fc[fc["ds"] > hist_end]
        fc_hist = fc[fc["ds"] <= hist_end]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc_future["ds"], fc_future["ds"][::-1]]),
            y=pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Güven Aralığı",
        ))
        fig_fc.add_trace(go.Scatter(
            x=prophet_data["ds"], y=prophet_data["y"],
            mode="lines", name="Gerçek Trend",
            line=dict(color="#3498db", width=2),
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_hist["ds"], y=fc_hist["yhat"],
            mode="lines", name="Model (Geçmiş)",
            line=dict(color="#95a5a6", width=1.5, dash="dot"),
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_future["ds"], y=fc_future["yhat"],
            mode="lines", name="Tahmin (30 gün)",
            line=dict(color="#e74c3c", width=2.5),
        ))
        fig_fc.add_shape(
            type="line",
            x0=hist_end_x,
            x1=hist_end_x,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#f39c12", width=2, dash="dash"),
        )
        fig_fc.add_annotation(
            x=hist_end_x,
            y=1,
            xref="x",
            yref="paper",
            text="Bugün",
            showarrow=False,
            yanchor="bottom",
            xanchor="left",
            font=dict(color="#f39c12"),
        )
        fig_fc.update_layout(
            xaxis_title="Tarih",
            yaxis_title="Skor",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        fig_trend = px.line(
            prophet_data, x="ds", y="y",
            labels={"ds": "Tarih", "y": "Günlük Ort. Skor"},
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Geniş Risk Analizi (kanıtlarla) ──────────────────────────────────────
    st.divider()
    st.subheader("🧭 Geniş Risk Analizi (Trend + Kanıt)")

    # Rolling / threshold based alerting on observed trend
    obs = prophet_data[["ds", "y"]].copy()
    obs = obs.sort_values("ds")
    obs["y_7d"] = obs["y"].rolling(7, min_periods=3).mean()
    obs["y_14d"] = obs["y"].rolling(14, min_periods=5).mean()

    p90 = float(obs["y"].quantile(0.90))
    p95 = float(obs["y"].quantile(0.95))
    last_7d = float(obs["y_7d"].iloc[-1]) if obs["y_7d"].notna().any() else float("nan")
    last_14d = float(obs["y_14d"].iloc[-1]) if obs["y_14d"].notna().any() else float("nan")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("7G Rolling", f"{last_7d:.4f}" if pd.notna(last_7d) else "—")
    a2.metric("14G Rolling", f"{last_14d:.4f}" if pd.notna(last_14d) else "—")
    a3.metric("P90 Eşiği", f"{p90:.4f}")
    a4.metric("P95 Eşiği", f"{p95:.4f}")

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=obs["ds"], y=obs["y"], mode="lines", name="Günlük", line=dict(color="#3498db", width=1.5)))
    fig_roll.add_trace(go.Scatter(x=obs["ds"], y=obs["y_7d"], mode="lines", name="7G Rolling", line=dict(color="#2ecc71", width=2)))
    fig_roll.add_trace(go.Scatter(x=obs["ds"], y=obs["y_14d"], mode="lines", name="14G Rolling", line=dict(color="#9b59b6", width=2)))
    # Plotly add_hline annotation can break when x-axis is datetime (int + datetime mean).
    # Use shapes + annotations with xref='paper' to keep everything in the same coordinate system.
    for y, label, color, dash in [
        (p90, "P90", "#f39c12", "dot"),
        (p95, "P95", "#e74c3c", "dash"),
    ]:
        fig_roll.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=y,
            y1=y,
            xref="paper",
            yref="y",
            line=dict(color=color, width=2, dash=dash),
        )
        fig_roll.add_annotation(
            x=0,
            y=y,
            xref="paper",
            yref="y",
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color=color),
        )
    fig_roll.update_layout(
        xaxis_title="Tarih",
        yaxis_title="Skor",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # Kanıt görsellerini analiz akışından koparmadan, modern şekilde “yan panel” olarak sun
    with st.expander("🖼️ Kanıt Görselleri (bu bölümle ilişkili)", expanded=False):
        e1, e2 = st.columns(2)
        with e1:
            p = PLOTS_DIR / "toksisite_trend.png"
            if p.exists():
                st.image(str(p), use_container_width=True, caption="Trend kanıtı")
        with e2:
            p = PLOTS_DIR / "06_toxicity_forecast.png"
            if p.exists():
                st.image(str(p), use_container_width=True, caption="Forecast kanıtı")

    # ── Rapor eşikleri ile uyumlu risk bantları ──────────────────────────────
    st.divider()
    st.subheader("🎚️ Risk Bantları (Rapor Eşikleri ile Uyumlu)")
    st.caption(f"Düşük < {RISK_LOW_MAX:.2f} · Orta {RISK_LOW_MAX:.2f}–{RISK_MED_MAX:.2f} · Yüksek > {RISK_MED_MAX:.2f}")

    if "toxicity_score" in df_agents.columns:
        s = pd.to_numeric(df_agents["toxicity_score"], errors="coerce").fillna(0.0)
        bands = pd.cut(
            s,
            bins=[-float("inf"), RISK_LOW_MAX, RISK_MED_MAX, float("inf")],
            labels=["Düşük", "Orta", "Yüksek"],
        )
        band_counts = (
            bands.value_counts()
            .reindex(["Düşük", "Orta", "Yüksek"])
            .fillna(0)
            .reset_index()
        )
        band_counts.columns = ["Risk Bandı", "Ajan Sayısı"]
        fig_band = px.bar(
            band_counts,
            x="Risk Bandı",
            y="Ajan Sayısı",
            color="Risk Bandı",
            color_discrete_map={"Düşük": "#2ecc71", "Orta": "#f39c12", "Yüksek": "#e74c3c"},
            text="Ajan Sayısı",
        )
        fig_band.update_traces(textposition="outside")
        fig_band.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_band, use_container_width=True)
    else:
        st.info("Risk bantları için `toxicity_score` bulunamadı.")

    # ── Küme Bazlı Risk Tablosu ──────────────────────────────────────────────
    st.divider()
    st.subheader("🚨 Küme Bazlı Risk Özeti")

    agg_spec = {
        "ajan_sayisi": ("agent_id", "count"),
    }
    if "toxicity_score" in df_agents.columns:
        agg_spec["ort_skor"] = ("toxicity_score", "mean")
    if "avg_post_score" in df_agents.columns:
        agg_spec["ort_post_skor"] = ("avg_post_score", "mean")
    if "avg_comment_score" in df_agents.columns:
        agg_spec["ort_yorum_skor"] = ("avg_comment_score", "mean")

    risk = df_agents.groupby("cluster_label").agg(**agg_spec).round(4).reset_index()

    if "ort_skor" in risk.columns:
        risk["risk_skoru"] = risk["ort_skor"].round(4)
    else:
        risk["risk_skoru"] = 0.0

    risk = risk.sort_values("risk_skoru", ascending=False).reset_index(drop=True)
    rename_cols = {"cluster_label": "Küme", "ajan_sayisi": "Ajan Sayısı", "risk_skoru": "Risk Skoru"}
    if "ort_skor" in risk.columns:
        rename_cols["ort_skor"] = "Ort. Skor"
    if "ort_post_skor" in risk.columns:
        rename_cols["ort_post_skor"] = "Ort. Post Skoru"
    if "ort_yorum_skor" in risk.columns:
        rename_cols["ort_yorum_skor"] = "Ort. Yorum Skoru"
    risk = risk.rename(columns=rename_cols)

    st.dataframe(
        risk.style.background_gradient(subset=["Risk Skoru"], cmap="Reds"),
        use_container_width=True,
    )

# ── Risk Çubuğu ───────────────────────────────────────────────────────────────
    fig_risk = px.bar(
        risk, x="Küme", y="Risk Skoru",
        color="Küme",
        color_discrete_map=CLUSTER_COLORS,
        text="Risk Skoru",
        labels={"Küme": "", "Risk Skoru": "Risk Skoru"},
    )
    fig_risk.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_risk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Cluster-level deeper view: distribution and top agents
    st.divider()
    st.subheader("🧩 Risk Sürücüleri: Kümeler ve Ajanlar")
    if "toxicity_score" in df_agents.columns:
        # distribution per cluster (violin)
        fig_v = px.violin(
            df_agents,
            x="cluster_label",
            y="toxicity_score",
            color="cluster_label",
            color_discrete_map=CLUSTER_COLORS,
            box=True,
            points=False,
            labels={"cluster_label": "Küme", "toxicity_score": "Skor"},
        )
        fig_v.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_v, use_container_width=True)

        # top risky agents (by proxy score)
        cols = ["agent_id", "cluster_label", "toxicity_score"]
        for c in ["avg_post_score", "avg_comment_score", "post_count", "comment_count", "karma"]:
            if c in df_agents.columns:
                cols.append(c)
        topn = (
            df_agents[cols]
            .sort_values("toxicity_score", ascending=False)
            .head(50)
            .reset_index(drop=True)
        )
        st.caption("En yüksek 50 ajan (skor bazlı). Bu skor, verilen verideki score metriklerinden türetilen proxy’dir.")
        st.dataframe(topn, use_container_width=True)

        with st.expander("🖼️ Dağılım Kanıtı", expanded=False):
            p = PLOTS_DIR / "toksisite_yogunluk_keman.png"
            if p.exists():
                st.image(str(p), use_container_width=True, caption="Yoğunluk / keman grafiği")

with tab_assets:
    st.subheader("🖼️ Kanıtlar")
    st.caption("Kanıt görselleri projede modern bir panel şeklinde sunulur (grid + detay).")

    evidence = [
        {
            "id": "trend",
            "title": "Trend Kanıtı",
            "file": "toksisite_trend.png",
            "insight": [
                "Zaman serisi davranışını (yükseliş/azalış) görsel olarak doğrular.",
                "Rolling eşiklerinin (P90/P95) anlamlı olduğu dönemleri işaret eder.",
            ],
        },
        {
            "id": "forecast",
            "title": "Tahmin Kanıtı",
            "file": "06_toxicity_forecast.png",
            "insight": [
                "30 günlük öngörü ve güven aralığı görseli.",
                "Tahmin belirsizliği (upper/lower band) risk yorumunu destekler.",
            ],
        },
        {
            "id": "volume",
            "title": "Gönderi Hacmi Projeksiyonu",
            "file": "12_fig3_prophet_volume.png",
            "insight": [
                "Gönderi hacmi projeksiyonu; risk skorunun bağlamını güçlendirir.",
            ],
        },
        {
            "id": "growth",
            "title": "Kümülatif Büyüme",
            "file": "kumulatif_buyume_tahmin.png",
            "insight": [
                "Kümülatif eğilim üzerinden uzun vadeli risk birikimini gösterir.",
            ],
        },
        {
            "id": "density",
            "title": "Yoğunluk / Dağılım",
            "file": "toksisite_yogunluk_keman.png",
            "insight": [
                "Dağılımın ağır kuyruklu olup olmadığını gösterir.",
                "Top-risk ajanların neden ayrılaştığını görsel olarak destekler.",
            ],
        },
        {
            "id": "activity",
            "title": "Aktivite Tahmini",
            "file": "05_activity_forecast.png",
            "insight": [
                "Aktivite projeksiyonu risk metriklerinin bağlamını güçlendirir.",
            ],
        },
    ]

    # Üstte grid (hızlı tarama)
    grid = st.container(border=True)
    with grid:
        st.markdown("**Hızlı görünüm**")
        cols = st.columns(3)
        for i, e in enumerate(evidence):
            with cols[i % 3]:
                p = PLOTS_DIR / e["file"]
                if p.exists():
                    st.image(str(p), use_container_width=True, caption=e["title"])
                else:
                    st.caption(f"Eksik: {e['file']}")

    # Altta detay (seçilebilir)
    options = {e["title"]: e for e in evidence}
    default_title = "Tahmin Kanıtı" if "Tahmin Kanıtı" in options else list(options.keys())[0]
    selected_title = st.selectbox("Detay görmek için kanıt seç", list(options.keys()), index=list(options.keys()).index(default_title))
    e = options[selected_title]

    p = PLOTS_DIR / e["file"]
    detail = st.container(border=True)
    with detail:
        h1, h2 = st.columns([2, 1])
        with h1:
            st.markdown(f"**{e['title']}**")
            for b in e["insight"]:
                st.markdown(f"- {b}")
        with h2:
            st.markdown("**Kaynak**")
            st.code(f"assets/plots/{e['file']}", language=None)

    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning(f"Görsel bulunamadı: {e['file']}")
