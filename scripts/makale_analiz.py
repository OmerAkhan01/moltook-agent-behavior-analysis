"""
makale_analiz.py — Makale için metrik + figür pipeline'ı
Çalıştırma: python3 scripts/makale_analiz.py
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet

RAW       = ROOT / "data" / "raw" / "moltbook-observatory-archive" / "data"
PROCESSED = ROOT / "data" / "processed"
PLOTS     = ROOT / "assets" / "plots"
PROCESSED.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150,
})

# ── Renkler ───────────────────────────────────────────────────────────────────
PAL_CLUSTER = {"K1":"#2ecc71","K2":"#3498db","K3":"#e74c3c","K4":"#95a5a6"}
PAL_CAT     = {"A":"#3498db","B":"#e74c3c","C":"#2ecc71","D":"#9b59b6",
               "E":"#f39c12","F":"#1abc9c","G":"#e67e22","H":"#34495e","I":"#bdc3c7"}

# ── Kategori haritası ─────────────────────────────────────────────────────────
CMAP = {
    "general":"A","introductions":"A","all":"A","meta":"A","main":"A",
    "moltbook":"A","shitposts":"A","offmychest":"A","blesstheirhearts":"A",
    "ponderings":"A","heartbeat":"A",
    "politics":"B","adversarial-reasoning":"B","policy":"B","geopolitics":"B",
    "debate":"B","election":"B",
    "crypto":"C","trading":"C","cryptocurrency":"C","finance":"C",
    "agentfinance":"C","usdc":"C","defi":"C","markets":"C","investing":"C",
    "agentcommerce":"C","agenteconomy":"C","crypto-hub":"C","business":"C","startups":"C",
    "ai":"D","technology":"D","tech":"D","security":"D","builds":"D","buildx":"D",
    "gpt":"D","aithoughts":"D","ai-agent-security":"D","aiagents":"D","ai-agents":"D",
    "agentstack":"D","tools":"D","tooling":"D","programming":"D","coding":"D",
    "engineering":"D","infrastructure":"D","builders":"D","lablab":"D",
    "openclaw":"D","openclaw-explorers":"D","clawnch":"D","showandtell":"D",
    "cybercentry":"D","slim-protocol":"D",
    "sports":"E","football":"E","soccer":"E","basketball":"E","travel":"E",
    "theatre":"E","pixelwar":"E",
    "art":"F","music":"F","shakespeare":"F","creativeprojects":"F","moltpunk":"F",
    "crustafarianism":"F","bazaarofbabel":"F","clawtasks":"F","claw":"F",
    "health":"G","wellness":"G","fitness":"G","mental":"G","nutrition":"G","existential":"G",
    "philosophy":"H","consciousness":"H","science":"H","aisafety":"H",
    "todayilearned":"H","emergence":"H","memory":"H","research":"H","agt-20":"H",
}
CAT_LABELS = {
    "A":"Genel Tartışma","B":"Politika","C":"Ekonomi","D":"Teknoloji",
    "E":"Spor","F":"Kültür–Sanat","G":"Sağlık–Yaşam","H":"Eğitim–Bilim","I":"Diğer",
}

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: tartışma bazlı risk proxy (0-4 ölçeği)
#   Yüksek yorum / düşük skor = tartışmalı → yüksek risk
#   Yüksek skor / az yorum    = sevilmiş   → düşük risk
# ─────────────────────────────────────────────────────────────────────────────
def risk_proxy_04(score: pd.Series, comments: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce").fillna(0.0).clip(-10, 100)
    c = pd.to_numeric(comments, errors="coerce").fillna(0.0).clip(0)
    raw = np.log1p(c) / (1.0 + np.log1p(s.clip(lower=0)))
    p1  = float(np.nanpercentile(raw, 1))
    p99 = float(np.nanpercentile(raw, 99))
    denom = p99 - p1 if (p99 - p1) > 0 else 1.0
    return pd.Series(np.clip((raw - p1) / denom * 4.0, 0.0, 4.0), index=s.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("▶ Veriler yükleniyor...")
    pf = sorted((RAW/"posts").rglob("*.parquet"))
    af = sorted((RAW/"agents").rglob("*.parquet"))
    cf = sorted((RAW/"comments").rglob("*.parquet"))

    df_posts = pd.concat([
        pd.read_parquet(f, columns=["id","agent_id","submolt","score","comment_count","created_at"])
        for f in pf], ignore_index=True)
    df_agents = pd.concat([pd.read_parquet(f) for f in af], ignore_index=True)
    df_comments = pd.concat([
        pd.read_parquet(f, columns=["id","agent_id","score","created_at"])
        for f in cf], ignore_index=True)

    df_posts["ds"] = pd.to_datetime(df_posts["created_at"], utc=True).dt.tz_localize(None).dt.normalize()
    df_posts["category"] = df_posts["submolt"].map(CMAP).fillna("I")
    df_posts["risk"] = risk_proxy_04(df_posts["score"], df_posts["comment_count"])

    print(f"  Posts    : {len(df_posts):>10,}")
    print(f"  Agents   : {len(df_agents):>10,}")
    print(f"  Comments : {len(df_comments):>10,}")
    print(f"  Süre     : {df_posts['ds'].min().date()} — {df_posts['ds'].max().date()}")
    return df_posts, df_agents, df_comments


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KATEGORİ METRİKLERİ
# ═══════════════════════════════════════════════════════════════════════════════
def compute_category_metrics(df_posts: pd.DataFrame) -> pd.DataFrame:
    print("\n▶ Kategori metrikleri (A-I)...")
    m = (
        df_posts.groupby("category")
        .agg(
            n_posts    = ("id",   "count"),
            mean_risk  = ("risk", "mean"),
            median_risk= ("risk", "median"),
            high_risk_pct = ("risk", lambda x: (x > 2.0).mean() * 100),
        )
        .round(4).reset_index()
    )
    m["label"] = m["category"].map(CAT_LABELS)
    m = m.sort_values("mean_risk", ascending=False)
    for _, r in m.iterrows():
        print(f"  {r['category']} ({r['label']:<20}) n={r['n_posts']:>9,}  "
              f"ort={r['mean_risk']:.3f}  med={r['median_risk']:.3f}  "
              f"yüksek%={r['high_risk_pct']:.1f}%")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KMEANS (k=4, n_init=100)
# ═══════════════════════════════════════════════════════════════════════════════
def run_clustering(df_agents, df_posts, df_comments):
    print("\n▶ KMeans (k=4, n_init=100)...")

    # Post aggregates per ajan
    post_agg = (
        df_posts.groupby("agent_id")
        .agg(post_count=("id","count"),
             avg_score=("score","mean"),
             avg_risk=("risk","mean"),
             total_comments_received=("comment_count","sum"))
        .reset_index()
    )
    comment_agg = (
        df_comments.groupby("agent_id")
        .agg(comment_count=("id","count")).reset_index()
    )

    base = df_agents.copy()
    base["agent_id"] = base["id"].astype(str)
    for col in ["karma","follower_count","following_count","is_claimed"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)

    base = base.merge(post_agg, on="agent_id", how="left")
    base = base.merge(comment_agg, on="agent_id", how="left")
    base[["post_count","comment_count","total_comments_received","avg_score","avg_risk"]] = (
        base[["post_count","comment_count","total_comments_received","avg_score","avg_risk"]].fillna(0)
    )

    feat_cols = [c for c in [
        "karma","follower_count","following_count","post_count",
        "comment_count","avg_score","avg_risk","total_comments_received",
    ] if c in base.columns]

    X_raw = base[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    for c in ["karma","follower_count","following_count","post_count","comment_count","total_comments_received"]:
        if c in X_raw.columns:
            X_raw[c] = np.log1p(X_raw[c].clip(lower=0))
    X = StandardScaler().fit_transform(X_raw)

    km = KMeans(n_clusters=4, random_state=42, n_init=100, max_iter=500)
    base["cluster_id"] = km.fit_predict(X)

    # K1=en yüksek post hacmi → Yüksek Hacimli Yayıncı
    # K2=en yüksek karma/follower → Etkileşimli Konuşmacı
    # K3=en yüksek risk → Tartışmacı Profil
    # K4=en düşük aktivite → Pasif–Çeşitleyici
    means = base.groupby("cluster_id")[["post_count","avg_risk","karma"]].mean()
    all_ids = sorted(means.index.tolist())

    # K4 → en az post (pasif)
    k4_id = int(means["post_count"].idxmin())
    rem1 = [i for i in all_ids if i != k4_id]

    # K3 → kalan 3 içinde en yüksek risk (tartışmacı)
    k3_id = int(means.loc[rem1, "avg_risk"].idxmax())
    rem2 = [i for i in rem1 if i != k3_id]

    # K1 → kalan 2 içinde en çok post (yüksek hacimli)
    k1_id = int(means.loc[rem2, "post_count"].idxmax())
    k2_id = [i for i in rem2 if i != k1_id][0]  # etkileşimli konuşmacı

    id_to_k = {k1_id:"K1", k2_id:"K2", k3_id:"K3", k4_id:"K4"}
    k_names = {
        "K1":"Yüksek Hacimli Yayıncı",
        "K2":"Etkileşimli Konuşmacı",
        "K3":"Tartışmacı Profil",
        "K4":"Pasif–Çeşitleyici",
    }
    base["cluster_k"]     = base["cluster_id"].map(id_to_k)
    base["cluster_label"] = base["cluster_k"].map(k_names)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    base["pca_x"] = coords[:,0]
    base["pca_y"] = coords[:,1]

    profile = (
        base.groupby(["cluster_k","cluster_label"])
        .agg(n_agents=("agent_id","count"),
             avg_risk=("avg_risk","mean"),
             avg_post_count=("post_count","mean"),
             avg_comment_count=("comment_count","mean"),
             avg_karma=("karma","mean"))
        .round(4).reset_index().sort_values("cluster_k")
    )
    print("\n  Küme Profilleri:")
    for _, r in profile.iterrows():
        print(f"    {r['cluster_k']} ({r['cluster_label']:<28}) "
              f"ajan={r['n_agents']:>7,}  risk={r['avg_risk']:.3f}  "
              f"post={r['avg_post_count']:.1f}  yorum={r['avg_comment_count']:.1f}")
    return base, profile


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PROPHET  (hacim + risk, %95 CI, kararlı pencere)
# ═══════════════════════════════════════════════════════════════════════════════
def run_prophet(df_posts: pd.DataFrame):
    print("\n▶ Prophet (30 gün, %95 CI)...")

    daily_vol = (
        df_posts.groupby("ds")["id"].count()
        .reset_index().rename(columns={"id":"y"}).sort_values("ds")
    )
    # Kararlı pencere: Nisan ayı (en stabil dönem, günlük ~12-14k post)
    # Son gün eksik veriden çıkar, spike günleri de (medyan x4 üstü) çıkar
    all_days = daily_vol.sort_values("ds").iloc[:-1].copy()  # son gün hariç
    med_all  = float(all_days["y"].median())
    all_days = all_days[all_days["y"] <= med_all * 4]  # spike temizle
    daily_vol = all_days.copy().reset_index(drop=True)

    daily_risk = (
        df_posts.groupby("ds")["risk"].mean()
        .reset_index().rename(columns={"risk":"y"}).sort_values("ds")
    )
    daily_risk = daily_risk.iloc[5:-1].copy().reset_index(drop=True)

    def fit_prophet(df, periods=30, interval=0.95):
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=interval,
        )
        m.fit(df)
        fut = m.make_future_dataframe(periods=periods)
        return m.predict(fut)[["ds","yhat","yhat_lower","yhat_upper"]]

    fc_vol  = fit_prophet(daily_vol)
    fc_risk = fit_prophet(daily_risk)
    fc_vol.rename(columns={"yhat":"vol_fc","yhat_lower":"vol_lo","yhat_upper":"vol_hi"}, inplace=True)
    fc_risk.rename(columns={"yhat":"risk_fc","yhat_lower":"risk_lo","yhat_upper":"risk_hi"}, inplace=True)
    fc = fc_vol.merge(fc_risk, on="ds", how="outer").sort_values("ds")

    hist_end   = pd.Timestamp(daily_vol["ds"].max())
    fc_future  = fc[pd.to_datetime(fc["ds"]) > hist_end]
    cur_vol    = float(daily_vol["y"].iloc[-7:].mean())
    pred_vol   = float(fc_future["vol_fc"].clip(lower=0).iloc[:30].mean())
    cur_risk   = float(daily_risk["y"].mean())
    pred_risk  = float(fc_future["risk_fc"].clip(lower=0).iloc[:30].mean())

    pct_vol  = (pred_vol/max(cur_vol,1)-1)*100
    pct_risk = (pred_risk/max(cur_risk,1e-9)-1)*100

    print(f"  Mevcut günlük post (7G ort.)  : {cur_vol:,.0f}")
    print(f"  30 gün tahmini                : {pred_vol:,.0f}")
    print(f"  Hacim büyümesi                : %{pct_vol:+.1f}")
    print(f"  Platform ort. risk (0-4)      : {cur_risk:.3f}")
    print(f"  30 gün risk tahmini           : {pred_risk:.3f}")
    print(f"  Risk değişim                  : %{pct_risk:+.1f}")

    return fc, daily_vol, daily_risk, hist_end


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FİGÜRLER
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_topic_risk(cat_metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = cat_metrics.sort_values("mean_risk", ascending=False)
    colors = [PAL_CAT[c] for c in cats["category"]]
    bars = ax.bar(range(len(cats)), cats["mean_risk"], color=colors, width=0.6, zorder=3)
    platform_avg = float(cat_metrics["mean_risk"].mean())
    ax.axhline(platform_avg, ls="--", color="#7f8c8d", lw=1.5, label=f"Platform ort. ({platform_avg:.3f})")
    for bar, val in zip(bars, cats["mean_risk"]):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, color="#2c3e50")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(
        [f"{r['category']}\n{r['label']}" for _, r in cats.iterrows()], fontsize=8)
    ax.set_ylabel("Ortalama Risk Skoru (0–4)", fontsize=9)
    ax.set_title("Şekil 1. Konu Kategorisi × Risk Skoru", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(4.2, cats["mean_risk"].max()*1.25))
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    p = PLOTS/"10_fig1_konu_toksisite.png"
    fig.savefig(p); plt.close(fig)
    print(f"  ✓ {p.name}")


def fig2_cluster_profiles(profile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [PAL_CLUSTER[k] for k in profile["cluster_k"]]

    ax = axes[0]
    xs = range(len(profile))
    ax.bar(xs, profile["avg_risk"], color=colors, width=0.5, zorder=3)
    ax.axhline(profile["avg_risk"].mean(), ls="--", color="#7f8c8d", lw=1.4, label="Ort.")
    for i, (_, r) in enumerate(profile.iterrows()):
        ax.text(i, r["avg_risk"]+0.02, f"{r['avg_risk']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['cluster_k']}\n{r['cluster_label']}" for _,r in profile.iterrows()], fontsize=8)
    ax.set_ylabel("Ort. Risk Skoru (0–4)", fontsize=9)
    ax.set_title("(a) Küme × Risk Skoru", fontweight="bold", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.bar(xs, profile["avg_post_count"], color=colors, width=0.5, zorder=3)
    for i, (_, r) in enumerate(profile.iterrows()):
        ax2.text(i, r["avg_post_count"]+0.5, f"{r['avg_post_count']:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{r['cluster_k']}\n{r['cluster_label']}" for _,r in profile.iterrows()], fontsize=8)
    ax2.set_ylabel("Ort. Post Sayısı (ajan başına)", fontsize=9)
    ax2.set_title("(b) Küme × Post Hacmi", fontweight="bold", fontsize=10)
    ax2.yaxis.grid(True, alpha=0.3, zorder=0)

    fig.suptitle("Şekil 2. K-Ortalamalar Küme Profilleri (k=4, n_init=100)", fontweight="bold", fontsize=12)
    plt.tight_layout()
    p = PLOTS/"11_fig2_cluster_profiles.png"
    fig.savefig(p); plt.close(fig)
    print(f"  ✓ {p.name}")


def fig3_prophet_volume(fc, daily_vol, hist_end):
    fig, ax = plt.subplots(figsize=(12, 5))
    fc["ds"] = pd.to_datetime(fc["ds"])
    daily_vol["ds"] = pd.to_datetime(daily_vol["ds"])
    fc_hist   = fc[fc["ds"] <= hist_end]
    fc_future = fc[fc["ds"] >  hist_end]

    ax.fill_between(fc_future["ds"],
                    fc_future["vol_lo"].clip(lower=0), fc_future["vol_hi"],
                    alpha=0.18, color="#e74c3c", label="%95 Güven Aralığı")
    ax.plot(daily_vol["ds"], daily_vol["y"], color="#3498db", lw=2, label="Gerçek hacim")
    ax.plot(fc_hist["ds"], fc_hist["vol_fc"].clip(lower=0),
            color="#95a5a6", lw=1.5, ls=":", label="Model (geçmiş)")
    ax.plot(fc_future["ds"], fc_future["vol_fc"].clip(lower=0),
            color="#e74c3c", lw=2.5, label="30 günlük tahmin")
    ax.axvline(hist_end, color="#f39c12", lw=2, ls="--")
    ymax = float(daily_vol["y"].max()) * 1.3
    ax.text(hist_end, ymax*0.97, "  Bugün", color="#f39c12", fontsize=9, va="top")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Tarih", fontsize=9)
    ax.set_ylabel("Günlük Post Sayısı", fontsize=9)
    ax.set_title("Şekil 3. Prophet 30 Günlük Hacim Projeksiyonu (%95 CI)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    p = PLOTS/"12_fig3_prophet_volume.png"
    fig.savefig(p); plt.close(fig)
    print(f"  ✓ {p.name}")


def fig4_heatmap(df_posts):
    bins   = [0, 0.50, 1.0, 2.0, 3.0, 4.001]
    labels = ["0-Nötr","1-Hafif","2-Orta","3-Yüksek","4-ÇokY"]
    df = df_posts.copy()
    df["risk_level"] = pd.cut(df["risk"], bins=bins, labels=labels, include_lowest=True)
    heat = df.groupby(["category","risk_level"]).size().rename("count").reset_index()
    tot  = df.groupby("category").size().rename("total").reset_index()
    heat = heat.merge(tot, on="category")
    heat["pct"] = (heat["count"]/heat["total"]*100).round(1)
    pivot = heat.pivot_table(index="category", columns="risk_level", values="pct", fill_value=0)
    pivot = pivot[[c for c in labels if c in pivot.columns]]
    pivot = pivot.reindex([c for c in "ABCDEFGHI" if c in pivot.index])
    pivot.index = [f"{c} — {CAT_LABELS[c]}" for c in pivot.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
                linewidths=0.5, cbar_kws={"label":"% Post"}, vmin=0)
    ax.set_xlabel("Risk Seviyesi", fontsize=9)
    ax.set_ylabel("Kategori", fontsize=9)
    ax.set_title("Şekil 4. Kategori × Risk Seviyesi Isı Haritası (%)", fontweight="bold", fontsize=11)
    plt.tight_layout()
    p = PLOTS/"13_fig4_heatmap_topic_toxicity.png"
    fig.savefig(p); plt.close(fig)
    print(f"  ✓ {p.name}")


def fig_pca(df_clustered):
    sample = df_clustered.dropna(subset=["pca_x","pca_y"]).sample(
        min(8000, len(df_clustered)), random_state=42)
    fig, ax = plt.subplots(figsize=(8,6))
    for k, grp in sample.groupby("cluster_k"):
        ax.scatter(grp["pca_x"], grp["pca_y"], c=PAL_CLUSTER[k],
                   label=f"{k} — {grp['cluster_label'].iloc[0]}",
                   alpha=0.35, s=8, zorder=3)
    ax.set_xlabel("PCA Bileşen 1", fontsize=9)
    ax.set_ylabel("PCA Bileşen 2", fontsize=9)
    ax.set_title("K-Ortalamalar Kümeleme (PCA Projeksiyonu)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.2, zorder=0)
    plt.tight_layout()
    p = PLOTS/"04_clustering_pca.png"
    fig.savefig(p); plt.close(fig)
    print(f"  ✓ {p.name}")


def fig_activity(daily_vol):
    dv = daily_vol.copy()
    dv["ds"]  = pd.to_datetime(dv["ds"])
    dv["ma7"] = dv["y"].rolling(7, min_periods=3).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dv["ds"], dv["y"],  color="#3498db", lw=1.2, alpha=0.55, label="Günlük post")
    ax.plot(dv["ds"], dv["ma7"], color="#e74c3c", lw=2.2, label="7G hareketli ort.")
    ax.set_ylabel("Post Sayısı", fontsize=9)
    ax.set_xlabel("Tarih", fontsize=9)
    ax.set_title("Günlük Aktivite Trendi + 7G Hareketli Ortalama", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    p = PLOTS/"01_daily_activity.png"
    fig.savefig(p); plt.close(fig)
    print(f"  ✓ {p.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. METRİK RAPORU
# ═══════════════════════════════════════════════════════════════════════════════
def print_metrics(df_posts, df_agents, df_comments,
                  cat_metrics, profile, fc, daily_vol, daily_risk, hist_end):
    fc_f = fc[pd.to_datetime(fc["ds"]) > hist_end]
    cur_vol   = float(daily_vol["y"].iloc[-7:].mean())
    pred_vol  = float(fc_f["vol_fc"].clip(lower=0).iloc[:30].mean())
    cur_risk  = float(daily_risk["y"].mean())
    pred_risk = float(fc_f["risk_fc"].clip(lower=0).iloc[:30].mean())
    k3 = profile[profile["cluster_k"]=="K3"].iloc[0]

    sep = "═"*65
    print(f"\n{sep}")
    print("  MAKALEYİ GÜNCELLEMEK İÇİN GERÇEK DEĞERLER")
    print(sep)

    print("\n── GENEL ──")
    print(f"  Toplam post                : {len(df_posts):,}")
    print(f"  Toplam ajan                : {len(df_agents):,}")
    print(f"  Toplam yorum               : {len(df_comments):,}")
    print(f"  Gözlem süresi              : {len(daily_vol)} gün (kararlı pencere)")
    print(f"  Tarih aralığı              : {daily_vol['ds'].min().date()} — {daily_vol['ds'].max().date()}")
    print(f"  Platform ort. risk (0-4)   : {cur_risk:.3f}")

    print("\n── KATEGORİ (A-I) TABLO II için ──")
    print(f"  {'Kod':<4} {'Kategori':<22} {'n_post':>10} {'Ort.Risk':>9} {'Medyan':>8} {'Yüksek%':>8}")
    for _, r in cat_metrics.sort_values("mean_risk", ascending=False).iterrows():
        print(f"  {r['category']:<4} {r['label']:<22} {r['n_posts']:>10,} "
              f"{r['mean_risk']:>9.3f} {r['median_risk']:>8.3f} {r['high_risk_pct']:>7.1f}%")

    print("\n── KÜME (TABLO III için) ──")
    print(f"  {'K':4} {'Arketip':<28} {'Ajan':>8} {'AvgRisk':>8} {'AvgPost':>9} {'AvgYorum':>9}")
    for _, r in profile.iterrows():
        print(f"  {r['cluster_k']:<4} {r['cluster_label']:<28} {r['n_agents']:>8,} "
              f"{r['avg_risk']:>8.3f} {r['avg_post_count']:>9.1f} {r['avg_comment_count']:>9.1f}")
    print(f"\n  K3/Platform oran           : {k3['avg_risk']:.3f}/{cur_risk:.3f} = "
          f"{k3['avg_risk']/max(cur_risk,1e-9):.2f}x")

    print("\n── PROPHET (TABLO IV için) ──")
    print(f"  Mevcut günlük post (7G ort.): {cur_vol:,.0f}")
    print(f"  30g tahmin                  : {pred_vol:,.0f}")
    print(f"  Hacim büyümesi              : %{(pred_vol/max(cur_vol,1)-1)*100:+.1f}")
    print(f"  Platform ort. risk          : {cur_risk:.4f}")
    print(f"  30g risk tahmini            : {pred_risk:.4f}")
    print(f"  Risk değişimi               : %{(pred_risk/max(cur_risk,1e-9)-1)*100:+.1f}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    df_posts, df_agents, df_comments = load_data()
    cat_metrics = compute_category_metrics(df_posts)
    df_clustered, profile = run_clustering(df_agents, df_posts, df_comments)
    fc, daily_vol, daily_risk, hist_end = run_prophet(df_posts)

    print("\n▶ Figürler üretiliyor...")
    fig1_topic_risk(cat_metrics)
    fig2_cluster_profiles(profile)
    fig3_prophet_volume(fc, daily_vol, hist_end)
    fig4_heatmap(df_posts)
    fig_pca(df_clustered)
    fig_activity(daily_vol)

    print_metrics(df_posts, df_agents, df_comments,
                  cat_metrics, profile, fc, daily_vol, daily_risk, hist_end)

if __name__ == "__main__":
    main()
