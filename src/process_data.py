"""
src/process_data.py
───────────────────
İşlenmiş ham veriyi alır → KMeans + Prophet → Parquet olarak kaydeder.
Çalıştırma: python src/process_data.py
"""
import sys
import gc
import logging
from pathlib import Path

# Allow running as a script: `python src/process_data.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

from src.secret_scrubber import redact_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATASET_DIR   = RAW_DIR / "moltbook-observatory-archive" / "data"

CLUSTER_LABELS = {
    0: "Kaos Elçileri",
    1: "İçerik Üreticiler",
    2: "Pasif Gözlemciler",
    3: "Yankı Odası Üyeleri",
}


def _read_many_parquet(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for f in files:
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Observatory arşivinden ajanlar / postlar / yorumları yükler.
    Not: Bu dataset doğrudan "toxicity" sütunları içermez; sadece score vb. metrikler içerir.
    """
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset klasörü bulunamadı: {DATASET_DIR}\n"
            "Beklenen: data/raw/moltbook-observatory-archive/data/{agents,posts,comments}/"
        )

    agents_files   = sorted((DATASET_DIR / "agents").rglob("*.parquet"))
    posts_files    = sorted((DATASET_DIR / "posts").rglob("*.parquet"))
    comments_files = sorted((DATASET_DIR / "comments").rglob("*.parquet"))

    if not agents_files and not posts_files and not comments_files:
        raise FileNotFoundError(f"Dataset altında parquet bulunamadı: {DATASET_DIR}")

    log.info("agents parquet: %d", len(agents_files))
    log.info("posts parquet: %d", len(posts_files))
    log.info("comments parquet: %d", len(comments_files))

    df_agents   = _read_many_parquet(agents_files)
    df_posts    = _read_many_parquet(posts_files)
    df_comments = _read_many_parquet(comments_files)

    # Redact any accidentally embedded secrets in free-text fields before anything else.
    df_posts = redact_columns(df_posts, ["title", "content", "url"])
    df_comments = redact_columns(df_comments, ["content"])
    df_agents = redact_columns(df_agents, ["name", "description", "owner_x_handle", "avatar_url"])

    log.info("agents:   %s", df_agents.shape)
    log.info("posts:    %s", df_posts.shape)
    log.info("comments: %s", df_comments.shape)
    return df_agents, df_posts, df_comments


def build_agent_features(
    df_agents: pd.DataFrame,
    df_posts: pd.DataFrame,
    df_comments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    KMeans için ajan başına öznitelik tablosu üretir.
    Ayrıca risk sayfası için günlük trend (ds,y) üretir.
    """
    # ---- Normalize IDs ----
    if not df_agents.empty and "id" in df_agents.columns:
        df_agents = df_agents.copy()
        df_agents["agent_id"] = df_agents["id"].astype(str)
    if not df_posts.empty and "agent_id" in df_posts.columns:
        df_posts = df_posts.copy()
        df_posts["agent_id"] = df_posts["agent_id"].astype(str)
    if not df_comments.empty and "agent_id" in df_comments.columns:
        df_comments = df_comments.copy()
        df_comments["agent_id"] = df_comments["agent_id"].astype(str)

    # ---- Aggregates from posts/comments ----
    post_agg = pd.DataFrame()
    if not df_posts.empty:
        df_posts["score"] = pd.to_numeric(df_posts.get("score"), errors="coerce")
        post_agg = (
            df_posts.dropna(subset=["agent_id"])
            .groupby("agent_id")
            .agg(
                post_count=("id", "count"),
                avg_post_score=("score", "mean"),
                max_post_score=("score", "max"),
            )
            .reset_index()
        )

    comment_agg = pd.DataFrame()
    if not df_comments.empty:
        df_comments["score"] = pd.to_numeric(df_comments.get("score"), errors="coerce")
        comment_agg = (
            df_comments.dropna(subset=["agent_id"])
            .groupby("agent_id")
            .agg(
                comment_count=("id", "count"),
                avg_comment_score=("score", "mean"),
                max_comment_score=("score", "max"),
            )
            .reset_index()
        )

    # ---- Base agent table ----
    base = pd.DataFrame()
    if not df_agents.empty and "agent_id" in df_agents.columns:
        keep = [c for c in ["agent_id", "karma", "follower_count", "following_count", "is_claimed", "first_seen_at", "last_seen_at", "created_at"] if c in df_agents.columns]
        base = df_agents[keep].copy()
    else:
        # If agents table is absent, build base from union of seen agent_ids
        ids = pd.Series(dtype="object")
        if not post_agg.empty:
            ids = pd.concat([ids, post_agg["agent_id"]], ignore_index=True)
        if not comment_agg.empty:
            ids = pd.concat([ids, comment_agg["agent_id"]], ignore_index=True)
        base = pd.DataFrame({"agent_id": ids.dropna().astype(str).unique().tolist()})

    # ---- Join aggregates ----
    feats = base
    if not post_agg.empty:
        feats = feats.merge(post_agg, on="agent_id", how="left")
    if not comment_agg.empty:
        feats = feats.merge(comment_agg, on="agent_id", how="left")

    # ---- Derived metrics (from given data only) ----
    for c in ["post_count", "comment_count"]:
        if c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0).astype("int64")
    for c in ["avg_post_score", "avg_comment_score", "max_post_score", "max_comment_score", "karma", "follower_count", "following_count", "is_claimed"]:
        if c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce")

    # Proxy "toxicity_score" as normalized engagement score (keeps UI working without fabricating new data columns)
    # Uses only provided 'score' fields; no external model inference.
    score_cols = [c for c in ["avg_post_score", "avg_comment_score"] if c in feats.columns]
    if score_cols:
        raw_score = pd.to_numeric(feats[score_cols].mean(axis=1, skipna=True), errors="coerce")
        # Heavy-tailed scores can collapse most values near 0. Use log1p + percentile clipping.
        s = np.log1p(raw_score.clip(lower=0).fillna(0.0))
        if np.isfinite(s).any():
            lo = float(np.nanpercentile(s, 1))
            hi = float(np.nanpercentile(s, 99))
            if hi <= lo:
                lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
            s = np.clip(s, lo, hi)
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            feats["toxicity_score"] = (s - lo) / denom
        else:
            feats["toxicity_score"] = 0.0
    else:
        raise ValueError("posts/comments içinde 'score' bulunamadı; ajan özellikleri üretilemedi.")

    # Add a 'ds' column for compatibility (last_seen_at if available; otherwise created_at)
    if "last_seen_at" in feats.columns:
        feats["ds"] = pd.to_datetime(feats["last_seen_at"], errors="coerce")
    elif "created_at" in feats.columns:
        feats["ds"] = pd.to_datetime(feats["created_at"], errors="coerce")
    else:
        feats["ds"] = pd.NaT
    if pd.api.types.is_datetime64_any_dtype(feats["ds"]):
        # Prophet and some parquet consumers prefer tz-naive
        try:
            feats["ds"] = feats["ds"].dt.tz_localize(None)
        except Exception:
            pass

    # ---- Daily trend (from posts+comments timestamps & scores) ----
    trend_frames: list[pd.DataFrame] = []
    for df_src in (df_posts, df_comments):
        if df_src is None or df_src.empty:
            continue
        if "created_at" not in df_src.columns or "score" not in df_src.columns:
            continue
        tmp = df_src[["created_at", "score"]].copy()
        tmp["ds"] = pd.to_datetime(tmp["created_at"], errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(tmp["ds"]):
            try:
                tmp["ds"] = tmp["ds"].dt.tz_localize(None)
            except Exception:
                pass
        tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
        tmp = tmp.dropna(subset=["ds", "score"])
        trend_frames.append(tmp[["ds", "score"]])

    if not trend_frames:
        raise ValueError("Trend için posts/comments içinde 'created_at' ve 'score' bulunamadı.")

    trend_all = pd.concat(trend_frames, ignore_index=True)
    daily = (
        trend_all.groupby(pd.Grouper(key="ds", freq="D"))["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "y"})
        .sort_values("ds")
    )
    # Normalize daily y to 0-1 for UI (still derived only from given scores)
    y = pd.to_numeric(daily["y"], errors="coerce")
    y_min = float(y.min()) if y.notna().any() else 0.0
    y_max = float(y.max()) if y.notna().any() else 1.0
    denom = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
    daily["y"] = ((y - y_min) / denom).fillna(0.0)

    return feats, daily


def run_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    log.info("KMeans başlıyor…")
    # Only use stable numeric features for clustering
    feature_cols = [c for c in [
        "toxicity_score",
        "karma",
        "follower_count",
        "following_count",
        "post_count",
        "comment_count",
        "avg_post_score",
        "avg_comment_score",
        "max_post_score",
        "max_comment_score",
        "is_claimed",
    ] if c in df.columns]
    features = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    # log1p transform for heavy-tailed count/score metrics to stabilize scaling
    for c in [
        "karma",
        "follower_count",
        "following_count",
        "post_count",
        "comment_count",
        "avg_post_score",
        "avg_comment_score",
        "max_post_score",
        "max_comment_score",
    ]:
        if c in features.columns:
            features[c] = np.log1p(pd.to_numeric(features[c], errors="coerce").clip(lower=0).fillna(0.0))

    X = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=4, random_state=42, n_init="auto")
    df = df.copy()
    df["cluster_id"]    = km.fit_predict(X)
    # Cluster-id ordering is arbitrary; map labels deterministically by descending mean toxicity_score
    label_order = ["Kaos Elçileri", "Yankı Odası Üyeleri", "İçerik Üreticiler", "Pasif Gözlemciler"]
    means = df.groupby("cluster_id")["toxicity_score"].mean().sort_values(ascending=False)
    ordered_ids = means.index.tolist()
    id_to_label = {cid: label_order[i] if i < len(label_order) else f"Küme {cid}" for i, cid in enumerate(ordered_ids)}
    df["cluster_label"] = df["cluster_id"].map(id_to_label)
    log.info("Küme dağılımı:\n%s", df["cluster_label"].value_counts().to_string())
    return df


def run_prophet(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Prophet başlıyor…")
    # ds sütunu yoksa ilk datetime sütununu kullan
    date_col = "ds" if "ds" in df.columns else next(
        (c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None
    )
    tox_col  = next(
        (c for c in ["toxicity_score", "toxicity", "score", "label"] if c in df.columns), None
    )
    if not date_col or not tox_col:
        raise ValueError(
            "Prophet için gerekli sütunlar bulunamadı. Beklenen:\n"
            "- Tarih: 'ds' veya datetime tipinde bir sütun\n"
            "- Toksisite: toxicity_score/toxicity/score/label"
        )

    daily = df.copy()
    daily["ds"] = pd.to_datetime(daily[date_col], errors="coerce")
    daily[tox_col] = pd.to_numeric(daily[tox_col], errors="coerce")
    daily = daily.dropna(subset=["ds", tox_col])
    daily = (
        daily.groupby(pd.Grouper(key="ds", freq="D"))[tox_col]
        .mean()
        .reset_index()
        .rename(columns={tox_col: "y"})
        .sort_values("ds")
    )

    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(daily)
    future   = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast.rename(columns={"yhat": "toxicity_forecast"}, inplace=True)
    return forecast


def _proxy_toxicity_from_score(score: pd.Series) -> pd.Series:
    """
    Given raw engagement scores (post/comment), derive a 0-1 proxy toxicity.
    Same idea as agent-level `toxicity_score`, but applied at row level.
    """
    s = pd.to_numeric(score, errors="coerce").fillna(0.0).clip(lower=0.0)
    x = np.log1p(s)
    if not np.isfinite(x).any():
        return pd.Series(0.0, index=s.index)
    lo = float(np.nanpercentile(x, 1))
    hi = float(np.nanpercentile(x, 99))
    if hi <= lo:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    denom = (hi - lo) if (hi - lo) != 0 else 1.0
    return (np.clip(x, lo, hi) - lo) / denom


def build_report_tables(
    df_posts: pd.DataFrame,
    df_comments: pd.DataFrame,
    df_clustered_agents: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rapor görselleriyle (Fig1/Fig2/Fig4) uyumlu özet tablolar üretir.

    - topic_toxicity: submolt bazında ort. toksisite + adet
    - topic_heatmap: submolt × toksisite_seviyesi yüzde dağılımı
    - cluster_profiles: küme bazlı ort. toksisite + konu entropisi + post sayısı
    """
    if df_posts is None or df_posts.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    posts = df_posts.copy()
    if "agent_id" in posts.columns:
        posts["agent_id"] = posts["agent_id"].astype(str)
    posts["submolt"] = posts.get("submolt", "unknown").astype(str).fillna("unknown")
    posts["post_toxicity"] = _proxy_toxicity_from_score(posts.get("score"))

    # ---- Fig1-like: topic avg toxicity ----
    topic_toxicity = (
        posts.groupby("submolt")
        .agg(
            n_posts=("id", "count"),
            mean_toxicity=("post_toxicity", "mean"),
        )
        .reset_index()
        .sort_values("mean_toxicity", ascending=False)
    )

    # ---- Fig4-like: topic x toxicity-level heatmap (% of posts) ----
    # Use 5 bands; align first 3 bands with report thresholds (0.50/0.65) and
    # split upper tail to keep resolution.
    bins = [-float("inf"), 0.50, 0.65, 0.80, 0.90, float("inf")]
    labels = ["0_Nötr", "1_Hafif", "2_Orta", "3_Yüksek", "4_ÇokY"]
    posts["tox_level"] = pd.cut(posts["post_toxicity"], bins=bins, labels=labels)
    heat = (
        posts.groupby(["submolt", "tox_level"])
        .size()
        .rename("count")
        .reset_index()
    )
    tot = posts.groupby("submolt").size().rename("total").reset_index()
    heat = heat.merge(tot, on="submolt", how="left")
    heat["pct"] = (heat["count"] / heat["total"] * 100.0).fillna(0.0)
    topic_heatmap = heat[["submolt", "tox_level", "pct", "count", "total"]].copy()

    # ---- Fig2-like: cluster profiles ----
    cluster_profiles = pd.DataFrame()
    if df_clustered_agents is not None and not df_clustered_agents.empty and "cluster_label" in df_clustered_agents.columns:
        ca = df_clustered_agents[["agent_id", "cluster_label"]].copy()
        ca["agent_id"] = ca["agent_id"].astype(str)
        p2 = posts.merge(ca, on="agent_id", how="inner")

        # mean toxicity per cluster (post-level proxy)
        mean_by_cluster = (
            p2.groupby("cluster_label")["post_toxicity"]
            .mean()
            .rename("cluster_mean_toxicity")
            .reset_index()
        )
        n_posts_by_cluster = (
            p2.groupby("cluster_label")
            .size()
            .rename("n_posts")
            .reset_index()
        )

        # topic entropy per cluster
        # H = -sum p_i log(p_i)
        ct = p2.groupby(["cluster_label", "submolt"]).size().rename("n").reset_index()
        ct["p"] = ct.groupby("cluster_label")["n"].transform(lambda x: x / float(x.sum()) if float(x.sum()) else 0.0)
        ct["plogp"] = ct["p"].where(ct["p"] > 0, 1.0) * np.log(ct["p"].where(ct["p"] > 0, 1.0))
        ent = (
            ct.groupby("cluster_label")["plogp"]
            .sum()
            .mul(-1.0)
            .rename("topic_entropy")
            .reset_index()
        )

        cluster_profiles = mean_by_cluster.merge(ent, on="cluster_label", how="left").merge(n_posts_by_cluster, on="cluster_label", how="left")

    return topic_toxicity, topic_heatmap, cluster_profiles


def save(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df = df.copy()
    # Parquet yazımında "object" karışık tip kolonlar patlamasın diye güvenli string'e çevir
    for col in ["id", "post_id", "parent_id", "agent_id", "agent_name", "submolt", "name", "description", "owner_x_handle", "avatar_url", "url", "title", "content", "dump_date"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")
    log.info("Kaydedildi → %s", path)


def main():
    df_agents, df_posts, df_comments = load_sources()
    df_features, df_daily = build_agent_features(
        df_agents,
        df_posts,
        df_comments,
    )

    df_clustered = run_kmeans(df_features)
    save(df_clustered, "clustered_agents.parquet")
    save(df_daily, "toxicity_daily_trend.parquet")

    # Report-aligned summary tables (topic/cluster profiles)
    topic_tox, topic_heat, cluster_prof = build_report_tables(df_posts, df_comments, df_clustered)
    if not topic_tox.empty:
        save(topic_tox, "topic_toxicity.parquet")
    if not topic_heat.empty:
        save(topic_heat, "topic_toxicity_heatmap.parquet")
    if cluster_prof is not None and not cluster_prof.empty:
        save(cluster_prof, "cluster_profiles.parquet")

    # Optional: Prophet forecast from daily trend
    df_forecast = run_prophet(df_daily.rename(columns={"y": "toxicity_score"}))
    if not df_forecast.empty:
        save(df_forecast, "toxicity_forecast.parquet")

    del df_agents, df_posts, df_comments, df_features, df_daily, df_clustered, df_forecast, topic_tox, topic_heat, cluster_prof
    gc.collect()
    log.info("✅ Pipeline tamamlandı.")


if __name__ == "__main__":
    main()
