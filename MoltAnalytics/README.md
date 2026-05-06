# MoltAnalytics — Otonom Ajan Davranış Analizi

> **Moltbook** platformundaki yapay zeka ajanlarının davranış profillerini, toksisite eğilimlerini ve gelecek projeksiyonlarını analiz eden uçtan uca veri bilimi + dashboard projesi.

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Canlı Metrikler](#canlı-metrikler)
- [Ekran Görüntüleri](#ekran-görüntüleri)
- [Proje Yapısı](#proje-yapısı)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Analiz Pipeline'ı](#analiz-pipelineı)
- [Dashboard Sayfaları](#dashboard-sayfaları)
- [Veri Akışı](#veri-akışı)

---

## Proje Hakkında

Bu proje, Moltbook sosyal platformundaki otonom yapay zeka ajanlarının:

- **Davranış kalıplarını** kümeleme algoritmaları ile gruplandırmayı
- **Toksisite düzeylerini** konu bazında haritalamayı
- **Gelecek trendlerini** zaman serisi modelleri ile tahmin etmeyi

amaçlamaktadır. Jupyter Notebook tabanlı araştırma ortamı ile Streamlit dashboard'u tek bir repoda bir arada sunar.

---

## Canlı Metrikler

| Metrik | Değer |
|---|---|
| Analiz Edilen Gönderi | **5.000** |
| Yüksek Riskli Gönderi (toksisite ≥ 3) | **395 (%7.9)** |
| Ortalama Toksisite Skoru | **0.55 / 5.0** |
| Şifreli (Base64) İçerik | **12** |
| Davranış Kümesi Sayısı | **4** |
| Prophet Tahmin Ufku | **30 gün** |

---

## Ekran Görüntüleri

### Toksisite Dağılımı
![Toksisite Dağılımı](plots/03_toxicity_distribution.png)

### Günlük Aktivite
![Günlük Aktivite](plots/01_daily_activity.png)

### Konu Dağılımı
![Konu Dağılımı](plots/02_topic_distribution.png)

### Davranış Kümeleri (PCA)
![Kümeleme PCA](plots/04_clustering_pca.png)

### Konu × Toksisite Isı Haritası
![Isı Haritası](plots/07_topic_toxicity_heatmap.png)

### Toksisite Yoğunluk Analizi (Keman Grafiği)
![Keman Grafiği](plots/08_toxicity_violin_analysis.png)

### 30 Günlük Aktivite Tahmini
![Aktivite Tahmini](plots/05_activity_forecast.png)

### 30 Günlük Toksisite Tahmini
![Toksisite Tahmini](plots/06_toxicity_forecast.png)

### Küme × Konu Kompozisyonu
![Küme Konu](plots/09_cluster_topic_composition.png)

---

## Proje Yapısı

```
moltook-agent-behavior-analysis/
│
├── 📓 Jupyter Notebooks
│   ├── 01_Veri_Analizi.ipynb          # Keşifsel veri analizi (EDA)
│   ├── 02_Gonderi_Kumeleme.ipynb      # KMeans kümeleme deneyleri
│   ├── 03_Zaman_Serisi_Tahmin.ipynb   # Prophet tahmin deneyleri
│   └── 04_Yonetici_Ozeti_Raporu.md    # Yönetici özeti
│
├── 🌐 MoltAnalytics/                  # Streamlit Dashboard
│   ├── app.py                         # Ana sayfa (metrikler + navigasyon)
│   ├── pages/
│   │   ├── 01_Davranis.py             # Davranış analizi sayfası
│   │   ├── 02_Kimlikler.py            # Kümeleme & PCA sayfası
│   │   └── 03_Tahmin.py               # Prophet tahmin sayfası
│   └── src/
│       ├── data_loader.py             # Cache'li veri yükleme modülü
│       ├── filters.py                 # Servis katmanı (filtreleme fonksiyonları)
│       ├── models.py                  # KMeans + Prophet model katmanı
│       ├── process_data.py            # Ham veri temizleme
│       ├── fetch_data.py              # HuggingFace veri çekme
│       └── ai_llm_service.py          # AI yorum servisi (mock)
│
├── ⚙️ Pipeline Scriptleri
│   ├── processor_mvp.py               # CSV çıktıları üretir (dashboard için)
│   ├── run_all.py                     # Tam analiz + 300 DPI grafik üretimi
│   └── clean_pipeline.py              # Zaman damgası entegrasyon hattı
│
├── 📊 plots/                          # Üretilen tüm analiz görselleri (300 DPI)
├── 📁 data/
│   ├── raw/                           # Ham veri (git'e eklenmez)
│   └── processed/
│       ├── ajan_kumeleri.csv          # Küme atamaları + PCA koordinatları
│       ├── prophet_tahmin.csv         # 30 günlük toksisite tahmini
│       └── dil_analizi.csv            # Normal / Şifreli içerik özeti
│
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
└── 📋 requirements.txt
```

---

## Kullanılan Teknolojiler

| Katman | Teknoloji | Amaç |
|---|---|---|
| Veri İşleme | `pandas`, `numpy` | DataFrame manipülasyonu |
| Makine Öğrenimi | `scikit-learn` | KMeans, PCA, StandardScaler |
| Zaman Serisi | `prophet` | 30 günlük toksisite tahmini |
| Görselleştirme | `plotly`, `matplotlib`, `seaborn` | İnteraktif ve statik grafikler |
| Dashboard | `streamlit` | Web arayüzü |
| Analiz Ortamı | `jupyterlab` | EDA ve deney notebook'ları |
| Veri Formatı | `pyarrow` | Parquet okuma/yazma |
| Konteyner | `docker`, `docker-compose` | Tekrarlanabilir ortam |

---

## Kurulum ve Çalıştırma

### Yöntem 1 — Docker ile (Önerilen)

```bash
# Repoyu klonla
git clone https://github.com/OmerAkhan01/moltook-agent-behavior-analysis.git
cd moltook-agent-behavior-analysis

# Tüm servisleri başlat
docker-compose up --build
```

| Servis | Adres |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| Jupyter Lab | http://localhost:8888 |

Durdurmak için:
```bash
docker-compose down
```

---

### Yöntem 2 — Direkt Python ile

```bash
# Repoyu klonla
git clone https://github.com/OmerAkhan01/moltook-agent-behavior-analysis.git
cd moltook-agent-behavior-analysis

# Bağımlılıkları kur
pip3 install -r requirements.txt

# Dashboard'u başlat
python3 -m streamlit run MoltAnalytics/app.py
```

Tarayıcı otomatik açılır → `http://localhost:8501`

---

### Analiz Pipeline'ını Çalıştırma

Ham veriden işlenmiş CSV'leri ve grafikleri üretmek için:

```bash
# 1. İşlenmiş CSV'leri üret (dashboard için zorunlu)
python3 processor_mvp.py

# 2. Tam analiz + 300 DPI grafikleri üret
python3 run_all.py
```

---

## Analiz Pipeline'ı

```
Ham Veri (moltbook_raw.csv)
        │
        ▼
clean_pipeline.py ──→ Zaman damgası temizleme + birleştirme
        │
        ▼
processor_mvp.py
  ├── Base64 şifreleme tespiti   ──→ dil_analizi.csv
  ├── KMeans + PCA kümeleme      ──→ ajan_kumeleri.csv
  └── Prophet 30 gün tahmini    ──→ prophet_tahmin.csv
        │
        ▼
run_all.py ──→ plots/ klasörüne 9 grafik (300 DPI)
        │
        ▼
MoltAnalytics Dashboard (Streamlit)
  ├── Ana Sayfa     → Özet metrikler
  ├── Davranış      → Toksisite + içerik dağılımı
  ├── Kimlikler     → PCA scatter + küme profilleri
  └── Tahminler     → Prophet grafik + AI yorum
```

---

## Dashboard Sayfaları

### Ana Sayfa
Platformun anlık durumunu 4 KPI kartıyla özetler: toplam gönderi, yüksek riskli içerik oranı, ortalama toksisite ve küme sayısı.

### Davranış Analizi
| Bileşen | Açıklama |
|---|---|
| Toksisite Bar Grafiği | Her skordaki gönderi sayısını gösterir |
| İçerik Türü Donut | Normal vs Şifreli (Base64) içerik oranı |
| Küme Karşılaştırma | Her kümenin ortalama toksisite düzeyi |
| Özet Tablo | Küme bazında risk istatistikleri |

### Ajan Kimlikleri
| Bileşen | Açıklama |
|---|---|
| PCA Scatter Plot | 5.000 gönderinin 2 boyutlu davranış uzayı |
| Küme Filtresi | Sidebar'dan istenen kümeleri seç/kaldır |
| Profil Tablosu | Küme başına gönderi, toksisite, upvote, şifre oranı |
| Detay Paneli | Tek küme seçildiğinde genişletilmiş istatistik |

### Gelecek Tahminleri
| Bileşen | Açıklama |
|---|---|
| Tahmin Grafiği | yhat çizgisi + %95 güven bandı |
| Tarih Filtresi | Sidebar'dan başlangıç/bitiş tarihi seç |
| Bugün İşareti | Kırmızı kesik çizgi ile geçmiş/gelecek ayrımı |
| Ham Veri | Genişletilebilir tablo (tarih, tahmin, alt/üst sınır) |
| AI Risk Yorumu | Trend özetine dayalı otomatik yorum |

---

## Veri Akışı

```
HuggingFace (AIcell/moltbook-data)
        │
        └──→ fetch_data.py ──→ data/raw/
                                    │
                            processor_mvp.py
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            dil_analizi.csv  ajan_kumeleri.csv  prophet_tahmin.csv
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                            MoltAnalytics/src/
                            data_loader.py (cache)
                            filters.py (filtre)
                            models.py (model)
                                    │
                            Streamlit Sayfaları
```

---

## Lisans

MIT License — © 2026 Sinem Türkoğlu
