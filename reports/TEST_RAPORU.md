## MoltAnalytics — Test Raporu

Tarih: 2026-05-06  
Ortam: macOS (darwin) · Python 3.9 · Streamlit multipage  

Bu rapor, projede yapılan değişikliklerden sonra temel çalışırlık (smoke) ve veri sözleşmesi (contract) testlerinin sonuçlarını özetler.

### Kapsam

- **Syntax / import güvenliği** (AST parse)
- **Veri sözleşmesi kontrolü** (`data/processed/*.parquet` kolonları ve dosya varlığı)
- **Pipeline end-to-end çalıştırma** (`python3 src/process_data.py`)

> Not: Bu repo büyük veri ve görselleştirme içerdiği için klasik “unit test” seti yok; burada amaç, uygulamanın çalışması için gereken kritik parçaların doğrulanmasıdır.

---

### 1) Syntax / import (AST Parse) — PASS

Amaç: Dosyaların syntax hatası olmadığını doğrulamak.

Kontrol edilen dosyalar:
- `app.py`
- `pages/1_Davranis_Analizi.py`
- `pages/2_Ajan_Kimlikleri.py`
- `pages/3_Risk_Tahmini.py`
- `src/process_data.py`
- `src/data_loader.py`

Sonuç: **PASS**

---

### 2) Veri sözleşmesi (Data Contract) — PASS

Amaç: Streamlit’in kullandığı işlenmiş çıktıların varlığını ve minimum kolon setini doğrulamak.

Zorunlu dosyalar:
- `data/processed/clustered_agents.parquet`
  - Beklenen kolonlar (min): `agent_id`, `cluster_id`, `cluster_label`, `toxicity_score`
- `data/processed/toxicity_daily_trend.parquet`
  - Beklenen kolonlar: `ds`, `y`

Opsiyonel dosya:
- `data/processed/toxicity_forecast.parquet`
  - Beklenen kolonlar: `ds`, `toxicity_forecast`, `yhat_lower`, `yhat_upper`

Sonuç özeti:
- `clustered_agents.parquet`: shape **(177337, 18)** · missing_cols **[]**
- `toxicity_daily_trend.parquet`: shape **(87, 2)** · missing_cols **[]**
- `toxicity_forecast.parquet`: shape **(117, 4)** · missing_cols **[]**

Sonuç: **PASS**

---

### 3) Pipeline End-to-End (`python3 src/process_data.py`) — PASS (Warning’li)

Amaç: Ham veriden `data/processed` çıktılarının yeniden üretilebildiğini doğrulamak.

Beklenen çıktı dosyaları:
- `clustered_agents.parquet`
- `toxicity_daily_trend.parquet`
- `toxicity_forecast.parquet` (Prophet başarılıysa)

Sonuç: **PASS**

Gözlemler:
- `sklearn` / `prophet` sırasında bazı **RuntimeWarning** mesajları (overflow/invalid) görülebiliyor.
- Buna rağmen pipeline **başarıyla tamamlanıyor** ve Parquet’ler oluşuyor.

---

### 4) `compileall` (bytecode derleme) — SKIP (Ortam kısıtı)

Durum: Bu ortamda Python’ın cache/pyc yazma girişimi **PermissionError** üretebiliyor.

Bu nedenle `compileall` yerine **AST parse** ile syntax doğrulaması kullanıldı.

---

### 5) Kabul Test Planı (Sprint 3–4) — MANUEL KONTROL LİSTESİ

Kaynak: “Test Planı - Sprint 3 (KMeans Kabul Testleri)” ve “Test Planı - Sprint 4 (Prophet Senaryo Testleri)” görselleri.

Bu testler otomatik değil; arayüz davranışı + performans + senaryo doğrulaması içerdiği için **manual** checklist olarak takip edilir.

#### Sprint 3 — KMeans Kabul Testleri

- **TEST-KMEANS-01 (Görsel Render Testi)**  
  - **Amaç**: Scatter noktalarına hover yapınca tooltip’in (ID vb.) takılmadan gelmesi.  
  - **Nasıl test edilir**: `Ajan Kimlikleri` sayfasında scatter üzerinde 10+ noktada hover dene.  
  - **Beklenen**: Tooltip gecikmesi ~0.5s altında ve sayfa donmuyor.  
  - **Durum**: NOT RUN (manual)

- **TEST-KMEANS-02 (Aykırı Değer/Noise Testi)**  
  - **Amaç**: Outlier davranışlarında KMeans merkezlerinin/dağılımın çöküp çökmediğini gözlemlemek.  
  - **Nasıl test edilir**: Pipeline’da outlier simülasyonu/ek satır ile yeniden cluster üret (script ile).  
  - **Beklenen**: Kümeleme çalışır, metrikler NaN/inf’e gitmez, UI kırılmaz.  
  - **Durum**: NOT RUN (manual)

- **TEST-KMEANS-03 (In-Memory Limit / Örnekleme Testi)**  
  - **Amaç**: Çok büyük N’de (örn. 100k+) render kilitlenmesin; sampling devreye girsin.  
  - **Nasıl test edilir**: `Davranış Analizi` / `Ajan Kimlikleri` sayfalarını aç; scatter’ın örneklemle çizildiğini gözlemle.  
  - **Beklenen**: Tarayıcı donmuyor; çizim sayısı sınırlı (örn. 8k/5k).  
  - **Durum**: PARTIAL (kodda sampling var; UI manual doğrulanmadı)

#### Sprint 4 — Prophet Senaryo Testleri

- **TEST-PROPHET-01 (Eşik Aşımı Kabul Testi)**  
  - **Amaç**: Forecast belirli bir “risk eşiği”ni aşarsa UI’da uyarı tetiklensin.  
  - **Nasıl test edilir**: `Risk Tahmini` sayfasında eşiği (varsa) düşürüp/ayarlayıp tahmin bandının aşımını kontrol et.  
  - **Beklenen**: Eşik aşımında görsel/uyarı (st.error veya eşdeğeri) görülür.  
  - **Durum**: NOT RUN (manual)  
  - **Not**: Mevcut sayfada P90/P95 çizgileri var; “st.error uyarısı” istenirse ayrıca eklenebilir.

- **TEST-PROPHET-02 (Tatil/Olay Testi)**  
  - **Amaç**: Ani sıçramaların “event” olarak modele eklenmesiyle aşırı öğrenme azalıyor mu?  
  - **Nasıl test edilir**: Prophet modeline event regressors eklenen bir varyant pipeline çalıştırılır (ayrı geliştirme).  
  - **Beklenen**: Overfitting azalır; forecast daha stabil.  
  - **Durum**: NOT IMPLEMENTED (ilerleme.txt’de hedef; bu repoda uygulanmadı)

- **TEST-CACHE-01 (Backend Önbellek Testi)**  
  - **Amaç**: DB olmadan cache doğru çalışıyor mu; RAM limiti aşılmıyor mu?  
  - **Nasıl test edilir**: Streamlit’i aç, sayfalar arası gezin, aynı sayfayı tekrar aç; diskten yeniden okuma yerine cache kullanıldığını gözlemle.  
  - **Beklenen**: Yeniden yüklemeler hızlanır; uygulama çökmez.  
  - **Durum**: PARTIAL (cache dekoratörleri mevcut; cloud RAM ölçümü manual)

---

### Açıklama / Yorum

- Bu projede “Skor/Toksisite” değerleri, kullanılan veri kaynağına göre değişebilir:
  - Observatory arşiv verisi toksisite alt-boyutları (threat/identity_attack vb.) içermez.
  - Bu durumda sayfalar **ajan metrikleri + proxy skor** üzerinden çalışacak şekilde tasarlanmıştır.
- Risk sayfasında Plotly’nin bazı yardımcı fonksiyonları (`add_hline` anotasyonlu) datetime ekseninde hata verebildiği için
  eşik çizgileri `add_shape` + `add_annotation` ile çizilecek şekilde stabilize edilmiştir.

