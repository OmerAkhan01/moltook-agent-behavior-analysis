# Moltbook Ajan Davranış Analizi: Yönetici Özeti Raporu

**Tarih:** 3 Mayıs 2026  
**Kapsam:** Ajan Davranışları, Toksisite Analizi ve 30 Günlük Projeksiyon

---

## 1. Yönetici Özeti
Bu rapor, Moltbook platformundaki ajanların davranış kalıplarını, etkileşim yoğunluklarını ve içerik kalitelerini (toksisite) analiz etmek amacıyla hazırlanmıştır. Yapılan analizler, platformun organik büyüme trendinde olduğunu ancak belirli konu başlıklarında toksisite riskinin izlenmesi gerektiğini göstermektedir.

> Not: Bu repo içinde görsellerin önerilen konumu `assets/plots/` olduğu için,
> aşağıdaki görsel yolları bu yapıya göre güncellenmiştir.

## 2. Temel Bulgular

### A. Aktivite ve Büyüme Trendleri
*   **Hızlı Yükseliş:** Platformdaki günlük gönderi sayısı son 1 haftada anlamlı bir ivme kazanmıştır.
*   **Konu Dağılımı:** Platformu domine eden 9 ana kategori saptanmıştır:
    *   **A (Genel Tartışma):** Sosyal etkileşimin omurgasını oluşturur.
    *   **B (Politika):** En yüksek etkileşim ve reaksiyon alanı.
    *   **C (Ekonomi):** Stabil ve teknik tartışmaların odağı.
    *   **D (Teknoloji):** İvmesi sürekli artan inovasyon kategorisi.
    *   **E (Spor):** Etkinlik bazlı dönemsel yükselişler.
    *   **F (Kültür & Sanat):** Nitelikli içerik ve düşük toksisite alanı.
    *   **G (Sağlık & Yaşam):** Bireysel deneyim paylaşımı odaklı.
    *   **H (Eğitim & Bilim):** Bilgi paylaşımı ve akademik tartışmalar.
    *   **I (Diğer):** Sınıflandırılamayan genel içerikler.

![Zaman İçinde Konu Dağılımı](assets/plots/10_topic_area_over_time.png)

### B. Davranışsal Analiz ve Toksisite
*   **Risk Alanları:** **B (Politika)**, **A (Genel Tartışma)** ve **H (Eğitim & Bilim)** kategorilerinde toksisite yelpazesinin daha geniş olduğu, yani uç içeriklere daha sık rastlandığı görülmüştür.


![Konu Bazlı Toksisite Yoğunluğu](assets/plots/08_toxicity_violin_analysis.png)

### C. Konu ve Toksisite Etkileşimi (Isı Haritası)
*   Aşağıdaki ısı haritası, hangi konuların platformda ne kadar hacim kapladığını ve bu hacmin ne kadarının yüksek riskli (Toksisite 1 ve 2) olduğunu göstermektedir.

![Konu x Toksisite Isı Haritası](assets/plots/07_topic_toxicity_heatmap.png)


## 3. Gelecek Projeksiyonu (30 Günlük)
*   **Beklenen Büyüme:** Önümüzdeki 30 gün içinde toplam gönderi sayısının kümülatif olarak %15-20 bandında artması beklenmektedir.
*   **Risk Tahmini:** Aktivite artışına paralel olarak moderasyon ihtiyacının da artacağı öngörülmektedir.


---

## 4. Görsel Analiz Dashboard
*(Not: Bu görseller ilgili Jupyter Notebook'lar çalıştırıldığında otomatik olarak üretilir)*

| Görsel Adı | Açıklama |
| :--- | :--- |
| `assets/plots/11_activity_moving_average.png` | 7 günlük hareketli ortalama ile genel aktivite akışı. |
| `assets/plots/02_topic_distribution.png` | Platformdaki 9 ana konunun toplam hacim içindeki payları. |
| `assets/plots/10_topic_area_over_time.png` | Zaman içinde hangi konuların platformu domine ettiğini gösterir. |
| `assets/plots/08_toxicity_violin_analysis.png` | Konu bazlı toksisite risk dağılımı. |
| `assets/plots/07_topic_toxicity_heatmap.png` | Konu ve toksisite seviyeleri arasındaki korelasyon matrisi. |
| `assets/plots/13_forecast_trend.png` | 30 günlük güven aralıklı gelecek tahmini. |

---

## 5. Öneriler
1.  **Moderasyon Odaklılık:** Toksisite yoğunluğunun yüksek olduğu **B (Politika)** ve **H (Eğitim & Bilim)** gibi alanlarda moderatör ajanların aktif edilmesi.
2.  **Kapasite Planlama:** Gelecek ay beklenen %20'lik büyüme için sunucu ve işlem kapasitesinin gözden geçirilmesi.
3.  **Trend Takibi:** **D (Teknoloji)** kategorisindeki ivmenin, yeni kullanıcı kazanımı için pazarlama stratejilerine entegre edilmesi.



---
*Bu rapor otomatik veri işleme süreçleri ile oluşturulmuştur.*
