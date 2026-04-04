# MoltAnalytics

MoltAnalytics, "Moltbook" platformundaki otonom yapay zeka ajanlarının davranışlarını analiz eden, kümeleyen ve toksisite tahmini yapan tam donanımlı bir veri dashboard projesidir.

## Kurulum
1. `pip install -r requirements.txt`
2. `streamlit run app.py` ile projeyi başlatın.

## Özellikler
- **Davranış Analizi:** Saatlik aktivite ve gizli dil (Base64) kullanım analizi.
- **Kimlikler:** Ajanların davranışlarına göre KMeans ile kümelenmesi.
- **Tahmin:** Prophet ile 30 günlük toksisite tahmini.
