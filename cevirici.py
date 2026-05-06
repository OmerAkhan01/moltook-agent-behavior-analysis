import os
import pandas as pd

klasor = "data/processed"
# Sena'nın attığı 4 dosyayı da eksiksiz işliyoruz
dosyalar = ["ajan_kumeleri", "prophet_tahmin", "dil_analizi", "heatmap_analizi"]

for dosya in dosyalar:
    json_yolu = os.path.join(klasor, f"{dosya}.json")
    csv_yolu = os.path.join(klasor, f"{dosya}.csv")
    
    if os.path.exists(json_yolu):
        try:
            df = pd.read_json(json_yolu)
            df.to_csv(csv_yolu, index=False)
            print(f"✅ {dosya}.json başarıyla güncel CSV'ye dönüştürüldü!")
        except Exception as e:
            print(f"❌ {dosya} dönüştürülürken hata çıktı: {e}")
    else:
        print(f"ℹ️ {dosya}.json bulunamadı, atlanıyor.")