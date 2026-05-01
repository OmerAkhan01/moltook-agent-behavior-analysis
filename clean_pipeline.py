# Copyright (c) 2026 Sinem Turkoglu - MIT License
# # ==========================================
# Proje: Moltbook Veri İşleme Hattı (H5)
# Hazırlayan: Sinem Türkoğlu
# Açıklama: Ham verilerin temizlenmesi ve standartlaştırılması
# ==========================================

import pandas as pd

def clean_pipeline():
    """
    H5 Gereksinimi: Veri temizleme adımlarını birleştirir ve 
    otomatik bir işleme hattı (pipeline) oluşturur.
    """
    print("Sistem: Veri temizleme işlemi başlatıldı...")

    try:
        # 1. Veri Okuma (H1)
        # Ham veri dosyasını sisteme yüklüyoruz.
        df = pd.read_csv('moltbook_ham.csv')
        
    
     # 2. Eksik Veri Temizliği (H2)
        # İçeriği boş olan veya analiz edilemeyecek satırları kaldırıyoruz.
        df = df.dropna()
        
        # 3. Zaman Damgası Standardizasyonu (H1)
        # Tarih formatlarını herkes için standart olan UTC formatına çeviriyoruz.
        time_columns = ['timestamp', 'created_at', 'date']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        
        # 4. Yinelenen Kayıtların Temizlenmesi (H2)
        # Analiz sonuçlarını bozmaması için mükerrer verileri siliyoruz.
        df = df.drop_duplicates()

        # 5. Çıktı Üretme (H2 & H7)
        # İşlenen temiz veriyi yeni bir CSV dosyası olarak dışa aktarıyoruz.
        output_file = 'moltbook_temiz.csv'
        df.to_csv(output_file, index=False)
        
        print("-" * 40)
        print("Sonuç: İşlem başarıyla tamamlandı.")
        print(f"Bilgi: Temizlenmiş veri '{output_file}' dosyasına kaydedildi.")
        print("-" * 40)

    except FileNotFoundError:
        print("Hata: 'moltbook_ham.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")

if __name__ == "__main__":
    clean_pipeline()
