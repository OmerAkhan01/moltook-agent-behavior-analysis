# Copyright (c) 2026 Sinem Turkoglu - MIT License
# # ==========================================
# Project: Moltbook Data Pipeline (Final Optimized)
# Author: Sinem Türkoğlu
# Description: Metadata integration and timestamp cleaning for Prophet.
# ==========================================

import pandas as pd
import ast
import logging

# İşlem takibi için loglama
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_timestamp(post_content):
    """
    Metadata içerisinden zaman damgasını ayıklar.
    
    Args:
        post_content (str): Ham veri içerisindeki JSON formatlı post sütunu.
        
    Returns:
        str: Ayıklanan created_at bilgisi veya hata durumunda None.
    """
    try:
        # String formatındaki sözlüğü Python sözlüğüne çevirir
        return ast.literal_eval(post_content).get('created_at')
    except Exception:
        return None

def clean_and_integrate_pipeline(cleaned_path, raw_path, output_path):
    """
    Veri temizleme ve zaman serisi entegrasyon hattını çalıştırır. 
    H5 gereksinimlerini karşılamak üzere tasarlanmıştır.
    
    Args:
        cleaned_path (str): Etiketlenmiş/Temizlenmiş ana veri seti yolu.
        raw_path (str): Ham metadata içeren veri seti yolu.
        output_path (str): İşlenmiş final veri setinin kaydedileceği yol.
    """
    logging.info("Sistem: Zaman hassasiyetli veri işleme hattı başlatıldı...")

    try:
        # 1. Veri Okuma
        df_cleaned = pd.read_csv(cleaned_path)
        df_raw = pd.read_csv(raw_path)

        # 2. ID Normalizasyonu (H2 - Eşleşme sorunlarını önlemek için)
        for df in [df_cleaned, df_raw]:
            df['id'] = df['id'].astype(str).str.lower().str.strip()

        # 3. Zaman Damgası Ayıklama (Feature Extraction)
        df_raw['timestamp'] = df_raw['post'].apply(extract_timestamp)

        # 4. Veri Birleştirme (Inner Join)
        df_final = pd.merge(df_cleaned, df_raw[['id', 'timestamp']], on='id', how='inner')

        # 5. Tarih Standardizasyonu (HASSAS AYAR: Saat bilgisi korunur)
        # Prophet'in kısıtlı veride hata vermemesi için saat/dakika silinmez.
        df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])
        
        # 6. Temizlik ve Tekilleştirme
        df_final = df_final.dropna(subset=['timestamp'])
        df_final = df_final.drop_duplicates()

        # 7. Çıktı Üretme (H7)
        df_final.to_csv(output_path, index=False)
        
        logging.info(f"Sonuç: İşlem tamamlandı. {len(df_final)} satır '{output_path}' dosyasına kaydedildi.")

    except Exception as e:
        logging.error(f"Beklenmedik bir hata oluştu: {e}")

if __name__ == "__main__":
    # Dosya isimlerinin repodakilerle birebir aynı olduğundan emin ol
    clean_and_integrate_pipeline('moltbook_final_v4.csv', 'moltbook_temiz.csv', 'moltbook_temiz_final.csv')