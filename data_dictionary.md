# Moltbook Veri Sözlüğü (H6)

Bu belge, temizlenmiş 'moltbook_temiz_final.csv' dosyasındaki verilerin yapısını açıklar.

| Sütun Adı | Açıklama | Veri Tipi | Örnek Veri |
| :--- | :--- | :--- | :--- |
| **id** | Her gönderi için sistem tarafından atanan benzersiz numara. | Integer | 102 |
| **post** | Temizlenmiş ve normalize edilmiş kullanıcı metni. | String | "Örnek içerik..." |
| **timestamp** | Gönderinin paylaşıldığı zaman damgası (UTC). *Not: Veri seti dar bir zaman aralığını kapsamaktadır.* | Datetime | 2026-05-01 |
| **toxic_level** | Gönderinin içerdiği toksisite oranı (0 ile 1 arası). | Float | 0.55 |
| **topic_label** | Gönderinin otomatik olarak atandığı kategori etiketi. | String | "Siyaset" |
