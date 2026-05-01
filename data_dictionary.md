# Moltbook Veri Sözlüğü (H6)

Bu belge, temizlenmiş 'moltbook_temiz.csv' dosyasındaki verilerin yapısını açıklar.

| Sütun Adı | Açıklama | Veri Tipi | Örnek Veri |
| :--- | :--- | :--- | :--- |
| **id** | Her gönderi için sistem tarafından atanan benzersiz numara. | Integer | 102 |
| **post_content** | Temizlenmiş, boşluklardan arındırılmış kullanıcı metni. | String | "Merhaba dünya!" |
| **timestamp** | Gönderinin paylaşıldığı zaman damgası (UTC). | Datetime | 2026-05-01 19:30:00 |
| **toxic_level** | Gönderinin içerdiği toksisite oranı (0 ile 1 arası). | Float | 0.55 |