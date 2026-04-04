# Moltbook Agent Behavior Analysis 🤖📈

Bu proje, **Moltbook** platformundaki otonom yapay zeka ajanlarının davranış profillerini, zaman içindeki aktivitelerini, etkileşim konularını ve düzeylerini (toksisite) incelemek amacıyla oluşturulmuş bir Veri Analizi ve Gösterge Paneli (Dashboard) projesidir.

## 📊 İçerdiği Analizler
Jupyter Notebook (`01_Veri_Analizi.ipynb`) kullanılarak derinlemesine gerçekleştirilen Keşifçi Veri Analizi (EDA) şunları içerir:
- **Günlük Aktivite Takibi:** Ajanların platformdaki gönderi sıklıklarının zaman serisi analizi.
- **Konu Dağılımı:** Hangi kategorilerde (iş, günlük, teknik vb.) daha çok etkileşimde bulunulduğunun görselleştirilmesi.
- **Toksisite Haritası ve Seviyeleri:** Ajan iletişimlerindeki toksisite düzeyinin (0-4 arası) konularla ilişkisinin incelenmesi.

## 📁 Proje Yapısı

```
moltook-agent-behavior-analysis/
│
├── 01_Veri_Analizi.ipynb         # 📓 Veri temizleme ve Keşifçi Veri Analizi (EDA) merkezi
├── MoltAnalytics/                # 🌐 Streamlit Dashboard ana klasörü
│   ├── app.py                    # Ana sayfa
│   ├── pages/                    # Alt paneller
│   └── src/                      # Arka plan kodları ve veri yükleyiciler
├── Dockerfile                    # 🐳 Çalışma ortamı konteyner yapılandırması
├── docker-compose.yml            # Servis orkestrasyonu (Jupyter & Streamlit)
├── requirements.txt              # Proje Python kütüphane bağımlılıkları
└── .gitignore                    # Dev CSV veri setlerini filtreleyen Git listesi
```

## 🚀 Kurulum ve Başlatma

Bu projeyi bilgisayarınızda çalıştırmak için **Docker** kullanmanız yeterlidir. Proje, iki ayrı servisi (Jupyter Lab ve Streamlit) port çakışması olmadan aynı anda çalıştıracak sekilde yapılandırılmıştır.

1. **Projeyi indirin ve klasöre gidin:**
   ```bash
   git clone https://github.com/OmerAkhan01/moltook-agent-behavior-analysis.git
   cd moltook-agent-behavior-analysis
   ```

2. **Docker konteynerini başlatın:**
   Aşağıdaki komut, tüm bağımlılıkları yükleyip Jupyter ve Streamlit'i başlatacaktır:
   ```bash
   docker-compose up --build -d
   ```

3. **Sayfalara Erişin:**
   - 🧮 **Jupyter Lab (Analiz Ortamı):** `http://localhost:8888` (Şifresiz erişim)
   - 🌐 **Streamlit (Dashboard):** `http://localhost:8501`

Sistemi tamamen kapatmak isterseniz `docker-compose down` komutunu kullanabilirsiniz.

## 🛠️ Kullanılan Teknolojiler
* **Python** (Veri işleme ve betikleme)
* **Pandas & Ast** (Veri manipülasyonu)
* **Matplotlib & Seaborn** (Veri Görselleştirme)
* **Jupyter Notebook** (EDA)
* **Streamlit** (Web UI / Gösterge Paneli)
* **Docker & Docker Compose** (Konteyner Mimarisi)
