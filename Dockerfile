FROM python:3.10-slim

WORKDIR /app

# 1. Gerekli sistem araçlarını ve derleyicileri kur (Çok Önemli!)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Pip, setuptools ve wheel araçlarını güncelle
# Hata mesajındaki 'metadata' sorunu genellikle wheel eksikliğinden çıkar
RUN pip install --upgrade pip setuptools wheel

# 3. Kütüphaneleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Proje dosyalarını kopyala
COPY . .

EXPOSE 8888 8501

# Jupyter Lab'i başlat
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]