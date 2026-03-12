# 1. Pakai sistem operasi Linux + Python 3.11 yang stabil
FROM python:3.11-slim

# 2. Bikin folder kerja di dalam server
WORKDIR /app

# 3. PAKSA install semua komponen layar (GUI) dengan nama versi terbaru
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy kodingan bot kamu
COPY . .

# 6. Nyalakan mesinnya!
CMD ["python", "main.py"]
