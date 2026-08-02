<div align="center">

# 🤖 Klipin AI Service - FastAPI & Machine Learning Microservice
> **Microservice AI Resmi untuk Klasifikasi Kurikulum Merdeka & Peringkasan LLM**

[![English](https://img.shields.io/badge/Language-English-red.svg)](README.md)
[![Bahasa Indonesia](https://img.shields.io/badge/Language-Bahasa_Indonesia-blue.svg)](README.id.md)

</div>

---

Selamat datang di **Repository AI Service** dari **Klipin - Smart Reference & Literature Research System**. Microservice Python FastAPI ini mendukung integrasi pencarian artikel ilmiah (Google Scholar), **Klasifikasi Topik Machine Learning 45-Kelas Kurikulum Merdeka**, ekstraksi teks PDF otomatis, dan peringkasan teks berbasis LLM.

---

## ✨ Kemampuan Utama AI

- **🔍 Crawling Google Scholar**: Memindai artikel akademik via SerpAPI dengan sistem cache kemiripan kata kunci.
- **🧬 Ekstraksi Vektor LaBSE**: Mengubah judul & abstrak menjadi vektor dense 768-dimensi menggunakan `sentence-transformers/LaBSE`.
- **🎯 Klasifikasi Ensemble Multi-Model**: Menguji vektor teks secara simultan pada 4 algoritma (**KNN**, **SVM**, **Logistic Regression**, **Random Forest**) untuk memprediksi subjek (`Biologi`, `Fisika`, `Kimia`, `Matematika`, `IPA`, `IPS`) dan jenjang (`SMA`, `SMP`, `SD`).
- **📄 Deteksi & Ekstraksi PDF Otomatis**: Mendeteksi link PDF naskah (termasuk penerbit OJS) serta memisahkan teks utama dan referensi dengan `PyMuPDF` (`fitz`).
- **🧠 Engine Peringkasan LLM**: Menyintesis dan menerjemahkan dokumen ke Bahasa Indonesia baku terstruktur dalam format JSON via **Groq Cloud API** (`llama-3.3-70b-versatile`).
- **⚡ Kompatibilitas ZeroGPU & CPU**: Berjalan lancar di Hugging Face ZeroGPU Spaces (`@spaces.GPU`) maupun eksekusi CPU.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/), `uvicorn`
- **Machine Learning & NLP**: PyTorch, `sentence-transformers`, `scikit-learn`, `joblib`, `numpy`
- **Pemrosesan Dokumen**: `PyMuPDF` (`fitz`), `beautifulsoup4`, `requests`
- **Integrasi LLM**: Groq Cloud API (`llama-3.3-70b-versatile`)
- **Database & ORM**: PostgreSQL, `SQLAlchemy`, `psycopg2-binary`

---

## ⚙️ Konfigurasi & Cara Menjalankan

### 1. Prasyarat
- **Python** v3.10.x atau v3.11.x
- **pip** dan `venv`

### 2. Langkah Instalasi
```bash
# Clone repository
git clone https://github.com/AI-Research-TS/Ai-Services.git
cd Ai-Services

# Buat & aktifkan virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Konfigurasi Variabel Lingkungan
Buat file `.env` pada folder utama:

```env
SERPAPI_KEY=your_serpapi_key_here
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://postgres:password@db.supabase.co:5432/postgres
```

### 4. Menjalankan Server FastAPI
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Server akan aktif pada: **`http://127.0.0.1:8000`**

---

## 🔗 Endpoint Utama API

| Endpoint | Method | Deskripsi |
| :--- | :--- | :--- |
| `GET /health` | `GET` | Memeriksa status kesehatan service & pemuatan model. |
| `POST /ai/search` | `POST` | Crawling artikel via SerpAPI, ekstraksi LaBSE, dan klasifikasi topik & jenjang. |
| `POST /ai/summarize` | `POST` | Mendeteksi PDF artikel, ekstraksi teks, dan menghasilkan ringkasan LLaMA 3.3. |

---

## 👥 Organisasi
Bagian dari Organisasi GitHub **AI-Research-TS**.
