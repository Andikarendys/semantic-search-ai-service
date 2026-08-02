<div align="center">

# 🤖 Klipin AI Service - FastAPI & Machine Learning Microservice
> **Official AI Microservice for Kurikulum Merdeka Classification & LLM Summarization**

[![English](https://img.shields.io/badge/Language-English-red.svg)](README.md)
[![Bahasa Indonesia](https://img.shields.io/badge/Language-Bahasa_Indonesia-blue.svg)](README.id.md)

</div>

---

Welcome to the **AI Service Repository** of **Klipin - Smart Reference & Literature Research System**. This Python FastAPI microservice powers literature search integration (Google Scholar), **45-Class Kurikulum Merdeka Machine Learning Topic Classification**, automated PDF text extraction, and LLM text summarization.

---

## ✨ Key AI Capabilities

- **🔍 Google Scholar Crawling**: Crawls academic papers via SerpAPI with intelligent query similarity caching.
- **🧬 LaBSE Feature Extraction**: Encodes paper titles & abstracts into 768-dimensional dense vector embeddings using `sentence-transformers/LaBSE`.
- **🎯 Multi-Model Ensemble Classification**: Evaluates paper embeddings simultaneously across 4 algorithms (**KNN**, **SVM**, **Logistic Regression**, **Random Forest**) to predict subject (`Biology`, `Physics`, `Chemistry`, `Math`, `IPA`, `IPS`) and level (`SMA`, `SMP`, `SD`).
- **📄 PDF Auto-Discovery & Extraction**: Automatically locates direct PDF links (including OJS publisher sites) and parses full-text & references using `PyMuPDF` (`fitz`).
- **🧠 LLM Summarization Engine**: Synthesizes and translates papers into formal Indonesian structured JSON schemas via **Groq Cloud API** running **LLaMA 3.3 70B Versatile**.
- **⚡ ZeroGPU & CPU Resilient**: Compatible with Hugging Face ZeroGPU Spaces (`@spaces.GPU`) and CPU fallback execution.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/), `uvicorn`
- **Machine Learning & NLP**: PyTorch, `sentence-transformers`, `scikit-learn`, `joblib`, `numpy`
- **Document Processing**: `PyMuPDF` (`fitz`), `beautifulsoup4`, `requests`
- **LLM Integration**: Groq Cloud API (`llama-3.3-70b-versatile`)
- **Database & ORM**: PostgreSQL, `SQLAlchemy`, `psycopg2-binary`

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- **Python** v3.10.x or v3.11.x
- **pip** and `venv`

### 2. Installation Steps
```bash
# Clone repository
git clone https://github.com/AI-Research-TS/Ai-Services.git
cd Ai-Services

# Create & activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Create a `.env` file in the root directory:

```env
SERPAPI_KEY=your_serpapi_key_here
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://postgres:password@db.supabase.co:5432/postgres
```

### 4. Start Development Server
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
The server will start at: **`http://127.0.0.1:8000`**

---

## 🔗 Main API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /health` | `GET` | Service health check & model loading status. |
| `POST /ai/search` | `POST` | Crawls papers via SerpAPI, embeds via LaBSE, and classifies topic & level. |
| `POST /ai/summarize` | `POST` | Discovers article PDF, extracts text, and generates LLaMA 3.3 summary. |

---

## 👥 Organization
Part of **AI-Research-TS** GitHub Organization.
