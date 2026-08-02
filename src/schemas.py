from pydantic import BaseModel
from typing import List, Optional, Any

# Skema pencarian (query)
class QueryRequest(BaseModel):
  query: str
  user_id: int
  # [BARU] Algoritma yang dipilih user SEBELUM melakukan pencarian.
  # None -> pakai DEFAULT_ALGORITHM (svm) di clasify_service.py
  algorithm: Optional[str] = None

# Response per artikel
class ArticleResponse(BaseModel):
    id: int
    title: str
    url: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = []
    year: Optional[str] = None
    subject: Optional[str] = None
    jenjang: Optional[str] = None
    relevance_score: Optional[float] = None
    confidence: Optional[float] = None

    class Config:
        from_attributes = True

# Response utama endpoint /search
class SearchResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # [FIX] hilangkan warning field model_comparison

    source: str
    similarity_score: Optional[float] = None
    # [FIX PENTING] Field ini SEBELUMNYA TIDAK ADA di skema, padahal router.py
    # sudah mengirimnya di response dict -- FastAPI otomatis membuang field
    # yang tidak terdaftar di response_model, jadi model_comparison SELALU
    # hilang sebelum sampai ke frontend. Sekarang didaftarkan eksplisit.
    algorithm: Optional[str] = None
    model_comparison: Optional[List[dict]] = None
    articles: List[ArticleResponse]

class SummarizeRequest(BaseModel):
    url: str

class SummaryDetail(BaseModel):
    identitas: str = ""
    author: str = ""
    kompetensi: str = ""
    isi_materi: str = ""
    temuan: str = ""
    kesimpulan: str = ""
    daftar_pustaka: str = ""

class SummarizeResponse(BaseModel):
    url: str
    summary: SummaryDetail