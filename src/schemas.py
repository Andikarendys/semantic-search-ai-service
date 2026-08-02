from pydantic import BaseModel
from typing import List, Optional

# Skema pencarian (query)
class QueryRequest(BaseModel):
  query: str
  user_id: int
  
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
    source: str
    similarity_score: Optional[float] = None
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