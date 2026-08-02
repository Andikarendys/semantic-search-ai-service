from fastapi import APIRouter, Depends, HTTPException
import os
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.orm import Session
from src.database import get_db
from src.models import Article

from src.schemas import QueryRequest, SearchResponse
from src.schemas import SummarizeRequest, SummarizeResponse

from src.services.search_service import search_serpapi
from src.services.query_service import save_query, find_most_similar_query
from src.services.article_service import save_article
from src.services.clasify_service import classify
from src.embed_model import embedding_model
from src.services.summarize_service import summarize_from_url

router = APIRouter()

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

def serialize_article(article: Article) -> dict:
    return {
        "id": article.id,
        "title": article.title,
        "relevance_score": article.relevance_score,
        "url": article.url,
        "abstract": article.abstract,
        "authors": article.authors,
        "year": article.year,
        "subject": article.subject,
        "jenjang": article.jenjang,
        "confidence": article.confidence
    }
    
# Search and Classify Konten
@router.post('/search', response_model=SearchResponse)
def search_content(request: QueryRequest, db: Session = Depends(get_db)):
    user_query = request.query
    user_id = request.user_id

    # Generate embedding query user
    new_embedding = embedding_model.encode(user_query)

    # Cek similarity dengan query sebelumnya
    similar_query, score = find_most_similar_query(
        db,
        new_embedding,
        SIMILARITY_THRESHOLD,
        user_id
    )

    # Jika query mirip, kembalikan hasil cache
    if similar_query:
        articles = db.query(Article).filter(
            Article.query_id == similar_query.id
        ).all()

        if not articles:
            # Cache ada tapi artikelnya kosong, hapus cache lama dan re-fetch
            try:
                db.delete(similar_query)
                db.commit()
            except Exception:
                db.rollback()
        else:
            return {
                "source": "cache",
                "similarity_score": round(score, 4),
                "articles": [serialize_article(a) for a in articles]
            }

    # Hit SerpAPI Google Scholar
    try:
        search_results = search_serpapi(user_query, embedding_model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil data dari SerpAPI: {str(e)}")

    if not search_results:
        return {
            "source": "serpapi",
            "similarity_score": round(score, 4) if score else None,
            "model_comparison": None,
            "articles": []
        }

    # Klasifikasi tiap artikel
    enriched_articles = []

    for item in search_results:
        # Gabung judul + abstract sebagai input klasifikasi
        text = f"{item['title']}. {item['abstract']}"
        
        try:
            result = classify(text)
            subject = result.get("subject", "Umum") if isinstance(result, dict) else "Umum"
            jenjang = result.get("jenjang", "Umum") if isinstance(result, dict) else "Umum"
            confidence = result.get("confidence", 0.85) if isinstance(result, dict) else 0.85
        except Exception:
            subject = "Umum"
            jenjang = "Umum"
            confidence = 0.85

        enriched_articles.append({
            "title": item["title"],
            "relevance_score": item["relevance_score"],
            "url": item["link"],
            "abstract": item["abstract"],
            "authors": item["authors"],
            "year": item["year"],
            "subject": subject,
            "jenjang": jenjang,
            "confidence": confidence
        })

    # Klasifikasi query untuk perbandingan multi-model
    query_class_result = classify(user_query)
    model_comparison_data = query_class_result.get("comparison") if query_class_result else None

    # Simpan query dan artikel ke DB
    saved_query = save_query(db, user_query, new_embedding, user_id, model_comparison=model_comparison_data)
    save_article(db, saved_query.id, enriched_articles)

    # Ambil artikel dari DB biar ada id-nya
    saved_articles = db.query(Article).filter(
        Article.query_id == saved_query.id
    ).all()

    return {
        "source": "serpapi",
        "similarity_score": round(score, 4) if score else None,
        "model_comparison": model_comparison_data,
        "articles": [serialize_article(a) for a in saved_articles]
    }

# Summarize Article
@router.post('/summarize', response_model=SummarizeResponse)
def summarize_article(request: SummarizeRequest):
    try:
        summary = summarize_from_url(request.url)
        return {
            "url": request.url,
            "summary": summary,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses artikel: {str(e)}")