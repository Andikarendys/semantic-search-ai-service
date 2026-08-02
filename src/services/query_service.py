import json
import numpy as np
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity

from src.models import Query

def save_query(db: Session, query: str, embedding, user_id: int, model_comparison: dict = None):
  # embedding ubah ke json
  embedding_json = json.dumps(embedding.tolist())
  
  new_query = Query(
    user_id=user_id,
    query_text=query,
    embedding=embedding_json,
    model_comparison=model_comparison,
  )
  
  db.add(new_query)
  db.commit()
  db.refresh(new_query)
  
  return new_query

def find_most_similar_query(db, new_embedding, threshold, user_id: int):
    queries = db.query(Query).filter(Query.user_id == user_id).all()
    
    best_score = 0
    best_query = None
    
    for q in queries:
        # ambil embedding yg disimpan
        if isinstance(q.embedding, list):
            stored_embedding = np.array(q.embedding)
        else:
            stored_embedding = np.array(json.loads(q.embedding))
        
        score = cosine_similarity(
            [new_embedding],
            [stored_embedding]
        )[0][0]
        
        if score > best_score:
            best_score = score
            best_query = q

    if best_score >= threshold:
        return best_query, best_score
    
    return None, best_score
  