from sqlalchemy import Column, Integer, Text, String, ForeignKey, DateTime, JSON, Float
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    query_text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)
    model_comparison = Column(JSON, nullable=True)
    # [BARU] Algoritma yang dipilih user & dipakai utk klasifikasi query ini
    algorithm = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    articles = relationship("Article", back_populates="query")

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"))
    title = Column(Text)
    url = Column(Text)
    abstract = Column(Text)
    authors = Column(JSON)
    year = Column(String(10))
    subject = Column(String(100))
    jenjang = Column(String(50))
    confidence = Column(Float(4))
    relevance_score = Column(Float(4))
    created_at = Column(DateTime, default=datetime.utcnow)

    query = relationship("Query", back_populates="articles")

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255))
    password = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)