import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres.gtbdhpbjqpzkxxmupitx")
DB_PASSWORD = os.getenv("DB_PASSWORD", "#SkripsiDB2026")
DB_HOST = os.getenv("DB_HOST", "aws-1-ap-southeast-1.pooler.supabase.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_PORT = os.getenv("DB_PORT", "6543")

try:
    port = int(DB_PORT)
except ValueError:
    port = 6543

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{port}/{DB_NAME}?sslmode=require"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Auto-migration for missing columns in Supabase PostgreSQL
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE queries ADD COLUMN IF NOT EXISTS model_comparison JSON;"))
        conn.commit()
except Exception as e:
    print(f"[DB AUTO-MIGRATION NOTICE] {e}")

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()