"""
NABTA AI service — FastAPI entrypoint.

Run locally:
    uvicorn app.main:app --reload

Once running, visit:
    http://localhost:8000/docs       (interactive Swagger UI)
    http://localhost:8000/healthz    (liveness + RAG state)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import chat as chat_router
from app.schemas.chat import HealthResponse
from app.services import vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the embedding model + vector store at startup so the first
    # request doesn't pay the cold-start cost (which can be 30+ seconds).
    print("[startup] warming up embedding model + vector store...")
    try:
        vectorstore.get_vector_store()
        n = vectorstore.count_chunks()
        print(f"[startup] vector store ready, {n} chunks indexed")
    except Exception as e:
        print(f"[startup] WARNING: vector store unavailable ({e}). "
              "Run scripts/build_vectorstore.py first.")
    yield
    print("[shutdown] bye")


app = FastAPI(
    title="NABTA AI",
    description="Multilingual agronomic assistant — RAG over Algerian crop knowledge.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)


@app.get("/healthz", response_model=HealthResponse)
def healthz():
    return HealthResponse(
        vector_store_loaded=vectorstore.count_chunks() > 0,
        embedding_model=settings.embed_model,
        claude_model=settings.claude_model,
        mock_mode=settings.mock_claude,
        chunks_indexed=vectorstore.count_chunks(),
    )
