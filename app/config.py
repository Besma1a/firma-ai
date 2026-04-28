"""Centralized configuration — all env vars are read here once."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file so it's found regardless of where
# uvicorn is launched from.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # --- LLM provider: Groq (free, fast, multilingual) ---
    groq_api_key: str = ""
    # claude_model keeps its name for back-compat; now holds a Groq model id.
    claude_model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 600

    # --- RAG ---
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_store_dir: str = "./chroma_db"
    kb_dir: str = "./app/data/kb"
    seed_json_path: str = "./app/data/nabta_crops_seed.json"
    top_k: int = 4

    # --- Dev ergonomics ---
    mock_claude: bool = False
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost:8081"

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), case_sensitive=False, extra="ignore")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()