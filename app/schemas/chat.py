"""Pydantic models for /chat request and response payloads."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         description="Farmer's question, any language.")
    user_id: Optional[str] = Field(None, description="Firebase uid, optional.")
    crop_hint: Optional[str] = Field(
        None, description="Restrict retrieval to a specific crop (e.g. 'tomato').")
    lang_filter: Optional[Literal["ar", "fr", "en"]] = Field(
        None, description="If set, only retrieve chunks of this language.")
    top_k: Optional[int] = Field(
        None, ge=1, le=10, description="Override the default number of chunks.")


class Source(BaseModel):
    crop: str
    topic: str
    lang: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    mock: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    vector_store_loaded: bool
    embedding_model: str
    claude_model: str
    mock_mode: bool
    chunks_indexed: int
