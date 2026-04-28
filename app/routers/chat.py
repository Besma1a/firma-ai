"""HTTP endpoints for the chat / RAG feature."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services import rag

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Synchronous RAG answer. Use this from Flutter where SSE is awkward."""
    try:
        result = rag.answer(
            message=req.message,
            top_k=req.top_k,
            lang_filter=req.lang_filter,
            crop_filter=req.crop_hint,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failure: {e}") from e


@router.post("/stream")
def chat_stream(req: ChatRequest):
    """Server-sent events stream of the answer. Use from React."""
    def event_generator():
        try:
            for delta in rag.answer_stream(
                    message=req.message,
                    top_k=req.top_k,
                    lang_filter=req.lang_filter,
                    crop_filter=req.crop_hint):
                # SSE frame format: "data: <json-or-text>\n\n"
                yield f"data: {delta}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
