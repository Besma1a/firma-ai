# NABTA AI Service — Django Integration Handoff

## What this service does

NABTA AI is a FastAPI microservice (port 8000) that answers multilingual agronomic
questions from Algerian farmers. It uses a RAG pipeline: the user's question is
embedded, matched against a Chroma vector store of 250+ crop knowledge-base chunks
(Arabic / French / English), and the top matching chunks are fed to a Groq-hosted
Llama 3.3 model which generates the final answer. Replies are automatically in the
same language the user wrote in (Darija, MSA, French, or English).

---

## Endpoints

### `POST /chat` — synchronous answer

**Request**
```json
{
  "message": "كيفاش نقاوم ذبابة الزيتون؟",
  "top_k": 6,
  "lang_filter": null,
  "crop_hint": null
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | User's question (any language) |
| `top_k` | int | no | Number of KB chunks to retrieve (default: 6) |
| `lang_filter` | string | no | `"ar"`, `"fr"`, or `"en"` — restrict KB search to one language |
| `crop_hint` | string | no | e.g. `"olive"` — restrict search to one crop |

**Response**
```json
{
  "answer": "باش تقاوم ذبابة الزيتون، رش سبينوساد كل 14 يوم من يوليو...",
  "sources": [
    {"crop": "olive", "topic": "treatment", "lang": "ar", "score": 0.87}
  ],
  "model": "llama-3.3-70b-versatile",
  "mock": false
}
```

---

### `POST /chat/stream` — streaming answer (Server-Sent Events)

Same request body as `/chat`. Response is an SSE stream:

```
data: باش تقاوم

data:  ذبابة الزيتون،

data:  رش سبينوساد...

event: done
data: [DONE]
```

Use this from a React/Vue frontend. Avoid it from Flutter (SSE support is awkward there — use `/chat` instead).

---

### `GET /healthz` — liveness check

**Response**
```json
{
  "vector_store_loaded": true,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "claude_model": "llama-3.3-70b-versatile",
  "mock_mode": false,
  "chunks_indexed": 270
}
```

Call this from your Django health check or load balancer probe.

---

## Environment variables

| Variable | Required | Example | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **yes** | `gsk_...` | Free key from console.groq.com/keys |
| `CLAUDE_MODEL` | no | `llama-3.3-70b-versatile` | Groq model ID |
| `MOCK_CLAUDE` | no | `false` | Set `true` to skip LLM calls (stub responses) |
| `TOP_K` | no | `6` | Chunks retrieved per query |
| `MAX_TOKENS` | no | `600` | Max LLM output tokens |
| `CORS_ORIGINS` | no | `http://localhost:3000` | Comma-separated allowed origins |

Copy `.env.example` → `.env` and fill in `GROQ_API_KEY`.

---

## How to start the service

```bash
# Install dependencies
pip install -r requirements.txt

# Build the knowledge base (only needed once, or after data changes)
python scripts/build_kb.py
python scripts/build_vectorstore.py

# Start the server
uvicorn app.main:app --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

---

## Two integration patterns

### Pattern A — Frontend hits FastAPI directly

```
Browser / Flutter  ──POST /chat──►  FastAPI :8000
```

Add `http://your-django-domain.com` to `CORS_ORIGINS` in `.env`.
No Django changes needed for read-only chat.

---

### Pattern B — Django proxies the AI service (recommended for auth / rate limiting)

```
Browser  ──►  Django view  ──requests──►  FastAPI :8000
```

Add this view to your Django app:

```python
# views.py
import requests
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

AI_SERVICE_URL = "http://localhost:8000"

@csrf_exempt
def chat_proxy(request):
    """Proxy POST /api/chat → FastAPI /chat."""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        resp = requests.post(
            f"{AI_SERVICE_URL}/chat",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return JsonResponse(resp.json())
    except requests.Timeout:
        return JsonResponse({"error": "AI service timeout"}, status=504)
    except requests.RequestException as exc:
        return JsonResponse({"error": str(exc)}, status=502)
```

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("api/chat/", views.chat_proxy),
]
```

---

## Quick smoke test

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "كيفاش نقاوم ذبابة الزيتون؟"}'
```

Expected: JSON with `answer` in Arabic Darija, `sources` listing olive treatment chunks.
