# FIRMA AI  Multilingual RAG Service

The AI core for the NABTA agronomic assistant.


---

## What's inside

```
nabta-ai/
├── README.md                       
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── app/
│   ├── main.py                     ← FastAPI entrypoint
│   ├── config.py                   ← env-var settings (singleton)
│   ├── routers/
│   │   └── chat.py                 ← POST /chat, POST /chat/stream
│   ├── services/
│   │   ├── kb_builder.py           ← JSON → markdown chunks
│   │   ├── vectorstore.py          ← Chroma + embedding model
│   │   ├── claude_client.py        ← Anthropic wrapper + mock mode
│   │   └── rag.py                  ← retrieve → augment → generate
│   ├── schemas/
│   │   └── chat.py                 ← pydantic request/response models
│   └── data/
│       ├── nabta_crops_seed.json   ← source of truth
│       └── kb/                     ← generated chunks (gitignored)
└── scripts/
    ├── build_kb.py                 ← run the KB builder
    ├── build_vectorstore.py        ← create chroma_db/
    ├── smoke_test.py               ← retrieval test (no Claude)
    └── test_chat.py                ← full RAG test (Claude)
```

---



## Quickstart — 6 commands to a working chatbot

```bash
# 1) Setup virtual env + deps
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Configure
cp .env.example .env
# edit .env — paste your ANTHROPIC_API_KEY

# 3) Build the knowledge base (JSON → markdown chunks)
python scripts/build_kb.py
#   → 120 chunks in app/data/kb/

# 4) Build the vector store (markdown → Chroma DB)
python scripts/build_vectorstore.py
#   → ~118 MB embedding model downloaded (first run only)
#   → chroma_db/ folder created

# 5) Smoke-test retrieval (free — no Claude calls)
python scripts/smoke_test.py
#   → should return tomato chunks for tomato questions in any language

# 6) Run the API
uvicorn app.main:app --reload --port 8000
#   → open http://localhost:8000/docs
```
frontend can now POST to `http://localhost:8000/chat`.

---



### Step 1 — `scripts/build_kb.py` (runs `app/services/kb_builder.py`)

**Input:** `app/data/nabta_crops_seed.json` (the structured 10-crop database)

**Output:** 120 markdown files in `app/data/kb/`, one per `(crop × topic × language)`.
Topics are: `planting`, `water`, `diseases`, `yield`. Languages are: `fr`, `ar`, `en`.

**Why:** Vector search needs *text*, not JSON. And smaller, focused chunks
beat one huge document — they let the AI retrieve only the relevant facts
instead of stuffing Claude's context with everything.

You can open any `.md` file by hand and edit a fact. The next run of step 2
picks it up.

### Step 2 — `scripts/build_vectorstore.py` (runs `app/services/vectorstore.py`)

**Input:** the markdown chunks from step 1

**What happens:**
1. Load all `.md` files as LangChain `Document` objects
2. Tag each with metadata `{crop_id, topic, lang}` (parsed from the filename)
3. Split anything over 600 chars into smaller pieces with 80-char overlap
4. Run each chunk through the multilingual embedding model
   → `paraphrase-multilingual-MiniLM-L12-v2` outputs 384-dim vectors
5. Persist `(text + vector + metadata)` triples to `./chroma_db/`

**Output:** the `./chroma_db/` folder. ~5 MB. Ship it with your container.

**Why multilingual embeddings:** the model maps Arabic, French, and Latin
script into the *same vector space*. A Darija question in Arabic letters
will retrieve French chunks if they're the most semantically similar — that's
the magic that makes one knowledge base serve three languages.

### Step 3 — `scripts/smoke_test.py`

**Input:** ad-hoc queries (no Claude)

**What it tests:**
- Cross-language retrieval: same question in 4 languages should retrieve the
  same chunks
- Topic-specific retrieval (e.g. "diseases of date palm")
- Metadata filtering (only Arabic chunks; only tomato chunks)

**Why this is non-negotiable:** if retrieval is broken, RAG is broken. Claude
will hallucinate confidently from wrong context. Always validate retrieval
*before* you turn on the LLM.

### Step 4 — `app/services/rag.py` (the RAG pipeline)

The function `answer(message)` is **literally** the entire pipeline:

```python
retrieved = vectorstore.search(message, k=4)     # 1. RETRIEVE
system    = SYSTEM_PROMPT.format(context=...)    # 2. AUGMENT
text      = claude_client.complete(system, message)  # 3. GENERATE
return {"answer": text, "sources": [...]}
```

**The `SYSTEM_PROMPT_TEMPLATE`** is the most important string in the whole
project. It instructs Claude to:
- Reply in the same language/script as the user wrote in (no MSA when the
  farmer writes in Darija)
- Use only the retrieved context (no hallucinations)
- Be concrete: kg/ha, days, active ingredients (not commercial brand names)
- Refer the farmer to the nearest ITDAS center if the answer isn't in the KB

If the chatbot misbehaves on a specific Darija phrase, you fix it here by
adding a few-shot example. **Never edit Claude logic in the router.**

### Step 5 — `app/routers/chat.py`

Two endpoints:

- `POST /chat` — sync, returns `{answer, sources, model, mock}`. Best for Flutter.
- `POST /chat/stream` — Server-Sent Events. Best for React, gives the typing
  effect.

Both call into `rag.answer()` / `rag.answer_stream()`. The router has no
business logic — that's all in the services layer.

### Step 6 — `app/main.py`

FastAPI bootstrap. Three things to notice:

1. **Lifespan hook** preloads the embedding model + vector store at startup,
   so the first user request doesn't pay a 30-second cold start.
2. **CORS** allows the dev origins listed in `.env` (`localhost:3000` for React,
   `localhost:8081` for Flutter web).
3. **`/healthz`** returns the live state of the RAG (model name, chunks
   indexed, mock mode). Hit it before the demo.

---



Request:
```json
{
  "message": "كيفاش نسقي الطماطم في الصيف؟",
  "user_id": "optional firebase uid",
  "crop_hint": "tomato",
  "lang_filter": "ar",
  "top_k": 4
}
```

Response:
```json
{
  "answer": "في فصل الصيف، اسقي الطماطم بالتنقيط...",
  "sources": [
    {"crop": "tomato", "topic": "water", "lang": "ar", "score": 0.21},
    {"crop": "tomato", "topic": "planting", "lang": "ar", "score": 0.34}
  ],
  "model": "claude-sonnet-4-6",
  "mock": false
}
```

### `POST /chat/stream`

Same request shape. Response is SSE:
```
data: في فصل
data:  الصيف
data: ، اسقي
...
event: done
data: [DONE]
```

### `GET /healthz`

```json
{
  "status": "ok",
  "vector_store_loaded": true,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "claude_model": "claude-sonnet-4-6",
  "mock_mode": false,
  "chunks_indexed": 120
}
```

---

## Deployment

### Local Docker

```bash
docker build -t nabta-ai .
docker run -p 8000:8000 --env-file .env nabta-ai
```
---

##= (after this works)

This module is intentionally limited to chat + RAG. The full NABTA backend
needs four more services. Each one slots in next to `app/services/rag.py`:

1. `app/services/vision.py` — TFLite image classification for `/diagnose`
2. `app/services/weather.py` — OpenWeatherMap proxy + alert rules
3. `app/services/pdf_report.py` — ReportLab weekly summary
4. `app/services/firestore.py` — Firebase admin wrapper for journal/diagnoses


