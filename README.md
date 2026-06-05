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
│   │   ├── claude_client.py        ←
│   │   └── rag.py                  ← retrieve → augment → generate
│   ├── schemas/
│   │   └── chat.py                 ← pydantic request/response models
│   └── data/
│       ├── nabta_crops_seed.json   ← source of truth
│       └── kb/                     ← generated chunks (gitignored)
└── scripts/
    ├── build_kb.py                 ← run the KB builder
    ├── build_vectorstore.py        ← create chroma_db/
    ├── smoke_test.py               ← retrieval test 
    └── test_chat.py                ← full RAG test 
```

---





