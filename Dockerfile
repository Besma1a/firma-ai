FROM python:3.11-slim

# System deps for sentence-transformers / chroma
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image so cold starts are fast.
# (~118 MB; happens at build time, not on first request.)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy the rest of the project
COPY . .

# Build the KB and the vector store at image-build time
RUN python scripts/build_kb.py && python scripts/build_vectorstore.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
