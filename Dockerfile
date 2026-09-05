FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_store ./rag_store

ENV HF_HOME=/opt/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV HOST=0.0.0.0
ENV PORT=8080
ENV INDEX_DIR=/data/indexes/msmarco-minilm

EXPOSE 8080
CMD ["python", "-m", "rag_store.server"]
