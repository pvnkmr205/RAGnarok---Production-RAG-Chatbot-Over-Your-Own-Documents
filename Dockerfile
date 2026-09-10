FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer gets cached across rebuilds
# unless requirements.txt actually changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the pre-ingested FastAPI corpus (chunks.json/embeddings.npy)
# as a default so the deployed demo has something to answer immediately,
# without requiring a visitor to upload a file first
COPY *.py ./
COPY chunks.json embeddings.npy corpus_info.json ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]