FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install pre-built CPU wheel for llama-cpp-python to keep Docker build lightning fast on Railway
RUN pip install --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python>=0.3.0 \
    fastapi>=0.110.0 \
    uvicorn>=0.28.0 \
    pydantic>=2.0.0 \
    huggingface-hub>=0.20.0

# Download SmolLM2-135M-Instruct-Q4_K_M.gguf (ultra-lightweight ~105MB, uses only ~180MB RAM)
RUN mkdir -p /app/models && \
    python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='bartowski/SmolLM2-135M-Instruct-GGUF', filename='SmolLM2-135M-Instruct-Q4_K_M.gguf', local_dir='/app/models')"

COPY app.py ./

ENV PORT=8000
ENV MODEL_PATH=/app/models/SmolLM2-135M-Instruct-Q4_K_M.gguf
ENV N_THREADS=2

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
