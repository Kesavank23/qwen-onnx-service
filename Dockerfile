FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model during build to ensure fast container boot
RUN mkdir -p /app/models/qwen2.5-0.5b/onnx && \
    python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct-ONNX', filename='onnx/model_q4.onnx', local_dir='/app/models/qwen2.5-0.5b'); \
hf_hub_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct-ONNX', filename='tokenizer.json', local_dir='/app/models/qwen2.5-0.5b')"

COPY app.py ./

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
