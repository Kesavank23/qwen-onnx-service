FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download SmolLM2-135M-Instruct ONNX (180MB model, runs cleanly under 350MB RAM)
RUN mkdir -p /app/models/smollm2-135m/onnx && \
    python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='onnx-community/SmolLM2-135M-Instruct-ONNX', filename='onnx/model_q4.onnx', local_dir='/app/models/smollm2-135m'); \
hf_hub_download(repo_id='onnx-community/SmolLM2-135M-Instruct-ONNX', filename='tokenizer.json', local_dir='/app/models/smollm2-135m'); \
hf_hub_download(repo_id='onnx-community/SmolLM2-135M-Instruct-ONNX', filename='config.json', local_dir='/app/models/smollm2-135m')"

COPY app.py ./

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
