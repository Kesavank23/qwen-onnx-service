# Qwen2.5-0.5B ONNX Model Service

A standalone, ultra-lightweight CPU inference microservice for Qwen2.5-0.5B-Instruct running on ONNX Runtime.

## Features
- **Pure Model Microservice**: Contains only the model inference engine (FastAPI + ONNX Runtime).
- **Fast CPU Inference**: INT4-quantized weights running at ~12–15 tokens/sec on standard CPU.
- **Under 500MB RAM**: Low memory footprint ideal for lightweight cloud containers (Railway, Fly.io, Render).
- **Zero Database Dependencies**: No Alembic, Postgres, Redis, or Node.js.

## Endpoints

### 1. `GET /health`
Returns service status.

### 2. `POST /generate`
Generates text/DBML completions from a prompt.

**Request body:**
```json
{
  "prompt": "Table users with id int pk and email varchar",
  "system_prompt": "You are a DBML code generator...",
  "max_new_tokens": 512,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "generated_text": "Table users {\n  id int [pk]\n  email varchar\n}",
  "tokens_generated": 28,
  "duration_seconds": 1.95
}
```

## Deploying to Railway

1. Push this directory to a new GitHub repository (e.g. `qwen-onnx-service`).
2. Log in to [Railway](https://railway.app) and click **New Project** -> **Deploy from GitHub repo**.
3. Select `qwen-onnx-service`.
4. Railway will automatically detect the `Dockerfile` and `railway.json`, build the image, pre-download the ONNX model, and provide a public URL like:
   ```
   https://qwen-onnx-service-production.up.railway.app
   ```
5. In your main API Gateway (`dbNexus-AIIntegration-API`), set:
   ```env
   LOCAL_ONNX_SERVICE_URL=https://qwen-onnx-service-production.up.railway.app
   ```
