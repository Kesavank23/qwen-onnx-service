# SmolLM2-135M-Instruct GGUF Model Microservice

Ultra-lightweight CPU-accelerated GGUF microservice for generating and refining DBML schemas. Optimized for Railway free tier container deployment (<250MB RAM).

## Model Specifications
- **Model**: `SmolLM2-135M-Instruct`
- **Format**: GGUF (`Q4_K_M`)
- **Size on disk**: ~105 MB
- **RAM usage**: ~180 MB – 220 MB
- **Engine**: `llama-cpp-python` (C++ AVX2 optimized)

## Endpoints

- `GET /health`: Returns `{ "status": "healthy", "model_ready": true }`
- `POST /generate`:
  ```json
  {
    "prompt": "create table users with id int and name varchar",
    "system_prompt": "Optional system prompt",
    "max_new_tokens": 512,
    "temperature": 0.2
  }
  ```
