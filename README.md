# Qwen2.5-Coder-0.5B GGUF Model Microservice

Lightweight CPU-accelerated GGUF microservice for generating and refining DBML schemas. Optimized for Railway container deployment.

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
