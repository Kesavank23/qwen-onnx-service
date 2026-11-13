import os
import time
import logging
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen2.5-coder-0.5b-gguf-service")

app = FastAPI(title="Qwen2.5-Coder-0.5B GGUF Model Service", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", "4"))

llm = None

def get_llm():
    global llm
    if llm is None:
        logger.info(f"[STARTUP] Loading Qwen2.5-Coder-0.5B GGUF model from {MODEL_PATH} (n_ctx={N_CTX}, threads={N_THREADS})...")
        t0 = time.time()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=512,
            verbose=False
        )
        logger.info(f"[STARTUP] Qwen2.5-Coder-0.5B GGUF model loaded in {time.time() - t0:.2f}s.")
    return llm

@app.on_event("startup")
def startup_event():
    try:
        get_llm()
    except Exception as e:
        logger.error(f"[STARTUP ERROR] Could not preload model: {e}")

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.1
    repetition_penalty: float = 1.05

class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    duration_seconds: float

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "qwen2.5-coder-0.5b-gguf",
        "model_loaded": llm is not None,
        "n_ctx": N_CTX
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_ready": llm is not None
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model = get_llm()
    t0 = time.time()

    logger.info(f"[LLM SERVICE] Received generate request | Prompt length: {len(req.prompt)} chars | System prompt length: {len(req.system_prompt or '')} chars")
    
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    try:
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            repeat_penalty=req.repetition_penalty
        )

        output_text = response["choices"][0]["message"]["content"] or ""
        tokens_gen = response.get("usage", {}).get("completion_tokens", len(output_text.split()))
        dur = time.time() - t0

        logger.info(f"[LLM SERVICE] Completed generation in {dur:.2f}s | Output tokens: {tokens_gen}")
        logger.info(f"[LLM SERVICE RAW OUTPUT PREVIEW]: {output_text[:300]}...")

        return GenerateResponse(
            generated_text=output_text.strip(),
            tokens_generated=tokens_gen,
            duration_seconds=round(dur, 2)
        )
    except Exception as e:
        logger.error(f"[LLM SERVICE ERROR]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8006))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
