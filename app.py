import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smollm2-135m-gguf-service")

app = FastAPI(title="SmolLM2-135M GGUF Model Service", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/SmolLM2-135M-Instruct-Q4_K_M.gguf")

llm = None

def get_llm():
    global llm
    if llm is None:
        logger.info(f"[STARTUP] Loading SmolLM2-135M GGUF model from {MODEL_PATH}...")
        t0 = time.time()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Ultra-low memory configuration: 135M model with 2048 context takes only ~150MB RAM!
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=int(os.getenv("N_THREADS", "2")),
            n_batch=512,
            verbose=False
        )
        logger.info(f"[STARTUP] SmolLM2-135M GGUF model loaded in {time.time() - t0:.2f}s.")
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
    max_new_tokens: int = 512
    temperature: float = 0.2
    repetition_penalty: float = 1.1

class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    duration_seconds: float

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "smollm2-135m-gguf",
        "model_loaded": llm is not None
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

    system_msg = req.system_prompt or (
        "You are an expert database engineer. Convert the given database tables into clean, valid DBML format "
        "strictly adhering to dbdiagram.io syntax. Mark primary keys as [pk] and place all foreign key references "
        "as 'Ref: table1.col > table2.col' under a '// Relationships' section at the end. Output only raw DBML code."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": req.prompt}
    ]

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

        return GenerateResponse(
            generated_text=output_text.strip(),
            tokens_generated=tokens_gen,
            duration_seconds=round(dur, 2)
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
