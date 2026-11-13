import os
import re
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

def parse_sql_or_dbml(text: str) -> str:
    # If already pure DBML Table syntax
    if "Table " in text:
        start = text.find("Table ")
        extracted = text[start:].strip()
        if "```" in extracted:
            extracted = extracted.split("```")[0].strip()
        return extracted
    
    # If output was SQL CREATE TABLE
    m = re.search(r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);?', text, re.DOTALL | re.IGNORECASE)
    if m:
        table_name = m.group(1)
        body = m.group(2)
        cols = []
        for line in body.split('\n'):
            line = line.strip().rstrip(',')
            if not line or line.upper().startswith(('FOREIGN KEY', 'PRIMARY KEY (', 'CONSTRAINT')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0]
                raw_type = parts[1].lower()
                col_type = 'int' if 'int' in raw_type else ('date' if 'date' in raw_type else 'varchar')
                pk = ' [pk]' if 'primary key' in line.lower() else ''
                cols.append(f"  {col_name} {col_type}{pk}")
        if not cols:
            cols.append("  id int [pk]")
        return f"Table {table_name} {{\n" + "\n".join(cols) + "\n}"
    return ""

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

    system_msg = req.system_prompt or "You are an expert database assistant."
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
