import os
import time
import logging
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen-onnx-service")

app = FastAPI(title="Qwen2.5-0.5B ONNX Model Service", version="1.0.0")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/qwen2.5-0.5b")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "onnx", "model_q4.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

session = None
tokenizer = None

def ensure_model_exists():
    """Download model if not present."""
    os.makedirs(os.path.join(MODEL_DIR, "onnx"), exist_ok=True)
    if not os.path.exists(ONNX_MODEL_PATH):
        logger.info("[INIT] Downloading model_q4.onnx from Hugging Face...")
        hf_hub_download(
            repo_id="onnx-community/Qwen2.5-0.5B-Instruct",
            filename="onnx/model_q4.onnx",
            local_dir=MODEL_DIR
        )
    if not os.path.exists(TOKENIZER_PATH):
        logger.info("[INIT] Downloading tokenizer.json from Hugging Face...")
        hf_hub_download(
            repo_id="onnx-community/Qwen2.5-0.5B-Instruct",
            filename="tokenizer.json",
            local_dir=MODEL_DIR
        )

@app.on_event("startup")
def load_model():
    global session, tokenizer
    ensure_model_exists()
    logger.info("[STARTUP] Initializing ONNX Runtime InferenceSession...")
    t0 = time.time()
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    logger.info(f"[STARTUP] Model and tokenizer loaded successfully in {time.time() - t0:.2f}s.")

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    repetition_penalty: float = 1.2

class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    duration_seconds: float

@app.get("/")
def root():
    return {"status": "ok", "service": "qwen2.5-0.5b-onnx"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if session is not None else "initializing",
        "model": "Qwen2.5-0.5B-Instruct-ONNX"
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if session is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model session is not ready yet.")

    t0 = time.time()
    system_msg = req.system_prompt or (
        "You are an expert database engineer. Convert the given database tables into clean, valid DBML format "
        "strictly adhering to dbdiagram.io syntax. Mark primary keys as [pk] and place all foreign key references "
        "as 'Ref: table1.col > table2.col' under a '// Relationships' section at the end. Output only raw DBML code."
    )

    chat_prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{req.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    enc = tokenizer.encode(chat_prompt)
    input_ids = np.array([enc.ids], dtype=np.int64)
    seq_len = input_ids.shape[1]
    attn_mask = np.ones((1, seq_len), dtype=np.int64)
    pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    num_layers = 24  # 48 past_key_values inputs = 24 layers * (key, value)
    feed = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "position_ids": pos_ids,
    }
    for l in range(num_layers):
        feed[f"past_key_values.{l}.key"] = np.zeros((1, 2, 0, 64), dtype=np.float32)
        feed[f"past_key_values.{l}.value"] = np.zeros((1, 2, 0, 64), dtype=np.float32)

    generated_tokens = []
    eos_token_ids = {151643, 151645}  # <|endoftext|>, <|im_end|>
    token_counter = Counter()

    for step in range(req.max_new_tokens):
        out = session.run(None, feed)
        logits = out[0][0, -1, :].copy()

        # Repetition penalty
        if req.repetition_penalty != 1.0 and len(token_counter) > 0:
            for tid, count in token_counter.items():
                if logits[tid] > 0:
                    logits[tid] /= (req.repetition_penalty ** count)
                else:
                    logits[tid] *= (req.repetition_penalty ** count)

        next_token = int(np.argmax(logits))
        if next_token in eos_token_ids:
            break

        generated_tokens.append(next_token)
        token_counter[next_token] += 1

        cur_seq_len = attn_mask.shape[1]
        attn_mask = np.ones((1, cur_seq_len + 1), dtype=np.int64)
        pos_ids = np.array([[cur_seq_len]], dtype=np.int64)

        feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": attn_mask,
            "position_ids": pos_ids,
        }
        for l in range(num_layers):
            feed[f"past_key_values.{l}.key"] = out[1 + 2 * l]
            feed[f"past_key_values.{l}.value"] = out[2 + 2 * l]

    output_text = tokenizer.decode(generated_tokens)
    dur = time.time() - t0

    return GenerateResponse(
        generated_text=output_text.strip(),
        tokens_generated=len(generated_tokens),
        duration_seconds=round(dur, 2)
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
