import os
import time
import logging
import threading
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
model_lock = threading.Lock()
is_loading = False

def ensure_model_files():
    """Ensure ONNX and tokenizer exist before loading."""
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

def get_session_and_tokenizer():
    global session, tokenizer, is_loading
    if session is not None and tokenizer is not None:
        return session, tokenizer

    with model_lock:
        if session is not None and tokenizer is not None:
            return session, tokenizer

        is_loading = True
        ensure_model_files()
        logger.info("[STARTUP] Initializing Ultra-Low Memory ONNX Runtime InferenceSession...")
        t0 = time.time()

        # Session options designed strictly for 1GB RAM budget:
        # - Single thread to prevent multi-core buffer duplication
        # - Sequential execution
        # - Disable CPU memory arena pool
        # - Disable graph optimizations (ORT_DISABLE_ALL) to avoid peak RAM allocation during graph rewrites
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        is_loading = False
        logger.info(f"[STARTUP] Model and tokenizer loaded successfully in {time.time() - t0:.2f}s.")
        return session, tokenizer

@app.on_event("startup")
def startup_event():
    # Start loading in background thread so HTTP server starts instantly and passes Railway health check
    threading.Thread(target=get_session_and_tokenizer, daemon=True).start()

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
    return {
        "status": "ok",
        "service": "qwen2.5-0.5b-onnx",
        "model_loaded": session is not None
    }

@app.get("/health")
def health_check():
    # Return 200 immediately so container is marked healthy by Railway
    return {
        "status": "healthy",
        "model_ready": session is not None,
        "is_loading": is_loading
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    sess, tok = get_session_and_tokenizer()

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

    enc = tok.encode(chat_prompt)
    input_ids = np.array([enc.ids], dtype=np.int64)
    seq_len = input_ids.shape[1]
    attn_mask = np.ones((1, seq_len), dtype=np.int64)
    pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    num_layers = 24
    feed = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "position_ids": pos_ids,
    }
    for l in range(num_layers):
        feed[f"past_key_values.{l}.key"] = np.zeros((1, 2, 0, 64), dtype=np.float32)
        feed[f"past_key_values.{l}.value"] = np.zeros((1, 2, 0, 64), dtype=np.float32)

    generated_tokens = []
    eos_token_ids = {151643, 151645}
    token_counter = Counter()

    for step in range(req.max_new_tokens):
        out = sess.run(None, feed)
        logits = out[0][0, -1, :].copy()

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

    del feed
    del out
    import gc
    gc.collect()

    output_text = tok.decode(generated_tokens)
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
