#!/usr/bin/env python3
"""Batch embedding server — POST /api/embed (ollama-compat), GPU via sentence-transformers."""
import os, time, logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("embed-server")

MODEL_ID   = os.getenv("EMBED_MODEL",  "nomic-ai/nomic-embed-text-v1.5")
DEVICE     = os.getenv("EMBED_DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "512"))

log.info(f"Loading {MODEL_ID} on {DEVICE} ...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_ID, trust_remote_code=True, device=DEVICE)
model.eval()
log.info("Model ready.")

app = FastAPI()

class EmbedRequest(BaseModel):
    model: str = ""
    input: List[str]

@app.post("/api/embed")
def embed(req: EmbedRequest):
    t0 = time.time()
    vecs = model.encode(req.input, batch_size=BATCH_SIZE, normalize_embeddings=True,
                        show_progress_bar=False)
    dt = time.time() - t0
    log.info(f"embedded {len(req.input)} in {dt:.3f}s ({len(req.input)/dt:.0f}/s)")
    return JSONResponse({"embeddings": vecs.tolist()})

@app.get("/api/tags")
def tags():
    return {"models": [{"name": MODEL_ID}]}

@app.get("/api/ps")
def ps():
    import torch
    mb = torch.cuda.memory_allocated() // 1024 // 1024 if DEVICE == "cuda" else 0
    return {"models": [{"name": MODEL_ID, "size_vram": mb * 1024 * 1024}]}

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", "11436"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
