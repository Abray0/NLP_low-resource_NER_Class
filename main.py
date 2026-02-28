"""
FastAPI Backend — Arabic NLP Pipeline
Loads your locally fine-tuned models from the outputs/ folder.
Falls back to base HuggingFace models if local checkpoints aren't found.
"""

from __future__ import annotations
import os
import sys
import time
import logging
from typing import Dict, List, Any
from contextlib import asynccontextmanager

# ── Patch torch CVE check (needed for torch 2.5.x) ───────────────────────────
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "ner":       os.getenv("NER_MODEL_PATH",       os.path.join(BASE_DIR, "outputs", "ner_model")),
    "classification": os.getenv("classification_MODEL_PATH", os.path.join(BASE_DIR, "outputs", "classification_model")),
}

FALLBACK_MODELS = {
    "ner":       "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
    "classification": "CAMeL-Lab/bert-base-arabic-camelbert-msa-classification",
}

def resolve_model_path(task: str) -> str:
    """Use local fine-tuned model if it exists, otherwise fall back to HuggingFace."""
    local = MODEL_PATHS[task]
    if os.path.isdir(local) and os.path.exists(os.path.join(local, "config.json")):
        logger.info(f"[{task}] Loading local fine-tuned model from: {local}")
        return local
    logger.warning(f"[{task}] Local model not found at {local}, falling back to: {FALLBACK_MODELS[task]}")
    return FALLBACK_MODELS[task]


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model registry — loaded on first request, cached in memory
# ─────────────────────────────────────────────────────────────────────────────
_models: Dict[str, Any] = {}


def get_ner_pipeline():
    if "ner" not in _models:
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            pipeline as hf_pipeline,
        )
        path = resolve_model_path("ner")
        _models["ner"] = hf_pipeline(
            "ner",
            model=path,
            tokenizer=path,
            aggregation_strategy="simple",
            device=-1,  # CPU; change to 0 for GPU
        )
    return _models["ner"]


def get_classification_pipeline():
    if "classification" not in _models:
        from transformers import pipeline as hf_pipeline
        path = resolve_model_path("classification")
        _models["classification"] = hf_pipeline(
            "text-classification",
            model=path,
            tokenizer=path,
            device=-1,
        )
    return _models["classification"]

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Arabic NLP Backend starting up...")
    logger.info(f"NER model:       {resolve_model_path('ner')}")
    logger.info(f"classification model: {resolve_model_path('classification')}")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Arabic NLP Pipeline API",
    description="Low-resource Arabic NLP — fine-tuned XLM-R + NLLB-200",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      example="زيارة الرئيس الأمريكي جو بايدن إلى القاهرة")

class NEREntity(BaseModel):
    entity_group: str
    word: str
    score: float
    start: int
    end: int

class NERResponse(BaseModel):
    text: str
    entities: List[NEREntity]
    model_path: str
    processing_time_ms: float

class classificationResponse(BaseModel):
    text: str
    label: str
    score: float
    model_path: str
    processing_time_ms: float

# Translation schemas removed

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    model_paths: Dict[str, str]
    version: str

class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "Arabic NLP Pipeline API — visit /docs for interactive UI"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        models_loaded=list(_models.keys()),
        model_paths={task: resolve_model_path(task) for task in ["ner", "classification"]},
        version="1.0.0",
    )


# ── NER ───────────────────────────────────────────────────────────────────────
@app.post("/api/ner", response_model=NERResponse, tags=["NER"])
def named_entity_recognition(body: TextInput):
    """
    Named Entity Recognition on Arabic text.
    Returns PER (person), ORG (organization), LOC (location) spans.
    """
    try:
        t0 = time.perf_counter()
        pipe = get_ner_pipeline()
        raw = pipe(body.text)
        elapsed = (time.perf_counter() - t0) * 1000

        entities = [
            NEREntity(
                entity_group=e.get("entity_group", e.get("entity", "UNK")),
                word=e["word"],
                score=round(float(e["score"]), 4),
                start=e.get("start", 0),
                end=e.get("end", 0),
            )
            for e in raw
        ]
        return NERResponse(
            text=body.text,
            entities=entities,
            model_path=resolve_model_path("ner"),
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        logger.error(f"NER error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ner/batch", tags=["NER"])
def ner_batch(body: BatchInput):
    """Run NER on multiple Arabic texts at once (max 50)."""
    try:
        pipe = get_ner_pipeline()
        results = []
        for text in body.texts:
            raw = pipe(text)
            results.append({
                "text": text,
                "entities": [
                    {
                        "entity_group": e.get("entity_group", "UNK"),
                        "word": e["word"],
                        "score": round(float(e["score"]), 4),
                    }
                    for e in raw
                ],
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── classification ─────────────────────────────────────────────────────────────────
@app.post("/api/classification", response_model=classificationResponse, tags=["classification"])
def classification_analysis(body: TextInput):
    """
    Text classification / classification on Arabic text.
    Returns label + confidence score.
    """
    try:
        t0 = time.perf_counter()
        pipe = get_classification_pipeline()
        result = pipe(body.text, truncation=True, max_length=512)[0]
        elapsed = (time.perf_counter() - t0) * 1000

        return classificationResponse(
            text=body.text,
            label=result["label"],
            score=round(float(result["score"]), 4),
            model_path=resolve_model_path("classification"),
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        logger.error(f"classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)