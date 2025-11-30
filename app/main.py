import os
import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from .model_loader import load_model_and_tokenizer, predict_url
from .gemini_client import explain_with_gemini_or_err
from .db import init_db, insert_history, get_history

# Read environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "url_deep_model.keras")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "tokenizer.pickle")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "url_detector_db")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", None)
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*")  # comma-separated or '*' for all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("url-detector-backend")

app = FastAPI(title="URL Detector Backend")

# CORS
origins = [o.strip() for o in ALLOW_ORIGINS.split(",")] if ALLOW_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & tokenizer at startup
@app.on_event("startup")
async def startup_event():
    global MODEL, TOKENIZER
    logger.info("Loading model and tokenizer...")
    MODEL, TOKENIZER = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)
    logger.info("Model and tokenizer loaded.")
    # init DB
    await init_db(MONGO_URI, MONGO_DB)
    logger.info("MongoDB initialized.")
    # initialize Gemini client inside gemini_client module (uses env var)
    # gemini client will be lazy-created on first call

# Pydantic request/response schemas
class PredictRequest(BaseModel):
    url: str = Field(..., example="https://example.com/login")
    explain: Optional[bool] = Field(True, description="Whether to call Gemini to obtain an explanation")

class PredictResponse(BaseModel):
    url: str
    label_index: int
    label: str
    confidence: float
    explanation: Optional[str] = None
    features: Optional[dict] = None

class HistoryItem(BaseModel):
    id: str
    url: str
    label_index: int
    label: str
    confidence: float
    explanation: Optional[str]
    timestamp: str

@app.post("/api/predict", response_model=PredictResponse)
async def api_predict(req: PredictRequest, background_tasks: BackgroundTasks):
    # Basic validation
    if not req.url or len(req.url.strip()) == 0:
        raise HTTPException(status_code=400, detail="url required")
    url = req.url.strip()

    # Use your loaded tokenizer/model to predict
    try:
        label_str, label_idx, conf, features = predict_url(MODEL, TOKENIZER, url)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction error: {e}")

    explanation = None
    if req.explain:
        # call Gemini synchronously (may add latency). Alternatively use background_tasks to fetch explanation later.
        try:
            explanation = explain_with_gemini_or_err(url, label_str, conf)
        except Exception as e:
            explanation = f"[Gemini error] {e}"

    # store record in DB asynchronously via background task (fast response to client)
    record = {
        "url": url,
        "label_index": int(label_idx),
        "label": label_str,
        "confidence": float(conf),
        "explanation": explanation,
        "features": features
    }
    background_tasks.add_task(insert_history, record)

    return PredictResponse(
        url=url,
        label_index=int(label_idx),
        label=label_str,
        confidence=float(conf),
        explanation=explanation,
        features=features
    )

@app.get("/api/history", response_model=List[HistoryItem])
async def api_history(limit: int = 50, skip: int = 0):
    """
    Get prediction history (most recent first). Use pagination with skip & limit.
    """
    items = await get_history(limit=limit, skip=skip)
    return items

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
