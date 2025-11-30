# backend/app/model_loader.py
import os
import pickle
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger("model-loader")
logger.setLevel(logging.INFO)

# === FIXED ABSOLUTE PATHS (Render checkout path) ===
# These are the exact paths used by Render after it clones your repo.
# Do NOT change these unless you moved the files in the repo.
MODEL_PATH = "/opt/render/project/src/models/url_deep_model.keras"
TOKENIZER_PATH = "/opt/render/project/src/models/tokenizer.pickle"

# Model input settings (must match training)
MAX_LEN = 150
LABELS_MAP = {0: "Malicious", 1: "Safe"}

def _filesize_mb(path):
    try:
        return os.path.getsize(path) / (1024.0 * 1024.0)
    except Exception:
        return None

def load_model_and_tokenizer():
    """
    Load model and tokenizer using explicit repository paths (no env vars).
    Returns (model, tokenizer)
    Raises ValueError with helpful instructions if files are missing.
    """
    # Check tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        msg = (
            f"Tokenizer file not found at hardcoded path:\n  {TOKENIZER_PATH}\n\n"
            "Please ensure tokenizer.pickle is present in the repo at: models/tokenizer.pickle\n"
            "Commit it and push to GitHub so Render can access it. Example repo path:\n"
            "  <repo-root>/models/tokenizer.pickle\n"
        )
        logger.error(msg)
        raise ValueError(msg)

    # Check model
    if not os.path.exists(MODEL_PATH):
        msg = (
            f"Model file not found at hardcoded path:\n  {MODEL_PATH}\n\n"
            "Please ensure url_deep_model.keras is present in the repo at: models/url_deep_model.keras\n"
            "Commit it and push to GitHub so Render can access it. Example repo path:\n"
            "  <repo-root>/models/url_deep_model.keras\n"
        )
        logger.error(msg)
        raise ValueError(msg)

    # Log sizes and warn if large
    model_size = _filesize_mb(MODEL_PATH)
    tok_size = _filesize_mb(TOKENIZER_PATH)
    if model_size is not None:
        logger.info(f"Model size: {model_size:.2f} MB")
        if model_size > 400:
            logger.warning(
                "Model is >400 MB. Render default disk is small (512 MB) and this may cause issues. "
                "Consider using a smaller model or hosting the model externally."
            )
    if tok_size is not None:
        logger.info(f"Tokenizer size: {tok_size:.2f} MB")

    # Load model
    logger.info("Loading model from %s", MODEL_PATH)
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load model")
        raise ValueError(f"Failed to load model at {MODEL_PATH}: {e}")

    # Load tokenizer
    logger.info("Loading tokenizer from %s", TOKENIZER_PATH)
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        logger.exception("Failed to load tokenizer")
        raise ValueError(f"Failed to load tokenizer at {TOKENIZER_PATH}: {e}")

    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def predict_url(model, tokenizer, url: str):
    """
    Predict single URL with loaded model/tokenizer.
    Returns: (label_str, label_index, confidence, features)
    """
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)[0]
    cls = int(pred.argmax())
    conf = float(pred.max())
    label_str = LABELS_MAP.get(cls, str(cls))
    features = {
        "url_length": len(url),
        "count_digits": sum(c.isdigit() for c in url),
        "has_https": 1 if url.lower().startswith("https") else 0
    }
    return label_str, cls, conf, features
