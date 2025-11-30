import os
import pickle
import requests
import shutil
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger("model-loader")
MAX_LEN = int(os.environ.get("MAX_LEN", 150))
LABELS_MAP = {0: "Malicious", 1: "Safe"}

def _download_file(url, dest_path, chunk_size=8192):
    logger.info(f"Downloading model from {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = 0
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    logger.info(f"Downloaded {total} bytes to {dest_path}")
    return dest_path

def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """
    Load model and tokenizer. Behavior:
      - If model_path exists on disk -> load it
      - Else if env MODEL_URL is set -> download to /tmp and load
      - Else: raise ValueError with clear message
    """
    # resolve tokenizer
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer file not found at: {tokenizer_path}. Upload file or set TOKENIZER_PATH env var.")

    # Resolve model path or download
    if os.path.exists(model_path):
        final_model_path = model_path
        logger.info(f"Found model at {model_path}")
    else:
        model_url = os.environ.get("MODEL_URL")
        if not model_url:
            raise ValueError(
                f"Model file not found at {model_path} and MODEL_URL env var is not set. "
                "Either upload the model to the repo and set MODEL_PATH correctly or set MODEL_URL to a downloadable link."
            )
        # download into /tmp
        tmp_path = "/tmp/url_deep_model.keras"
        # Remove any stale file first
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        _download_file(model_url, tmp_path)
        final_model_path = tmp_path

    # Optional storage check: warn if file likely too big for available disk (512MB)
    try:
        size_bytes = os.path.getsize(final_model_path)
        size_mb = size_bytes / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.2f} MB")
        # if deploy environment limited to 512 MB, warn early
        if size_mb > 400:
            logger.warning("Model is large (>400MB). On Render with 512MB disk this may cause issues. Consider making a smaller model or using remote inference.")
    except Exception:
        logger.exception("Could not determine model size.")

    # load model and tokenizer
    model = load_model(final_model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_url(model, tokenizer, url: str):
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
