import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = int(os.environ.get("MAX_LEN", 150))
LABELS_MAP = {0: "Malicious", 1: "Safe"}

def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """
    Loads TF model and tokenizer. Returns (model, tokenizer).
    Throws on error so startup fails if missing.
    """
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_url(model, tokenizer, url: str):
    """
    Run model prediction for a single URL.
    Returns (label_str, label_index, confidence, features)
    """
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)[0]
    cls = int(np.argmax(pred))
    conf = float(np.max(pred))
    label_str = LABELS_MAP.get(cls, str(cls))
    # Basic features for storage/analysis (add more if your model uses them)
    features = {
        "url_length": len(url),
        "count_digits": sum(c.isdigit() for c in url),
        "has_https": 1 if url.lower().startswith("https") else 0
    }
    return label_str, cls, conf, features
