# backend/app/model_loader.py
import os
import pickle
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger("model-loader")
logger.setLevel(logging.INFO)

MAX_LEN = int(os.environ.get("MAX_LEN", 150))
LABELS_MAP = {0: "Malicious", 1: "Safe"}

def _search_for_file(filename, extra_candidates=None):
    """
    Search for filename in a set of sensible repo-relative locations.
    Returns the first existing absolute path or None.
    """
    # directory where this file lives: backend/app/
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))  # try to reach repo root
    candidates = []

    # Common locations relative to repo root and current dir
    candidates.extend([
        os.path.join(here, filename),                        # backend/app/<file>
        os.path.join(here, "..", filename),                  # backend/<file>
        os.path.join(repo_root, "models", filename),         # <repo>/models/<file>
        os.path.join(repo_root, "model", filename),          # <repo>/model/<file>
        os.path.join(repo_root, "backend", "models", filename), # <repo>/backend/models/<file>
        os.path.join(repo_root, "backend", filename),        # <repo>/backend/<file>
        os.path.join(repo_root, filename),                   # <repo>/<file>
        os.path.join(repo_root, "app", "models", filename),  # <repo>/app/models/<file>
        os.path.join(repo_root, "src", "models", filename),  # <repo>/src/models/<file>
    ])

    # allow user-provided extra candidate paths (absolute or relative)
    if extra_candidates:
        for p in extra_candidates:
            # convert relative to absolute if needed
            if not os.path.isabs(p):
                p = os.path.join(repo_root, p)
            candidates.insert(0, p)

    # normalize and deduplicate while preserving order
    seen = set()
    normalized = []
    for c in candidates:
        c_norm = os.path.abspath(c)
        if c_norm not in seen:
            normalized.append(c_norm)
            seen.add(c_norm)

    # find first that exists
    for c in normalized:
        if os.path.exists(c):
            logger.info(f"Found file at: {c}")
            return c

    # nothing found
    logger.warning("Searched the following paths for '%s' and found none:\n%s",
                   filename, "\n".join(normalized))
    return None

def _filesize_mb(path):
    try:
        return os.path.getsize(path) / (1024.0 * 1024.0)
    except Exception:
        return None

def load_model_and_tokenizer(model_filename="url_deep_model.keras", tokenizer_filename="tokenizer.pickle", extra_model_paths=None, extra_tokenizer_paths=None):
    """
    Locate and load model + tokenizer from repository-relative locations.
    - model_filename: filename of .keras file (default 'url_deep_model.keras')
    - tokenizer_filename: filename of tokenizer pickle (default 'tokenizer.pickle')
    - extra_model_paths / extra_tokenizer_paths: optional lists of extra candidate paths (relative to repo root or absolute)
    Returns (model, tokenizer)
    Raises ValueError with helpful message if not found.
    """

    # find tokenizer first (we need it)
    tok_path = _search_for_file(tokenizer_filename, extra_candidates=extra_tokenizer_paths)
    if not tok_path:
        raise ValueError(
            f"Tokenizer file not found. Expected one of the common locations for '{tokenizer_filename}'.\n"
            "Please place the tokenizer file in one of these locations: \n"
            "  - <repo>/models/{0}\n  - <repo>/backend/models/{0}\n  - <repo>/{0}\n"
            "Or provide an explicit path by editing the code to pass extra_tokenizer_paths.\n".format(tokenizer_filename)
        )

    # find model file next
    model_path = _search_for_file(model_filename, extra_candidates=extra_model_paths)
    if not model_path:
        raise ValueError(
            f"Model file not found. Expected one of the common locations for '{model_filename}'.\n"
            "Please place the model file in one of these locations: \n"
            "  - <repo>/models/{0}\n  - <repo>/backend/models/{0}\n  - <repo>/{0}\n"
            "Or provide an explicit path by editing the code to pass extra_model_paths.\n".format(model_filename)
        )

    # Log file sizes and warn if large (Render default small disk)
    model_size_mb = _filesize_mb(model_path)
    tokenizer_size_mb = _filesize_mb(tok_path)
    if model_size_mb is not None:
        logger.info(f"Model file size: {model_size_mb:.2f} MB")
        if model_size_mb > 400:
            logger.warning(
                "Model larger than 400 MB. On Render with 512 MB disk you may encounter storage issues. "
                "Consider using an external hosted model or a smaller model (TFLite)."
            )
    if tokenizer_size_mb is not None:
        logger.info(f"Tokenizer file size: {tokenizer_size_mb:.2f} MB")

    # Load model
    logger.info("Loading model from %s ...", model_path)
    model = load_model(model_path)
    logger.info("Model loaded.")

    # Load tokenizer
    logger.info("Loading tokenizer from %s ...", tok_path)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    logger.info("Tokenizer loaded.")

    return model, tokenizer

def predict_url(model, tokenizer, url: str):
    """
    Run a prediction for a single URL using the loaded tokenizer/model.
    Returns (label_str, label_index, confidence, features_dict)
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
