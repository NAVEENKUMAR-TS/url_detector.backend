import os
import logging
from typing import Optional

logger = logging.getLogger("gemini-client")

try:
    import google.genai as genai
except Exception:
    genai = None

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # default; change if your key supports different model

_client = None

def _get_client():
    global _client
    if _client:
        return _client
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    if genai is None:
        raise RuntimeError("google-genai library not installed")
    _client = genai.Client(api_key=GEMINI_KEY)
    return _client

def explain_with_gemini_or_err(url: str, label: str, confidence: float) -> str:
    """
    Synchronous call to Gemini through google-genai. Returns short textual explanation.
    If an error occurs, raises exception to be handled by caller.
    """
    client = _get_client()
    prompt = (
        f"URL: {url}\n"
        f"Model Prediction: {label}\n"
        f"Confidence: {confidence:.4f}\n\n"
        "Explain in one line why the URL is likely as the prediction and if the URL appears adversarial, "
        "mention the likely adversarial technique and one insight."
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        logger.exception("Gemini call failed")
        raise
