"""
engine/inference.py
-------------------
OpenRouter-based neural inference for Hemo-Sclera diagnostic pipeline.

Verified available vision models (as of 2026-06):
  - google/gemini-2.5-flash        (primary)
  - google/gemini-2.5-flash-lite   (fallback 1 - faster/cheaper)
  - openai/gpt-4o-mini             (fallback 2 - cross-provider)

Root cause of previous 404:
  'google/gemini-2.0-flash-001' and 'google/gemini-1.5-flash' do NOT exist
  on OpenRouter. Always verify IDs at https://openrouter.ai/api/v1/models.
"""

import os
import json
import base64
import logging
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("hemo_inference")

# ── Load environment ───────────────────────────────────────────────────────────
load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Reads OPENROUTER_API_KEY first; falls back to legacy GEMINI_API_KEY
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")

# ── Model cascade (verified live on OpenRouter 2026-06-24) ─────────────────────
# Primary + fallback list — all confirmed to support image modality.
MODEL_CASCADE = [
    "google/gemini-2.5-flash",       # Primary: best Gemini vision model on OR
    "google/gemini-2.5-flash-lite",  # Fallback 1: faster, cheaper, same vision
    "openai/gpt-4o-mini",            # Fallback 2: cross-provider safety net
]

# ── Startup validation ─────────────────────────────────────────────────────────
def _validate_startup() -> str:
    """
    Validates API key and selects a working model from MODEL_CASCADE.
    Logs configuration details. Raises EnvironmentError if nothing works.
    """
    log.info("=== Hemo-Sclera Neural Engine Startup ===")
    log.info(f"  Base URL : {OPENROUTER_BASE_URL}")
    log.info(f"  API Key  : {'SET (' + API_KEY[:8] + '...)' if API_KEY else 'MISSING'}")
    log.info(f"  Cascade  : {MODEL_CASCADE}")

    if not API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-..."
        )

    # Probe the models endpoint to confirm key validity and find an active model
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            available = {m["id"] for m in json.loads(resp.read())["data"]}
    except Exception as e:
        log.warning(f"Could not fetch model list (will proceed anyway): {e}")
        available = set()

    for model in MODEL_CASCADE:
        if not available or model in available:
            log.info(f"  ✅ Selected model: {model}")
            if available and model not in available:
                log.warning(f"  ⚠️  Model {model} not in verified list — attempting anyway.")
            return model
        else:
            log.warning(f"  ❌ Model {model} not available on OpenRouter, trying next...")

    raise EnvironmentError(
        f"None of the models in MODEL_CASCADE are available on OpenRouter.\n"
        f"Tried: {MODEL_CASCADE}\n"
        f"Check https://openrouter.ai/models for current model IDs."
    )


# Run startup validation once at import time
try:
    ACTIVE_MODEL = _validate_startup()
except EnvironmentError as _startup_err:
    log.error(f"Startup validation FAILED: {_startup_err}")
    ACTIVE_MODEL = MODEL_CASCADE[0]  # Soft-fail: allow app to load, error at inference time

# ── OpenAI client (OpenRouter-compatible) ─────────────────────────────────────
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=API_KEY or "not-set",
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def encode_image(img: Image.Image) -> str:
    """Encode PIL image to base64 JPEG string."""
    buffered = BytesIO()
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _call_model(model: str, base64_image: str, prompt: str, timeout: int) -> dict:
    """Send a single inference request to OpenRouter and return parsed JSON."""
    log.info(f"  → Sending request | model={model} | base_url={OPENROUTER_BASE_URL}")

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        timeout=timeout,
        extra_headers={
            "HTTP-Referer": "https://github.com/udhayan26-hub/Hemo-sclera",
            "X-Title": "Hemo-Sclera Diagnostic",
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    raw_text = response.choices[0].message.content.strip()
    log.info(f"  ← Response received | provider={response.model} | length={len(raw_text)} chars")
    log.debug(f"  RAW: {raw_text}")

    result = json.loads(raw_text)

    # Validate required keys
    required = {"sclera_color_assessment", "risk_level", "confidence_score", "clinical_reasoning"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Model response missing keys: {missing}")

    return result


# ── Main inference function ────────────────────────────────────────────────────
INFERENCE_PROMPT = """You are an expert ophthalmological AI triage system.
Analyze the provided image of a patient's eye to detect scleral icterus (jaundice) or conjunctival pallor (anemia).

CRITICAL INSTRUCTIONS:
1. Focus ONLY on the sclera (white of the eye) and inner eyelid conjunctiva.
2. Completely IGNORE surrounding skin tone, eyelashes, and camera flash glare.
3. Perform a visual heuristic analysis — do NOT attempt colorimetric math.

Return ONLY valid JSON in this EXACT format (no markdown, no extra text):
{
  "sclera_color_assessment": "Describe tissue appearance (e.g. 'Clear white', 'Mild yellowing', 'Severe yellowing', 'Paler than normal')",
  "risk_level": "LOW",
  "confidence_score": 85,
  "clinical_reasoning": "Explain your reasoning and any lighting/skin artifacts considered."
}

Rules:
- risk_level must be exactly one of: LOW, MEDIUM, HIGH
- confidence_score must be an integer 0-100
- Do not include any text outside the JSON object"""


def execute_neural_inference(image_input, timeout: int = 30) -> dict:
    """
    Run multimodal sclera analysis via OpenRouter.

    Args:
        image_input: PIL Image or file path string.
        timeout: Request timeout in seconds.

    Returns:
        dict with keys: sclera_color_assessment, risk_level,
                        confidence_score, clinical_reasoning.
        On failure: also includes 'error' (True) and 'error_detail'.
    """
    if not API_KEY:
        return _error_response("OPENROUTER_API_KEY is not set in .env file.")

    # Load image
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")
        base64_image = encode_image(img)
    except Exception as e:
        return _error_response(f"Image loading failed: {e}")

    # Try primary model, then fallbacks
    models_to_try = [ACTIVE_MODEL] + [m for m in MODEL_CASCADE if m != ACTIVE_MODEL]
    last_error = "Unknown error"

    for model in models_to_try:
        try:
            log.info(f"Attempting inference with model: {model}")
            result = _call_model(model, base64_image, INFERENCE_PROMPT, timeout)
            log.info(f"✅ Inference SUCCESS with model: {model}")
            result["_model_used"] = model  # Attach model info for debugging
            return result

        except APIStatusError as e:
            last_error = str(e)
            status = e.status_code
            log.error(f"❌ APIStatusError {status} with {model}: {last_error}")
            if status == 404:
                log.warning(f"   → Model '{model}' not found on OpenRouter. Trying next...")
                continue  # Try next model in cascade
            elif status in (401, 403):
                return _error_response(
                    f"Invalid or unauthorized API key (HTTP {status}). "
                    "Please check OPENROUTER_API_KEY in your .env file."
                )
            else:
                last_error = f"OpenRouter API error {status}: {last_error}"
                continue

        except APIConnectionError as e:
            last_error = f"Network connection failed: {e}"
            log.error(f"❌ Connection error with {model}: {last_error}")
            break  # Network errors won't be fixed by trying another model

        except APITimeoutError as e:
            last_error = f"Request timed out after {timeout}s: {e}"
            log.error(f"❌ Timeout with {model}: {last_error}")
            continue  # Next model might be faster

        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON from model: {e}"
            log.error(f"❌ JSON decode error with {model}: {last_error}")
            continue

        except Exception as e:
            last_error = str(e)
            log.error(f"❌ Unexpected error with {model}: {last_error}")
            continue

    log.error(f"All models in cascade failed. Last error: {last_error}")
    return _error_response(f"All models failed. Last error: {last_error}")


def _error_response(detail: str) -> dict:
    """Structured error response dict."""
    return {
        "error": True,
        "error_detail": detail,
        "sclera_color_assessment": "Error during inference — please retry.",
        "clinical_reasoning": f"System Alert: {detail}",
        "risk_level": "UNKNOWN",
        "confidence_score": 0,
    }


if __name__ == "__main__":
    log.info("Module loaded successfully.")
    log.info(f"Active model : {ACTIVE_MODEL}")
    log.info(f"Full cascade : {MODEL_CASCADE}")
