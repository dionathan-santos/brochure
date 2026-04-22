import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 32000
TEMPERATURE = 0.0
TIMEOUT = int(os.environ.get("LLM_TIMEOUT_SECONDS", "90"))  # seconds per call


class LLMError(Exception):
    pass


class PipelineError(Exception):
    pass


def call_llm(
    prompt: str,
    pdf_text: str,
    model: str = "gemini-flash-lite",
    attempt: int = 1,
) -> str:
    """
    Returns raw string response from LLM.
    Raises: LLMError on HTTP error or timeout.
    """
    if model == HAIKU_MODEL or model.startswith("claude"):
        return _call_haiku(prompt, pdf_text, attempt)
    return _call_gemini(prompt, pdf_text, attempt)


def extract_with_fallback(prompt: str, pdf_text: str) -> dict:
    """
    Orchestrates: Gemini attempt 1 → Gemini attempt 2 → Haiku attempt.
    Returns parsed JSON dict with _pipeline_meta or raises PipelineError.
    """
    attempts = [
        (GEMINI_MODEL, 1),
        (GEMINI_MODEL, 2),
        (HAIKU_MODEL, 1),
    ]

    last_exc = None
    for model, attempt in attempts:
        try:
            raw = call_llm(prompt, pdf_text, model=model, attempt=attempt)
            result = _parse_json(raw)
            result.setdefault("_pipeline_meta", {})
            result["_pipeline_meta"].update({
                "model_used": model,
                "attempt": attempt,
            })
            logger.info("Success: %s attempt %d", model, attempt)
            return result
        except (LLMError, ValueError) as exc:
            last_exc = exc
            logger.warning("%s attempt %d failed: %s", model, attempt, exc)
            _sleep_before_retry(model=model, attempt=attempt)

    raise PipelineError("All LLM attempts (Gemini×2 + Haiku) failed") from last_exc


def _sleep_before_retry(model: str, attempt: int) -> None:
    """Simple exponential backoff with jitterless deterministic delays."""
    # No wait after final fallback attempt
    if model == HAIKU_MODEL and attempt == 1:
        return

    base = float(os.environ.get("LLM_RETRY_BASE_SECONDS", "1.5"))
    delay = min(base * (2 ** (attempt - 1)), 8.0)
    logger.info("Retry backoff: sleeping %.1fs", delay)
    time.sleep(delay)


# ── private helpers ────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Strip optional markdown fences and parse JSON. Raises ValueError on failure."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode error: {exc}") from exc


def _call_gemini(prompt: str, pdf_text: str, attempt: int) -> str:
    """Call Gemini via the REST API directly (bypasses SDK/proxy issues in Colab)."""
    from prompts.universal_extraction import SYSTEM_PROMPT

    # Read at call-time so Colab env vars set after module import are picked up
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise LLMError("GEMINI_API_KEY not set")

    # Allow notebook-level model override via GEMINI_MODEL_OVERRIDE env var
    model_name = os.environ.get("GEMINI_MODEL_OVERRIDE") or GEMINI_MODEL

    full_prompt = prompt.format(pdf_text=pdf_text)
    start = time.time()

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={gemini_api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"content-type": "application/json"},
            timeout=TIMEOUT,
        )
        if not resp.ok:
            body = resp.text[:300]
            duration = time.time() - start
            raise LLMError(
                f"Gemini HTTP {resp.status_code} (attempt {attempt}, {duration:.1f}s): {body}"
            )
        duration = time.time() - start
        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Gemini response missing text: {data}") from exc
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", len(full_prompt) // 4)
        logger.info(
            "Gemini attempt %d (%s): %s input tokens, %.1fs",
            attempt, model_name, input_tokens, duration,
        )
        return text
    except LLMError:
        raise
    except requests.exceptions.RequestException as exc:
        duration = time.time() - start
        raise LLMError(
            f"Gemini request error (attempt {attempt}, {duration:.1f}s): {exc}"
        ) from exc


def _call_haiku(prompt: str, pdf_text: str, attempt: int) -> str:
    """Call Anthropic Claude Haiku via the Messages API."""
    from prompts.universal_extraction import SYSTEM_PROMPT

    # Read at call-time for same reason as Gemini key above
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY not set")

    full_prompt = prompt.format(pdf_text=pdf_text)
    start = time.time()

    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": HAIKU_MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": full_prompt}],
    }

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
        if not resp.ok:
            body = resp.text[:300]
            duration = time.time() - start
            raise LLMError(
                f"Haiku HTTP {resp.status_code} (attempt {attempt}, {duration:.1f}s): {body}"
            )
        duration = time.time() - start
        data = resp.json()
        input_tokens = data.get("usage", {}).get("input_tokens", "?")
        logger.info(
            "Haiku attempt %d: %s input tokens, %.1fs", attempt, input_tokens, duration
        )
        return data["content"][0]["text"]
    except LLMError:
        raise
    except requests.exceptions.RequestException as exc:
        duration = time.time() - start
        raise LLMError(
            f"Haiku request error (attempt {attempt}, {duration:.1f}s): {exc}"
        ) from exc
