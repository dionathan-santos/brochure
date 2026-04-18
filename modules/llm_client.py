import json
import logging
import time

import requests

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash-lite"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 4000
TEMPERATURE = 0.0
TIMEOUT = 60  # seconds per call


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
    Returns parsed JSON dict or raises PipelineError.
    Logs which model succeeded and attempt number.
    """
    # Attempt 1: Gemini
    try:
        raw = call_llm(prompt, pdf_text, model=GEMINI_MODEL, attempt=1)
        result = _parse_json(raw)
        logger.info("Success: Gemini attempt 1")
        return result
    except (LLMError, ValueError) as exc:
        logger.warning("Gemini attempt 1 failed: %s", exc)

    # Attempt 2: Gemini retry
    try:
        raw = call_llm(prompt, pdf_text, model=GEMINI_MODEL, attempt=2)
        result = _parse_json(raw)
        logger.info("Success: Gemini attempt 2")
        return result
    except (LLMError, ValueError) as exc:
        logger.warning("Gemini attempt 2 failed: %s", exc)

    # Attempt 3: Haiku fallback
    try:
        raw = call_llm(prompt, pdf_text, model=HAIKU_MODEL, attempt=1)
        result = _parse_json(raw)
        logger.info("Success: Haiku attempt 1")
        return result
    except (LLMError, ValueError) as exc:
        logger.warning("Haiku attempt failed: %s", exc)
        raise PipelineError("All LLM attempts (Gemini×2 + Haiku) failed") from exc


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
    """Call Gemini via the google-generativeai SDK."""
    # Import lazily so the module still loads without the package installed
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise LLMError("google-generativeai not installed") from exc

    from config import GEMINI_API_KEY
    from prompts.universal_extraction import SYSTEM_PROMPT

    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY not set")

    start = time.time()
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_obj = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        generation_config = genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        full_prompt = prompt.format(pdf_text=pdf_text)
        response = model_obj.generate_content(
            full_prompt,
            generation_config=generation_config,
            request_options={"timeout": TIMEOUT},
        )
        duration = time.time() - start
        # Rough token estimate: 1 token ≈ 4 chars
        est_tokens = len(full_prompt) // 4
        logger.info(
            "Gemini attempt %d: ~%d input tokens, %.1fs", attempt, est_tokens, duration
        )
        return response.text
    except Exception as exc:
        duration = time.time() - start
        raise LLMError(f"Gemini error (attempt {attempt}, {duration:.1f}s): {exc}") from exc


def _call_haiku(prompt: str, pdf_text: str, attempt: int) -> str:
    """Call Anthropic Claude Haiku via the Messages API."""
    from config import ANTHROPIC_API_KEY
    from prompts.universal_extraction import SYSTEM_PROMPT

    if not ANTHROPIC_API_KEY:
        raise LLMError("ANTHROPIC_API_KEY not set")

    full_prompt = prompt.format(pdf_text=pdf_text)
    start = time.time()

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
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
        resp.raise_for_status()
        duration = time.time() - start
        data = resp.json()
        input_tokens = data.get("usage", {}).get("input_tokens", "?")
        logger.info(
            "Haiku attempt %d: %s input tokens, %.1fs", attempt, input_tokens, duration
        )
        return data["content"][0]["text"]
    except requests.exceptions.RequestException as exc:
        duration = time.time() - start
        raise LLMError(
            f"Haiku HTTP error (attempt {attempt}, {duration:.1f}s): {exc}"
        ) from exc
