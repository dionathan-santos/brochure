def call_llm(
    prompt: str,
    pdf_text: str,
    model: str = "gemini-flash-lite",
    attempt: int = 1
) -> str:
    """
    Returns raw string response from LLM.
    Raises: LLMError on HTTP error or timeout.
    """
    # Placeholder implementation
    return ""

def extract_with_fallback(prompt: str, pdf_text: str) -> dict:
    """
    Orchestrates: Gemini attempt 1 → Gemini attempt 2 → Haiku attempt.
    Returns parsed JSON dict or raises PipelineError.
    Logs which model succeeded and attempt number.
    """
    # Placeholder implementation
    return {}
