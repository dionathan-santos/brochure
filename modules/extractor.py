import io
import logging
import re

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

_LOW_WORD_THRESHOLD = 80
_OCR_TRIGGER_RATIO = 0.30
_OCR_DPI = 300
_PAGE_SEPARATOR = "\n--- PAGE {n} ---\n"
_MAX_CHARS_PER_PDF = 30_000


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Returns: {
        "pages": [str, str, ...],   # text per page
        "total_pages": int,
        "extraction_method": "pymupdf" | "ocr",
        "quality_flag": "ok" | "low_text" | "ocr_fallback"
    }
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if total_pages == 0:
        doc.close()
        return {
            "pages": [],
            "total_pages": 0,
            "extraction_method": "pymupdf",
            "quality_flag": "ok",
        }

    # Phase 1: extract raw blocks per page
    all_page_blocks = []
    for page_num in range(total_pages):
        all_page_blocks.append(_get_text_blocks(doc[page_num]))
    doc.close()

    # Phase 2: remove blocks repeated on >50% of pages (navigation / UI elements)
    all_page_blocks = _deduplicate_blocks(all_page_blocks)

    # Phase 3: reconstruct page text and measure quality
    pages_text = []
    low_text_count = 0
    for blocks in all_page_blocks:
        text = _clean_page_text("\n\n".join(blocks))
        if len(text.split()) < _LOW_WORD_THRESHOLD:
            low_text_count += 1
        pages_text.append(text)

    low_text_ratio = low_text_count / total_pages

    if low_text_ratio > _OCR_TRIGGER_RATIO:
        logger.info(
            "%.0f%% of pages below word threshold — switching to OCR for %s",
            low_text_ratio * 100,
            pdf_path,
        )
        pages_text = _ocr_pdf(pdf_path, total_pages)
        return {
            "pages": pages_text,
            "total_pages": total_pages,
            "extraction_method": "ocr",
            "quality_flag": "ocr_fallback",
        }

    quality_flag = "low_text" if low_text_count > 0 else "ok"
    return {
        "pages": pages_text,
        "total_pages": total_pages,
        "extraction_method": "pymupdf",
        "quality_flag": quality_flag,
    }


def _get_text_blocks(page) -> list:
    """Return spatially-sorted, non-empty text block strings from a page."""
    raw_blocks = page.get_text("blocks")
    sorted_blocks = sorted(
        (b for b in raw_blocks if b[6] == 0),        # skip image blocks
        key=lambda b: (round(b[1], 1), round(b[0], 1)),
    )
    return [b[4].strip() for b in sorted_blocks if b[4].strip()]


def _deduplicate_blocks(all_page_blocks: list) -> list:
    """Remove text blocks that appear on more than half the pages.

    Interactive PDFs embed navigation menus, headers, and button labels that
    repeat on every page. These inflate the text sent to the LLM without
    adding extraction value.
    """
    n_pages = len(all_page_blocks)
    if n_pages < 3:
        return all_page_blocks

    # Count pages each normalised block text appears on
    page_count: dict = {}
    for page_blocks in all_page_blocks:
        seen: set = set()
        for block in page_blocks:
            key = " ".join(block.split())
            if key and key not in seen:
                page_count[key] = page_count.get(key, 0) + 1
                seen.add(key)

    threshold = n_pages * 0.5
    repeating = {k for k, v in page_count.items() if v > threshold}

    if repeating:
        logger.debug("Removed %d repeated block(s) (>50%% of pages)", len(repeating))

    return [
        [b for b in page_blocks if " ".join(b.split()) not in repeating]
        for page_blocks in all_page_blocks
    ]


def _clean_page_text(text: str) -> str:
    """Strip single-character noise lines and collapse runs of blank lines."""
    lines = text.splitlines()
    cleaned = ["" if len(line.strip()) <= 1 else line for line in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def _ocr_pdf(pdf_path: str, total_pages: int) -> list:
    """Render each page at 300 DPI and run pytesseract OCR."""
    doc = fitz.open(pdf_path)
    pages_text = []
    scale = _OCR_DPI / 72
    mat = fitz.Matrix(scale, scale)

    for page_num in range(total_pages):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        pages_text.append(text)
        logger.debug("OCR page %d/%d", page_num + 1, total_pages)

    doc.close()
    return pages_text


def pages_to_text(pages: list) -> str:
    """Join per-page strings with the standard separator.

    Stops adding pages once the total reaches _MAX_CHARS_PER_PDF so the LLM
    receives a bounded, predictable input size regardless of PDF length.
    """
    parts = []
    total_chars = 0
    for n, page_text in enumerate(pages, start=1):
        block = _PAGE_SEPARATOR.format(n=n) + page_text
        if total_chars + len(block) > _MAX_CHARS_PER_PDF:
            remaining = len(pages) - n + 1
            parts.append(
                f"\n[... {remaining} page(s) omitted — character limit reached]"
            )
            break
        parts.append(block)
        total_chars += len(block)
    return "".join(parts)
