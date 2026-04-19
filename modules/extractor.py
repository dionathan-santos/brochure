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
_MAX_CHARS_PER_PDF = 50_000


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

    pages_text = []
    low_text_count = 0

    for page_num in range(total_pages):
        page = doc[page_num]
        text = _clean_page_text(_extract_page_text(page))
        word_count = len(text.split())
        if word_count < _LOW_WORD_THRESHOLD:
            low_text_count += 1
        pages_text.append(text)

    doc.close()

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


def _extract_page_text(page) -> str:
    """Extract text using spatial block sorting, bypassing the tag/structure tree.

    Interactive PDFs often have a broken structure tree that causes PyMuPDF to
    duplicate content when using the default text extraction mode. Extracting
    blocks and re-sorting by (y, x) coordinates avoids that entirely.
    """
    blocks = page.get_text("blocks")
    # type 0 = text block, type 1 = image block — skip images
    text_blocks = sorted(
        (b for b in blocks if b[6] == 0),
        key=lambda b: (round(b[1], 1), round(b[0], 1)),
    )
    return "\n\n".join(b[4].strip() for b in text_blocks if b[4].strip())


def _clean_page_text(text: str) -> str:
    """Strip single-character noise lines and collapse runs of blank lines."""
    lines = text.splitlines()
    cleaned = ["" if len(line.strip()) <= 1 else line for line in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def _ocr_pdf(pdf_path: str, total_pages: int) -> list:
    """Render each page at 300 DPI and run pytesseract OCR."""
    doc = fitz.open(pdf_path)
    pages_text = []
    scale = _OCR_DPI / 72  # 72 DPI is fitz default
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
