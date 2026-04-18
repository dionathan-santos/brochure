import io
import logging

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

_LOW_WORD_THRESHOLD = 80
_OCR_TRIGGER_RATIO = 0.30
_OCR_DPI = 300
_PAGE_SEPARATOR = "\n--- PAGE {n} ---\n"


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
        text = page.get_text()
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
    """Join per-page strings with the standard separator for LLM consumption."""
    parts = []
    for n, page_text in enumerate(pages, start=1):
        parts.append(_PAGE_SEPARATOR.format(n=n))
        parts.append(page_text)
    return "".join(parts)
