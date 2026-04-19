"""Tests for modules/extractor.py"""
import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy native dependencies so tests run without pymupdf / pytesseract
# installed in the test environment.
# ---------------------------------------------------------------------------
def _stub_fitz():
    fitz = types.ModuleType("fitz")

    class FakePage:
        def __init__(self, text=""):
            self._text = text

        def get_text(self, mode="text"):
            if mode == "blocks":
                # Return a single text block: (x0, y0, x1, y1, text, block_no, type)
                return [(0, 0, 100, 20, self._text, 0, 0)]
            return self._text

        def get_pixmap(self, matrix=None):
            pix = MagicMock()
            pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            return pix

    class FakeDoc:
        def __init__(self, pages):
            self._pages = pages

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, idx):
            return self._pages[idx]

        def close(self):
            pass

    fitz.open = lambda path: FakeDoc([FakePage("word " * 200)])
    fitz.Matrix = lambda sx, sy: None
    sys.modules["fitz"] = fitz
    return fitz


def _stub_pytesseract():
    pt = types.ModuleType("pytesseract")
    pt.image_to_string = lambda img: "OCR extracted text " * 50
    sys.modules["pytesseract"] = pt


def _stub_pil():
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.open = lambda buf: MagicMock()
    pil.Image = image_mod
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = image_mod


_stub_fitz()
_stub_pytesseract()
_stub_pil()

from modules.extractor import extract_text_from_pdf, pages_to_text  # noqa: E402


class TestExtractTextFromPdf(unittest.TestCase):

    def setUp(self):
        """Reset fitz.open to the default rich-text stub before each test."""
        import fitz as fitz_mod
        fitz_mod.open = lambda path: _make_fake_doc(["word " * 200])

    def test_returns_required_keys(self):
        result = extract_text_from_pdf("fake.pdf")
        for key in ("pages", "total_pages", "extraction_method", "quality_flag"):
            self.assertIn(key, result)

    def test_normal_text_pdf_uses_pymupdf(self):
        result = extract_text_from_pdf("fake.pdf")
        self.assertEqual(result["extraction_method"], "pymupdf")

    def test_pages_is_list(self):
        result = extract_text_from_pdf("fake.pdf")
        self.assertIsInstance(result["pages"], list)

    def test_total_pages_matches_pages_length(self):
        result = extract_text_from_pdf("fake.pdf")
        self.assertEqual(result["total_pages"], len(result["pages"]))

    def test_quality_ok_when_all_pages_rich(self):
        result = extract_text_from_pdf("fake.pdf")
        self.assertEqual(result["quality_flag"], "ok")

    def test_ocr_fallback_when_low_text(self):
        """When >30% pages have <80 words, extraction_method should be 'ocr'."""
        import fitz as fitz_mod
        sparse_page_text = "word " * 10  # 10 words < 80 threshold
        fitz_mod.open = lambda path: _make_fake_doc([sparse_page_text] * 5)
        result = extract_text_from_pdf("sparse.pdf")
        self.assertEqual(result["extraction_method"], "ocr")
        self.assertEqual(result["quality_flag"], "ocr_fallback")

    def test_low_text_flag_when_some_pages_sparse(self):
        """When ≤30% pages are sparse, stay with pymupdf but flag low_text."""
        import fitz as fitz_mod
        # Use distinct content so deduplication does not remove the rich blocks
        sparse = "word " * 10
        fitz_mod.open = lambda path: _make_fake_doc([
            "alpha " * 200, "beta " * 200, "gamma " * 200, sparse,
        ])
        result = extract_text_from_pdf("mixed.pdf")
        self.assertEqual(result["extraction_method"], "pymupdf")
        self.assertEqual(result["quality_flag"], "low_text")


class TestPagesToText(unittest.TestCase):

    def test_separator_present(self):
        pages = ["page one content", "page two content"]
        text = pages_to_text(pages)
        self.assertIn("PAGE 1", text)
        self.assertIn("PAGE 2", text)
        self.assertIn("page one content", text)
        self.assertIn("page two content", text)

    def test_empty_pages(self):
        self.assertEqual(pages_to_text([]), "")


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_fake_doc(page_texts):
    import fitz as fitz_mod

    class FakePage:
        def __init__(self, text):
            self._text = text

        def get_text(self, mode="text"):
            if mode == "blocks":
                return [(0, 0, 100, 20, self._text, 0, 0)]
            return self._text

        def get_pixmap(self, matrix=None):
            pix = MagicMock()
            pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            return pix

    class FakeDoc:
        def __init__(self, pages):
            self._pages = pages

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, idx):
            return self._pages[idx]

        def close(self):
            pass

    return FakeDoc([FakePage(t) for t in page_texts])


if __name__ == "__main__":
    unittest.main()
