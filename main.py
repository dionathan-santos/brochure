import logging
import os
from datetime import date, datetime

import pandas as pd

from config import (
    AVAILABILITIES_PATH,
    MASTER_INVENTORY_PATH,
    OUTPUT_DIR,
    BROCHURE_DIR,
)
from modules.comparator import run_comparator
from modules.extractor import extract_text_from_pdf, pages_to_text
from modules.llm_client import PipelineError, extract_with_fallback
from modules.output_builder import build_output_excel
from modules.validator import (
    StructuralError,
    validate_schema,
    validate_semantic,
    validate_structural,
)
from prompts.universal_extraction import USER_PROMPT_TEMPLATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Mapping from AvailabilityRecord field names → DB column names
_FIELD_TO_DB = {
    "property_name":    "Property Name",
    "floor":            "Floor",
    "suite":            "Suite",
    "size_sf":          "Size",
    "headlease_sublease": "Headlease / Sublease",
    "min_rent":         "Min Rent",
    "max_rent":         "Max Rent",
    "rent_type":        "Rent Type",
    "op_cost":          "Op. Cost",
    "op_cost_year":     "Op. Cost Year",
    "availability":     "Availability",
    "occupancy_status": "Occupancy Status",
    "future_available": "Future Available (Month or Quarter)",
    "sublease_expiration": "Sublease Expiration Date",
    "listing_agency":   "Listing Agency",
    "lead_agent":       "Lead Agent",
    "link_to_listing":  "Link to Listing",
    "date_on_market":   "Date on Market",
    "property_comments": "Property Comments",
    "confidence":       "Extraction_Confidence",
}


def run_pipeline(brochure_dir: str, brokerage: str) -> str:
    """
    1. Load Master Inventory and Availabilities from config paths
    2. Find all PDF files in brochure_dir
    3. For each PDF:
       a. extract_text_from_pdf()
       b. extract_with_fallback() using LLM
       c. validate_structural() → validate_schema() → validate_semantic()
       d. On validation failure > 2 issues: log and skip to next PDF (do not crash)
    4. Merge all valid records into single list
    5. run_comparator() against Availabilities DB
    6. build_output_excel()
    7. Save to OUTPUT_DIR with dated filename
    8. Print summary: total PDFs processed, total listings extracted,
       NEW/UPDATE/OK/REMOVED/REVIEW counts, models used, errors
    Return: path to output Excel file
    """
    logger.info("Pipeline start — brokerage: %s", brokerage)
    today = date.today()

    # ── Load reference data ──────────────────────────────────────────────────
    inventory_df = _load_excel_safe(MASTER_INVENTORY_PATH, "Master Inventory")
    db_df = _load_excel_safe(AVAILABILITIES_PATH, "Availabilities")

    # ── Discover PDF files ───────────────────────────────────────────────────
    pdf_files = sorted(
        f for f in os.listdir(brochure_dir) if f.lower().endswith(".pdf")
    )
    logger.info("Found %d PDF(s) in %s", len(pdf_files), brochure_dir)

    all_records: list = []
    errors: list = []
    models_used: set = set()

    # ── Process each PDF ─────────────────────────────────────────────────────
    for pdf_file in pdf_files:
        pdf_path = os.path.join(brochure_dir, pdf_file)
        logger.info("Processing: %s", pdf_file)

        try:
            # a) Extract text
            extraction = extract_text_from_pdf(pdf_path)
            pdf_text = pages_to_text(extraction["pages"])
            logger.info(
                "  Extracted %d pages via %s (quality: %s)",
                extraction["total_pages"],
                extraction["extraction_method"],
                extraction["quality_flag"],
            )

            # b) LLM extraction
            parsed = extract_with_fallback(USER_PROMPT_TEMPLATE, pdf_text)

            # c) Validation
            structural = validate_structural(
                __import__("json").dumps(parsed)  # re-serialise for layer 1
            )
            schema_records = validate_schema(structural)
            valid_records, issues = validate_semantic(schema_records, pdf_text, inventory_df)

            if len(issues) > 2:
                logger.warning(
                    "  Skipping %s — %d semantic issues: %s",
                    pdf_file, len(issues), issues,
                )
                errors.append({"pdf": pdf_file, "reason": "semantic_issues", "issues": issues})
                continue

            # d) Convert to DB-column dicts and attach pipeline metadata
            for rec in valid_records:
                row = _record_to_db_dict(rec, pdf_file, today)
                all_records.append(row)

            logger.info("  Accepted %d record(s) from %s", len(valid_records), pdf_file)

        except StructuralError as exc:
            logger.error("  Structural error in %s: %s", pdf_file, exc)
            errors.append({"pdf": pdf_file, "reason": "structural_error", "detail": str(exc)})
        except PipelineError as exc:
            logger.error("  LLM pipeline error for %s: %s", pdf_file, exc)
            errors.append({"pdf": pdf_file, "reason": "llm_error", "detail": str(exc)})
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("  Unexpected error for %s: %s", pdf_file, exc, exc_info=True)
            errors.append({"pdf": pdf_file, "reason": "unexpected", "detail": str(exc)})

    logger.info("Total records after all PDFs: %d", len(all_records))

    # ── Comparator ───────────────────────────────────────────────────────────
    diff_results = run_comparator(all_records, db_df)
    removed_records = [r for r in diff_results if r.get("Action") == "REMOVED"]

    # ── Build Excel output ───────────────────────────────────────────────────
    out_dir = OUTPUT_DIR or "."
    os.makedirs(out_dir, exist_ok=True)
    output_filename = f"{today}_{brokerage}_BrochureReview.xlsx"
    output_path = os.path.join(out_dir, output_filename)

    build_output_excel(
        brochure_records=all_records,
        diff_results=diff_results,
        removed_records=removed_records,
        output_path=output_path,
        run_date=today,
        source_brokerage=brokerage,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    action_counts = {}
    for row in diff_results:
        action = row.get("Action", "UNKNOWN")
        action_counts[action] = action_counts.get(action, 0) + 1

    print("\n" + "=" * 60)
    print(f"PIPELINE SUMMARY — {brokerage}  ({today})")
    print("=" * 60)
    print(f"  PDFs processed : {len(pdf_files)}")
    print(f"  Listings found : {len(all_records)}")
    print(f"  NEW            : {action_counts.get('NEW', 0)}")
    print(f"  UPDATE         : {action_counts.get('UPDATE', 0)}")
    print(f"  OK             : {action_counts.get('OK', 0)}")
    print(f"  REMOVED        : {action_counts.get('REMOVED', 0)}")
    print(f"  REVIEW         : {action_counts.get('REVIEW', 0)}")
    print(f"  Errors         : {len(errors)}")
    for err in errors:
        print(f"    ✗ {err['pdf']} — {err['reason']}")
    print(f"  Output         : {output_path}")
    print("=" * 60 + "\n")

    return output_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_excel_safe(path: str | None, label: str) -> pd.DataFrame:
    if not path:
        logger.warning("%s path not set — operating without it", label)
        return pd.DataFrame()
    try:
        df = pd.read_excel(path)
        logger.info("Loaded %s: %d rows", label, len(df))
        return df
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not load %s (%s): %s — continuing without it", label, path, exc)
        return pd.DataFrame()


def _derive_quarter(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"Q{q} {d.year}"


def _record_to_db_dict(record, source_brochure: str, run_date: date) -> dict:
    """Convert an AvailabilityRecord to a flat dict keyed by DB column names."""
    row: dict = {}

    # Map model fields → DB columns
    for field, db_col in _FIELD_TO_DB.items():
        val = getattr(record, field, None)
        if val is not None and hasattr(val, "isoformat"):
            val = val.isoformat()
        row[db_col] = val

    # Pipeline-generated fields
    row["Updated By"] = "PIPELINE"
    row["Last Verified"] = run_date.isoformat()
    row["Days Since Last Verified"] = 0
    row["Quarter"] = _derive_quarter(run_date)
    row["Source_Brochure"] = source_brochure

    # Avg. Asking Rent
    min_r = row.get("Min Rent")
    max_r = row.get("Max Rent")
    try:
        row["Avg. Asking Rent"] = round((float(min_r) + float(max_r)) / 2, 2)
    except (TypeError, ValueError):
        row["Avg. Asking Rent"] = None

    # Days on Market
    dom_str = row.get("Date on Market")
    try:
        dom = date.fromisoformat(str(dom_str))
        row["Days on Market"] = (run_date - dom).days
    except (TypeError, ValueError):
        row["Days on Market"] = None

    return row


if __name__ == "__main__":
    if BROCHURE_DIR:
        run_pipeline(BROCHURE_DIR, "GENERAL")
    else:
        print("BROCHURE_DIR not set in environment.")
