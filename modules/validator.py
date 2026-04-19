import json
import logging
from typing import Optional, Literal
from datetime import date

import pandas as pd
from pydantic import BaseModel, ValidationError
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


class StructuralError(Exception):
    pass


class AvailabilityRecord(BaseModel):
    property_name: str
    suite: Optional[str] = None
    floor: Optional[int] = None
    size_sf: Optional[float] = None
    headlease_sublease: Optional[Literal["HL", "SL"]] = None
    min_rent: Optional[float] = None   # None if "Market" or blank
    max_rent: Optional[float] = None
    op_cost: Optional[float] = None
    op_cost_year: Optional[int] = None
    availability: Optional[str] = None
    occupancy_status: Optional[str] = None
    future_available: Optional[str] = None
    sublease_expiration: Optional[str] = None
    listing_agency: Optional[str] = None
    lead_agent: Optional[str] = None
    link_to_listing: Optional[str] = None
    date_on_market: Optional[date] = None
    property_comments: Optional[str] = None
    rent_type: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    extraction_note: Optional[str] = None  # LLM flags uncertainty here


def validate_structural(raw_response: str) -> dict:
    """Layer 1: json.loads(). Returns parsed dict or raises StructuralError."""
    # Strip markdown code fences that some LLMs emit
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop opening fence (```json or ```) and closing fence (```)
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[start:end])

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        raise StructuralError(f"JSON parse failed: {exc}") from exc


def validate_schema(parsed: dict) -> list:
    """Layer 2: Pydantic. Returns list of valid AvailabilityRecord. Logs field errors."""
    listings = parsed.get("listings", [])
    if not isinstance(listings, list):
        logger.warning("'listings' key is not a list — got %s", type(listings))
        return []

    valid_records = []
    for i, item in enumerate(listings):
        try:
            record = AvailabilityRecord(**item)
            valid_records.append(record)
        except ValidationError as exc:
            logger.warning("Record %d failed schema validation: %s", i, exc)
        except TypeError as exc:
            logger.warning("Record %d has unexpected type: %s", i, exc)

    return valid_records


def validate_semantic(
    records: list,
    pdf_text: str,
    inventory_df: pd.DataFrame,
) -> tuple:
    """
    Layer 3: Domain rules against Master Inventory.
    Returns: (valid_records, issues_list)
    Issues trigger fallback if len(issues) > 2.
    """
    issues = []

    # Rule 1: at least one record
    if len(records) == 0:
        issues.append({"rule": "min_records", "detail": "No records extracted"})

    # Rule 2: sanity cap
    if len(records) >= 150:
        issues.append(
            {"rule": "max_records", "detail": f"Record count {len(records)} exceeds cap 150"}
        )

    # Build inventory name list for fuzzy matching
    inv_names = _get_inventory_names(inventory_df)
    pdf_lower = pdf_text.lower()

    valid_records = []
    for record in records:
        rec_issues = _check_record(record, pdf_lower, inv_names)
        if rec_issues:
            issues.extend(rec_issues)
            logger.warning(
                "Semantic issues for '%s': %s", record.property_name, rec_issues
            )
        else:
            valid_records.append(record)

    return (valid_records, issues)


# ── private helpers ────────────────────────────────────────────────────────────

def _get_inventory_names(inventory_df: pd.DataFrame) -> list:
    if inventory_df is None or (isinstance(inventory_df, pd.DataFrame) and inventory_df.empty):
        return []
    for col in ("Property Name", "property_name", "name"):
        if col in inventory_df.columns:
            return inventory_df[col].dropna().tolist()
    return []


def _check_record(record: AvailabilityRecord, pdf_lower: str, inv_names: list) -> list:
    issues = []

    # Rule 3: property name present in PDF text
    if record.property_name.lower() not in pdf_lower:
        issues.append({
            "rule": "property_name_in_text",
            "detail": f"'{record.property_name}' not found in PDF text",
            "record": record.property_name,
        })

    # Rule 4: Edmonton rent range
    if record.min_rent is not None and not (5 < record.min_rent < 80):
        issues.append({
            "rule": "rent_range",
            "detail": f"min_rent {record.min_rent} outside Edmonton range (5, 80)",
            "record": record.property_name,
        })

    # Rule 5: size range
    if record.size_sf is not None and not (100 < record.size_sf < 200_000):
        issues.append({
            "rule": "size_range",
            "detail": f"size_sf {record.size_sf} outside range (100, 200000)",
            "record": record.property_name,
        })

    # Rule 6: inventory fuzzy match (very loose — catches hallucinations only)
    if inv_names:
        best_score = max(
            fuzz.token_sort_ratio(record.property_name, name) for name in inv_names
        )
        if best_score <= 55:
            issues.append({
                "rule": "inventory_fuzzy_match",
                "detail": (
                    f"'{record.property_name}' best inventory score {best_score} ≤ 55"
                ),
                "record": record.property_name,
            })

    return issues
