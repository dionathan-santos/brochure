import re
import logging

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

MATCH_KEY_COLUMNS = ["Property Name", "Suite", "Size"]

COMPARE_COLUMNS = [
    "Min Rent", "Max Rent", "Op. Cost", "Op. Cost Year",
    "Availability", "Occupancy Status", "Future Available (Month or Quarter)",
    "Headlease / Sublease", "Listing Agency", "Lead Agent",
]

_INVENTORY_COLS = [
    "Address", "Submarket", "Class", "Building Size",
    "Construction Status", "Landlord", "Direct/Sub", "PID",
]

_CONFIRMED_THRESHOLD = 85
_REVIEW_THRESHOLD = 75


def run_comparator(extracted_records: list, existing_db_df) -> list:
    """
    Match brochure records against current Availabilities DB.
    Returns all rows tagged with Action, Change_Notes, Match_Score.
    REMOVED rows (unmatched DB entries) are appended at the end.
    """
    import pandas as pd
    from data.name_aliases import NAME_ALIASES  # noqa: F401 (used inside _normalize)

    if existing_db_df is None or (
        isinstance(existing_db_df, pd.DataFrame) and existing_db_df.empty
    ):
        results = []
        for rec in extracted_records:
            row = rec.copy()
            row["Action"] = "NEW"
            row["Change_Notes"] = ""
            row["Match_Score"] = 0
            results.append(row)
        return results

    db_records = existing_db_df.to_dict(orient="records")
    matched_db_indices: set = set()
    # Track best fuzzy score achieved for each DB record (for REMOVED cause inference)
    db_best_scores = [0] * len(db_records)

    results = []
    for brochure_row in extracted_records:
        br_name = _normalize_name(str(brochure_row.get("Property Name", "")))

        best_score = 0
        best_db_idx = None

        for i, db_row in enumerate(db_records):
            db_name = _normalize_name(str(db_row.get("Property Name", "")))
            score = fuzz.token_sort_ratio(br_name, db_name)
            if score > best_score:
                best_score = score
                best_db_idx = i
            if score > db_best_scores[i]:
                db_best_scores[i] = score

        result_row = brochure_row.copy()
        result_row["Match_Score"] = best_score

        if best_score >= _CONFIRMED_THRESHOLD:
            matched_db_indices.add(best_db_idx)
            db_row = db_records[best_db_idx]
            change_notes = build_change_notes(db_row, brochure_row)
            result_row["Change_Notes"] = change_notes
            result_row["Action"] = "UPDATE" if change_notes else "OK"
            _copy_inventory_fields(result_row, db_row)

        elif _REVIEW_THRESHOLD <= best_score < _CONFIRMED_THRESHOLD:
            db_row = db_records[best_db_idx] if best_db_idx is not None else {}
            br_num = _street_number(str(brochure_row.get("Address", "")))
            db_num = _street_number(str(db_row.get("Address", "")))

            if br_num and br_num == db_num:
                # Street number match upgrades to confirmed
                matched_db_indices.add(best_db_idx)
                change_notes = build_change_notes(db_row, brochure_row)
                result_row["Change_Notes"] = change_notes
                result_row["Action"] = "UPDATE" if change_notes else "OK"
                _copy_inventory_fields(result_row, db_row)
            else:
                result_row["Action"] = "REVIEW"
                result_row["Change_Notes"] = ""
        else:
            result_row["Action"] = "NEW"
            result_row["Change_Notes"] = ""

        results.append(result_row)

    # Append REMOVED: DB rows that were never matched
    for i, db_row in enumerate(db_records):
        if i not in matched_db_indices:
            removed_row = db_row.copy()
            removed_row["Action"] = "REMOVED"
            removed_row["Change_Notes"] = ""
            removed_row["Match_Score"] = db_best_scores[i]
            results.append(removed_row)

    return results


def build_change_notes(db_row: dict, brochure_row: dict) -> str:
    """
    Returns string like:
    "Min Rent: Market → $21.00 | Op. Cost: $16.54 → $17.20"
    Returns empty string if no differences.
    """
    changes = []
    for col in COMPARE_COLUMNS:
        db_val = db_row.get(col)
        br_val = brochure_row.get(col)
        db_str = str(db_val).strip() if db_val is not None else ""
        br_str = str(br_val).strip() if br_val is not None else ""

        if db_str == br_str:
            continue
        if not db_str and not br_str:
            continue

        if col in ("Min Rent", "Max Rent", "Op. Cost"):
            db_display = f"${db_str}" if db_str else "Market"
            br_display = f"${br_str}" if br_str else "Market"
        else:
            db_display = db_str or "—"
            br_display = br_str or "—"

        changes.append(f"{col}: {db_display} → {br_display}")

    return " | ".join(changes)


def infer_possible_cause(db_row: dict, brochure_brokerage: str) -> str:
    """
    Infers possible cause for a record being removed.
    """
    causes = []

    days_on_market = db_row.get("Days on Market")
    if days_on_market and int(days_on_market) > 730:
        causes.append("Long-standing listing (2yr+) — possible lease or withdrawal")

    listing_agency = db_row.get("Listing Agency")
    if listing_agency != brochure_brokerage:
        causes.append(
            f"Listed by {listing_agency}, not in {brochure_brokerage} brochure"
            " — check other brokerage"
        )

    match_score = db_row.get("Match_Score")
    if match_score is not None and 60 < int(match_score) < 75:
        causes.append("Low fuzzy score — possible building name alias")

    if not causes:
        causes.append("No obvious cause — verify manually")

    return " | ".join(causes)


# ── private helpers ────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Strip extra whitespace, then apply known aliases (checking both original and title-cased)."""
    from data.name_aliases import NAME_ALIASES

    stripped = " ".join(name.strip().split())
    # Check the original casing first (e.g. "ATCO Place" must hit the alias before title())
    if stripped in NAME_ALIASES:
        return NAME_ALIASES[stripped]
    titled = stripped.title()
    return NAME_ALIASES.get(titled, titled)


def _street_number(address: str) -> str:
    """Extract leading street number, e.g. '10020 101 St' → '10020'."""
    m = re.match(r"^(\d+)", address.strip())
    return m.group(1) if m else ""


def _copy_inventory_fields(target: dict, db_row: dict) -> None:
    """Overwrite FROM_INVENTORY fields in target with authoritative DB values."""
    for col in _INVENTORY_COLS:
        if col in db_row:
            target[col] = db_row[col]
