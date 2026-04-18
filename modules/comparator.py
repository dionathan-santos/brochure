MATCH_KEY_COLUMNS = ["Property Name", "Suite", "Size"]

COMPARE_COLUMNS = [
    "Min Rent", "Max Rent", "Op. Cost", "Op. Cost Year",
    "Availability", "Occupancy Status", "Future Available (Month or Quarter)",
    "Headlease / Sublease", "Listing Agency", "Lead Agent",
]

def run_comparator(extracted_records: list[dict], existing_db_df: any) -> list[dict]:
    """
    Match brochure records against current Availabilities DB.
    Placeholder return.
    """
    return []

def build_change_notes(db_row: dict, brochure_row: dict) -> str:
    """
    Returns string like:
    "Min Rent: Market → $21.00 | Op. Cost: $16.54 → $17.20"
    Returns empty string if no differences.
    """
    # Placeholder implementation
    return ""

def infer_possible_cause(db_row: dict, brochure_brokerage: str) -> str:
    """
    Infers possible cause for a record being removed.
    """
    # Placeholder implementation
    return ""
