SYSTEM_PROMPT = """
You are a structured data extraction assistant for commercial real estate.
You extract office space availability data from brokerage brochures.
You return ONLY valid JSON. No preamble. No explanation. No markdown. No backticks.
If you are uncertain about a value, set it to null and set confidence to "low".
Never invent values. If a field is not present in the document, return null.
"""

USER_PROMPT_TEMPLATE = """
STEP 1 - ANALYZE LAYOUT:
Before extracting, identify: What brokerage produced this brochure?
What terminology do they use for: base rent, operating costs, suite number, floor number?
Note this in extraction_meta.

STEP 2 - EXTRACT ALL LISTINGS:
For each available office space found in the document, extract one record.
Extract by SUITE when suite numbers are available.
Extract by FLOOR only when no suite numbers exist (use "Floor X" as suite value).

STEP 3 - RETURN JSON:
{{
  "extraction_meta": {{
    "brokerage": "string",
    "total_listings_found": int,
    "base_rent_label_used": "string",
    "op_cost_label_used": "string"
  }},
  "listings": [
    {{
      "property_name": "string",
      "suite": "string or null",
      "floor": int or null,
      "size_sf": float or null,
      "headlease_sublease": "HL" or "SL" or null,
      "min_rent": float or null,
      "max_rent": float or null,
      "rent_type": "Net" or "Gross" or null,
      "op_cost": float or null,
      "op_cost_year": int or null,
      "availability": "Immediately" or "Future" or null,
      "occupancy_status": "Vacant" or "Occupied" or null,
      "future_available": "string or null",
      "sublease_expiration": "string or null",
      "listing_agency": "string or null",
      "lead_agent": "string or null",
      "link_to_listing": "string or null",
      "date_on_market": "YYYY-MM-DD or null",
      "property_comments": "string or null",
      "confidence": "high" or "medium" or "low",
      "extraction_note": "string or null"
    }}
  ]
}}

RULES:
- min_rent and max_rent: extract as float (e.g. 18.50). If listed as "Market" or "TBD" or "Negotiable": return null.
- op_cost: extract as float. If unavailable: null.
- size_sf: always as float in square feet.
- future_available: preserve exact text from brochure (e.g. "Q3 2026", "September 2026", "6 months notice").
- confidence: "high" if all key fields present, "medium" if 1-2 fields missing, "low" if core fields (size or property_name) are inferred.
- extraction_note: use this to flag anything unusual (e.g. "rent not listed, inferred from comparable suite").

BROCHURE TEXT:
{pdf_text}
"""
