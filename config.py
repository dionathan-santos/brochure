import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Paths
MASTER_INVENTORY_PATH = os.getenv("MASTER_INVENTORY_PATH")
AVAILABILITIES_PATH = os.getenv("AVAILABILITIES_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
BROCHURE_DIR = os.getenv("BROCHURE_DIR")

# Data Schema Definition
DB_COLUMNS = [
    "Property Name",           # col 1
    "Address",                 # col 2
    "Submarket",               # col 3
    "Class",                   # col 4
    "Building Size",           # col 5
    "Construction Status",     # col 6
    "Days Since Last Verified",# col 7
    "Last Verified",           # col 8
    "Updated By",              # col 9
    "Headlease / Sublease",    # col 10
    "Sale",                    # col 11
    "Floor",                   # col 12
    "Suite",                   # col 13
    "Size",                    # col 14
    "Max Contiguous",          # col 15
    "Full Floor",              # col 16
    "Min Rent",                # col 17
    "Max Rent",                # col 18
    "Avg. Asking Rent",        # col 19
    "Rent Type",               # col 20
    "Rent Frequency",          # col 21
    "Sale Price",              # col 22
    "Op. Cost",                # col 23
    "Op. Cost Year",           # col 24
    "TIA",                     # col 25
    "Availability",            # col 26
    "Occupancy Status",        # col 27
    "Future Available (Month or Quarter)", # col 28
    "Sublease Expiration Date",# col 29
    "Property Comments",       # col 30
    "Internal Comments",       # col 31
    "In Office Stats (Yes / No)", # col 32
    "Reason Not in Office Stats", # col 33
    "Landlord",                # col 34
    "Listing Agency",          # col 35
    "Lead Agent",              # col 36
    "AVANT Link",              # col 37
    "Link to Listing",         # col 38
    "Date on Market",          # col 39
    "Days on Market",          # col 40
    "Direct/Sub",              # col 41
    "Quarter",                 # col 42
    "PID",                     # col 43
    "Source_Brochure",         # col 44
    "Extraction_Confidence",   # col 45
    "Match_Score",             # col 46
]

COLUMN_SOURCES = {
    "FROM_BROCHURE": [
        "Property Name", "Floor", "Suite", "Size", "Max Contiguous",
        "Full Floor", "Headlease / Sublease", "Sale", "Min Rent", "Max Rent",
        "Rent Type", "Rent Frequency", "Sale Price", "Op. Cost", "Op. Cost Year",
        "TIA", "Availability", "Occupancy Status",
        "Future Available (Month or Quarter)", "Sublease Expiration Date",
        "Property Comments", "Listing Agency", "Lead Agent", "Link to Listing",
        "Date on Market",
    ],
    "FROM_INVENTORY": [
        "Address", "Submarket", "Class", "Building Size",
        "Construction Status", "Landlord", "Direct/Sub", "PID",
    ],
    "FROM_PIPELINE": [
        "Days Since Last Verified", "Last Verified", "Updated By",
        "Avg. Asking Rent", "Days on Market", "Quarter",
        "Source_Brochure", "Extraction_Confidence", "Match_Score",
    ],
    "INTENTIONALLY_BLANK": [
        "Internal Comments", "In Office Stats (Yes / No)",
        "Reason Not in Office Stats", "AVANT Link",
    ],
}
