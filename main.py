import os
from datetime import date
import pandas as pd
from config import MASTER_INVENTORY_PATH, AVAILABILITIES_PATH, OUTPUT_DIR, BROCHURE_DIR
from modules.extractor import extract_text_from_pdf
from modules.llm_client import extract_with_fallback
from modules.validator import validate_structural, validate_schema, validate_semantic
from modules.comparator import run_comparator
from modules.output_builder import build_output_excel
from prompts.universal_extraction import USER_PROMPT_TEMPLATE

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
    print(f"Starting pipeline for brokerage: {brokerage}")
    
    # Placeholder: Load dataframes
    # inventory_df = pd.read_excel(MASTER_INVENTORY_PATH)
    # db_df = pd.read_excel(AVAILABILITIES_PATH)
    
    # Placeholder: List PDF files
    # pdf_files = [f for f in os.listdir(brochure_dir) if f.endswith('.pdf')]
    
    all_extracted_records = []
    
    # Loop over PDFs (Placeholder)
    # for pdf_file in pdf_files:
    #     pdf_path = os.path.join(brochure_dir, pdf_file)
    #     ... execution ...
    
    today = date.today()
    output_filename = f"{today}_{brokerage}_BrochureReview.xlsx"
    output_path = os.path.join(OUTPUT_DIR or ".", output_filename)
    
    # Placeholder final step
    # build_output_excel(...)
    
    print(f"Pipeline complete. Summary: ...")
    return output_path

if __name__ == "__main__":
    # Example execution
    if BROCHURE_DIR:
        run_pipeline(BROCHURE_DIR, "GENERAL")
    else:
        print("BROCHURE_DIR not set in environment.")
