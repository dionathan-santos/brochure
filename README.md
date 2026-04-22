# Edmonton Office Brochure Extraction Pipeline

## Overview
Build a Python pipeline that reads PDF brochures from Edmonton office brokerages, extracts structured availability data using an LLM API, validates it against a Master Inventory, and produces a 3-tab Excel output for review.

## Repository Structure
- `config.py`: Environment configuration and schema constants.
- `main.py`: Main entry point for the extraction pipeline.
- `modules/`: Contains logic for extraction, LLM client, validation, comparison, and output building.
- `prompts/`: LLM prompt templates.
- `data/`: Data constants and aliases.
- `notebooks/`: Google Colab entry point.
- `tests/`: Unit tests for various modules.

## Setup Instructions

### Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys and paths:
   ```bash
   cp .env.example .env
   ```
4. Run the pipeline:
   ```bash
   python main.py
   ```

### Google Colab
Follow instructions in `notebooks/run_pipeline.ipynb` to run the pipeline in a Colab environment with Google Drive integration.


## Recent Reliability Improvements
- Added deterministic LLM retry backoff and model/attempt metadata capture.
- Added severity-based semantic issue handling in the pipeline (while still keeping partial valid records per PDF).
- Added provenance columns in output records (`Model_Used`, `Extraction_Attempt`, `Source_Page_Range`, `Validation_Issue_Count`).
- Improved comparator matching to use a weighted composite of name + suite + size similarity.

## Running Tests
Run from repo root:
```bash
pytest -q
```

Optional reliability envs:
- `LLM_RETRY_BASE_SECONDS` (default: `1.5`)
- `LLM_TIMEOUT_SECONDS` (default: `90`)

## License
Proprietary - Avison Young Market Intelligence
