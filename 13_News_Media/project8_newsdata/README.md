# 8_newsdata — Vietnamese Disaster Information Extraction

This workspace aggregates multiple extraction pipelines to process Vietnamese news into structured disaster intelligence:
- Keyword-based extraction (`keyword_extraction`)
- NER + Relation Extraction (`ner_entity_extraction`, `relation_extraction`)
- LLM-assisted extraction (`llm_extraction`)
- RAG pipeline (`rag_extraction`)

It includes ready-to-run scripts, sample configs, and produces CSV/JSON outputs under `data/`.

## Quick Setup
- Python 3.13 (as per current environment) and `venv`.
- Create/activate virtual environment and install root dependencies:

```bash
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For module-specific extras, install their `requirements.txt` as needed (e.g., `keyword_extraction/requirements.txt`, `relation_extraction/requirements.txt`, `llm_extraction/requirements.txt`).

## Data
Primary dataset files (large):
- `data/disaster_data_multisource_20251207_165113.json`
- `data/disaster_data_multisource_20251207_165113.csv`

Outputs are written to `data/` or each module’s `data/` and then copied to the root `data/` for convenience.

## Run — Keyword Extraction (full dataset)
Processes all articles from the JSON dataset, extracting disaster-relevant sentences and keywords.

```bash
cd keyword_extraction
python run.py                 # runs the full demo (now loads the JSON dataset)
```

Result files:
- `data/keyword_extraction_demo.csv`
- `data/keyword_extraction_demo.json`

These are also copied to the root `data/` for convenience.

## Run — Relation Extraction (demo)
Runs rule-based and LLM RE demos. For full accuracy, supply entities from NER first.

```bash
cd relation_extraction
python run.py                 # runs demo + comparison
```

Result files:
- `relation_extraction/data/re_results_rule.json`
- `relation_extraction/data/re_results_llm.json`
- `relation_extraction/data/re_comparison_summary.json`
- Aggregated CSV: `relation_extraction/data/relation_extraction_results.csv` (created by `scripts/export_to_csv.py`)

Note: LLM runs require API keys; without keys, LLM will be skipped or return empty relations.

## Enable GPT-5.1-Codex-Max (Preview) for All Clients
This project is configured to use the OpenAI preview model `gpt-5.1-codex-max` across LLM clients.

What was changed:
- `llm_extraction/config/llm_config.py`
	- Added config for `gpt-5.1-codex-max` and set it as `DEFAULT_MODEL`.
	- Included the model in OpenAI `supported_models`.
- `relation_extraction/config/re_config.py`
	- Switched `llm_re.model` to `gpt-5.1-codex-max`.

Required environment variable:
- `OPENAI_API_KEY` must be set for all shells running LLM modules.

Set your key on Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY = "<your-key>"
```

Optionally, add it to a `.env` and load via your shell profile.

Verification (non-invasive):
- `cd llm_extraction && python -c "from config.llm_config import get_available_models; print(get_available_models())"` should list `gpt-5.1-codex-max` when the key is present.
- `cd relation_extraction && python run.py --test-loading` should report LLM loaded successfully.

## Troubleshooting
- Missing API key: ensure `OPENAI_API_KEY` is exported in the same shell where you run the scripts.
- Transformers/PhoBERT import errors: install the module’s `requirements.txt` and pin compatible `transformers` versions.
- Large outputs: CSV files may be very large; prefer filtering or chunked processing for Excel.

## Project Structure (abridged)
- `keyword_extraction/` — keyword detection, sentence extraction
- `relation_extraction/` — rule + LLM relation extraction; CSV export utility in `scripts/export_to_csv.py`
- `llm_extraction/` — unified LLM config and prompts
- `rag_extraction/` — RAG components (optional)
- `data/` — inputs and consolidated outputs

## License
Internal project; no external license headers added.
