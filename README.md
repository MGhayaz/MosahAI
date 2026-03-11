# MosahAI Phase 1.5 Final Refinement

Lightweight scope:
- RSS intake from Google News
- 3-segment batch script generation with Gemini (`gemini-1.5-flash`)
- JSON output storage (`shorts_master.json`)
- SQLite API key health tracking (`usage_tracker.db`)

## Project Files

- `config.py`
- `mosah_ai_engine.py` (`MosahAIBrain`)
- `MultiKey_APIHealth_SQLite_ResilienceTracker.py` (`APIHealthTracker`)
- `key_manager.py` (`KeyManager`)
- `ThreeSegment_DynamicScript_Synthesis_Processor.py` (`ThreeSegmentDynamicScriptSynthesisProcessor`)
- `SemanticSearch_NicheHistory_EvolutionaryVault_Manager.py` (`EvolutionaryVaultManager`)
- `MosahAI_FullPipeline_Execution_Bridge.py` (`MosahAIFullPipelineExecutionBridge`)
- `.env`
- `shorts_master.json`
- `requirements.txt`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r MosahAI/requirements.txt
```

## Environment

Set `MosahAI/.env`:

```env
GEMINI_KEYS=key1,key2,key3,key4,key5
GEMINI_MODEL=gemini-1.5-flash
```

## Run

```bash
python MosahAI_FullPipeline_Execution_Bridge.py
```

## Runtime Flow

1. Load keys from `.env`
2. Sync keys into SQLite `api_keys_health` table
3. Fetch top 6 RSS items per niche (AI, Hyderabad, Geopolitics)
4. Score and select top 3 titles
5. Generate 3 scripts in one Gemini batch request
6. Normalize:
   - each script to ~35-45 words
   - `duration_target: "15-18s"`
   - transition metadata:
     - segment 1 -> cut / trigger true
     - segment 2 -> cut / trigger true
     - segment 3 -> end / trigger false
7. Save JSON record with `status: pending_review`

## SQLite Table

`usage_tracker.db` table: `api_keys_health`

- `id`
- `api_key`
- `usage_count`
- `status` (`Active`, `Cooldown`, `Blocked`)
- `last_used`
- `error_type`

Rules:
- `429` -> cooldown for 24 hours
- repeated `429` can move key to `Blocked`
- `404` -> rotate key and retry with another key
