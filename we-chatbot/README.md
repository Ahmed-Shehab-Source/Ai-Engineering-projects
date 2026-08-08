# Aether — WE Business Assistant

A RAG chatbot for WE (Telecom Egypt) business services, built with Streamlit,
Qdrant, and OpenRouter (free-tier LLMs with automatic fallback).

## Pipeline

This is the actual pipeline used to build and run the chatbot, in order:

1. **`sc2_chunk_data.py`** — reads markdown files from a `chatbot_demo_data/`
   folder (not included — bring your own source docs) and splits them into
   overlapping chunks, saving them to `chunks.json`.
2. **`emb_sc_upload.py`** — embeds every chunk in `chunks.json` (via
   OpenRouter's `openai/text-embedding-3-small`) and uploads them into a
   Qdrant Cloud collection called `we_chatbot`.
3. **`retriever.py`** — reusable retrieval helper: given a query, returns the
   top-k most relevant chunks from Qdrant with similarity scores. Also
   runnable standalone for a quick manual test.
4. **`sc5_query_test.py`** — a command-line RAG test script: retrieves
   context via `retriever.py` and asks an OpenRouter LLM to answer using
   only that context. Useful for testing retrieval + prompt before touching
   the UI.
5. **`app.py`** — the Streamlit chat UI (Aether). This is the main app you
   run day to day; it has its own self-contained retrieval + multi-model
   fallback logic.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Every script has placeholder API keys near the top — replace them with
   your own before running:
   ```python
   OPENROUTER_API_KEY = "openrouter_api_key_here"
   QDRANT_URL = "qdrant_url_here"
   QDRANT_API_KEY = "qdrant_api_key_here"
   ```
   - Get an OpenRouter key: https://openrouter.ai/keys
   - Get a Qdrant Cloud cluster + key: https://cloud.qdrant.io

3. To build your knowledge base from scratch:
   ```bash
   # 1. Put your source .md files in a chatbot_demo_data/ folder
   python3 sc2_chunk_data.py        # -> produces chunks.json

   # 2. Fill in your keys in emb_sc_upload.py, then:
   python3 emb_sc_upload.py         # embeds + uploads chunks.json to Qdrant
   ```

4. Test retrieval + the LLM from the command line (optional):
   ```bash
   # Fill in your keys in retriever.py and sc5_query_test.py, then:
   python3 sc5_query_test.py
   ```

5. Run the actual chat UI:
   ```bash
   # Fill in your keys in app.py, then:
   streamlit run app.py
   ```

## Notes

- `app.py`'s LLM chain automatically tries several free OpenRouter models in
  order and falls back if one is rate-limited — see
  `VERIFIED_FREE_MODELS_AUG_2026` near the top of the file. There's also a
  "Refresh free-model list" button in the sidebar that re-checks OpenRouter's
  live model list on demand.
- OpenRouter's free tier caps at 50 requests/day per account by default
  (shared across all free models). Add $10 credit to your OpenRouter account
  to raise that to 1000/day if you hit the limit.
- Never commit real API keys. Keep placeholders in version control and fill
  in real values only in your local, untracked copy.
