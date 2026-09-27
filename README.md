# TechRisk AI (MVP)

Upload a project's **BRD** and/or **SRD** and get a **technical risk register in Excel**, produced by an LLM via **OpenRouter**.

## How it works

```
BRD / SRD (.pdf .docx .txt .md)
   │  app/parsing.py      extract text, split by section into LLM-sized chunks
   ▼
Profiler agent            project name, domain, summary (context for the analyzer)
   ▼
RAG (LangChain + Chroma)  embed small pieces; for each chunk retrieve related pieces of the OTHER
                          document (BRD<->SRD) and relevant knowledge-base guidance
   ▼
Risk analyzer agent       one LLM call per chunk (+ retrieved context) -> JSON risks with a verbatim evidence quote
   ▼
Grounding check           verifies each evidence quote exists in the chunk or the retrieved BRD/SRD excerpt
   ▼
Consolidator              merges duplicates (title match, same evidence, or embedding similarity),
                          keeps the highest score
   ▼
Scoring + Excel           Score = Likelihood x Impact -> Critical / High / Medium / Low
```

Excel sheets: **Summary** (category × severity matrix, heat map, top 10), **Risk Register**
(filterable, with Owner/Status columns to fill in), **Open Questions** (for the BA/architect), **Run Info**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit it
```

## OpenRouter configuration

1. Get an API key at https://openrouter.ai/keys
2. In `.env` set `LLM_PROVIDER=openrouter` and `OPENROUTER_API_KEY=sk-or-v1-...`
3. Pick a model id from https://openrouter.ai/models:

| Model | Context | Price in/out per M tokens | Notes |
|---|---|---|---|
| `anthropic/claude-sonnet-5` (default) | 1M | $2 / $10 | best quality/cost balance |
| `anthropic/claude-opus-5.5` | 1M | $4 / $20 | highest quality |
| `google/gemini-3.8-flash` | 1M | $0.75 / $3.75 | fast, cheap; default fallback |
| `deepseek/deepseek-v4.1-flash` | 1M | very low | budget option |

(prices as listed on OpenRouter at time of writing)

### Context & size settings

All size settings default to `auto`, which reads the model's context length and max output tokens
from OpenRouter's model catalog. Set a number to override.

| Setting | Auto value | Meaning |
|---|---|---|
| `LLM_CONTEXT_TOKENS` | model's context length | total window per call |
| `LLM_MAX_OUTPUT_TOKENS` | min(8192, model limit) | max tokens generated per call; raise it if you see "reply was cut off" |
| `CHUNK_TOKENS` | fits the context, capped at `MAX_CHUNK_TOKENS` | document text per call |
| `MAX_CHUNK_TOKENS` | 30000 | smaller = more thorough per-section analysis, larger = fewer calls |

### Automatic model switching

Calls go through a chain: `LLM_MODEL` → `LLM_FALLBACK_MODELS` → `LLM_PAID_FALLBACK_MODELS`,
always using the first model that isn't blocked.

| Situation (OpenRouter error) | What happens |
|---|---|
| Model rate-limited upstream / busy (429), provider down (5xx, after 1 retry), no endpoint (404) | that model is skipped for `MODEL_COOLDOWN_SECONDS`, next model is used |
| Per-minute free limit (`free-models-per-min`) – account-wide | waits for the reset (≤ `RATE_LIMIT_MAX_WAIT_SECONDS`) and retries |
| Daily free limit (`free-models-per-day`) – account-wide | **all** free models are skipped until the reset; switches to `LLM_PAID_FALLBACK_MODELS` if set, otherwise stops with a clear message |
| No credits (402) on a paid model | paid models are skipped |
| Invalid key (401) | stops immediately |
| Every model in the chain briefly blocked | waits for the first to free up (up to `CHAIN_MAX_WAIT_SECONDS`, default 300) |

Blocks are remembered for the life of the process (e.g. across Streamlit runs). Every switch is listed in the
Streamlit results, and in the **Run Info** sheet; the sidebar and `/health` show which models are currently blocked.
Leave `LLM_PAID_FALLBACK_MODELS` empty to guarantee no credits are spent.

Oversized settings are clamped so chunk + output always fit in the model's context.
Check the resolved values, and your key's balance, with `curl localhost:8000/health` (API running), or in
the Streamlit sidebar. Each report's **Run Info** sheet records the model used, tokens and the USD cost reported by OpenRouter.

`LLM_PROVIDER=mock` (or `python cli.py --mock`) uses keyword rules instead of a model, for testing without a key.

## RAG: vector DB + knowledge base

Built with LangChain (`langchain-openai` embeddings via OpenRouter, `langchain-chroma`, text splitters) and Chroma.

| What | Where | Lifetime |
|---|---|---|
| Uploaded BRD/SRD pieces | in-memory Chroma collection | deleted after each run (confidential docs never hit disk) |
| Knowledge base | `VECTOR_DB_DIR` (default `data/chroma`) | persistent; only new/changed files are re-embedded |

What it adds to the analysis:
- **BRD ↔ SRD cross-check**: each BRD chunk sees the most related SRD pieces (and vice versa), so the model reports
  conflicts (e.g. 50,000 vs 5,000 users) and business requirements with no system requirement.
- **Knowledge base guidance**: put checklists, architecture standards, security policies or lessons learned from past
  projects (`.md .txt .pdf .docx`) in `knowledge_base/`. The matching guideline appears in the
  "Knowledge-base reference" column. A starter `tech_risk_checklist.md` is included.
- **Semantic de-duplication**: risks whose title+description embeddings have cosine similarity ≥ `DEDUP_SIMILARITY`
  are merged. The default 0.65 was calibrated on `nvidia/nemotron-3-embed-1b`; re-check it if you change `EMBEDDING_MODEL`.

Each text is embedded once per run (cached), and retrieval searches with the stored vectors, so
no extra embedding calls are made. If embedding fails, the run continues without retrieval, and Run Info says so.

```bash
python cli.py --index-kb          # (re)index knowledge_base/ after adding files (also done automatically per run)
python cli.py --brd BRD.pdf --srd SRD.docx --no-rag   # disable retrieval
```

Changing `EMBEDDING_MODEL` creates a separate knowledge-base collection (vectors from different models can't be mixed).

## Run

```bash
# Web UI
streamlit run ui/streamlit_app.py

# REST API  (POST multipart: brd, srd, project_name -> .xlsx)
uvicorn app.api:app --reload
curl -F brd=@BRD.pdf -F srd=@SRD.docx -F project_name="My Project" \
     -F model=anthropic/claude-sonnet-5 http://localhost:8000/api/analyze -o report.xlsx

# CLI
python cli.py --brd samples/sample_brd.md -o output/report.xlsx \
       [--model google/gemini-3.8-flash] [--chunk-tokens 20000] [--max-output-tokens 16000] [--mock]

# Tests
pytest -q
```

## API key & secrets

The key is never committed. Configuration is read, in order of precedence, from:
1. real environment variables
2. Streamlit secrets – `.streamlit/secrets.toml` locally, or the **Secrets** box on Streamlit Community Cloud
3. `.env` (local development)

Both `.env` and `.streamlit/secrets.toml` are git-ignored; templates are `.env.example` and
`.streamlit/secrets.toml.example`.

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io, sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
2. Repository `kantamprakash/TechRiskAI`, branch `main`, main file path **`ui/streamlit_app.py`**.
3. **Advanced settings**: Python version **3.12**, and paste the contents of `.streamlit/secrets.toml.example`
   into **Secrets** with your real `OPENROUTER_API_KEY`.
4. Deploy. Secrets can be changed later under App → Settings → Secrets (the app restarts automatically).

Notes:
- Anyone who can open the app uses *your* OpenRouter key/quota. Restrict viewers under App → Settings → Sharing,
  and keep `LLM_PAID_FALLBACK_MODELS` empty (or set a credit limit on the key) to cap spending.
- The cloud file system is temporary: the knowledge-base vector DB is rebuilt automatically after a restart.
- Uploaded documents are sent to OpenRouter model providers; free-model providers may log prompts.

## Project layout

```
app/config.py        settings from .env
app/parsing.py       PDF/DOCX/TXT extraction + section-aware chunking
app/llm.py           OpenRouter client (model catalog, sizing, JSON mode + repair, cost tracking), mock LLM
app/prompts.py       profiler and risk-analyzer prompts, scoring rubric
app/rag.py           LangChain + Chroma: embeddings, project retriever, knowledge-base indexing
app/pipeline.py      agent orchestration, grounding check, consolidation
knowledge_base/      risk checklists / standards / lessons learned used for retrieval
app/excel_report.py  Excel workbook
app/api.py           FastAPI service
ui/streamlit_app.py  Streamlit UI
cli.py               command line
```

## Next steps (post-MVP)
- LLM-based critic agent to review/merge risks and adjust scores
- Requirements traceability sheet (each BRD requirement -> best matching SRD requirement, with coverage status)
- Chat with the documents and the report (retrieval over the same vector store)
- Per-category specialist agents run in parallel
- OCR for scanned PDFs; async job queue for large documents
