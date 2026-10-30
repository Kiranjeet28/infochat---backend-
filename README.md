# Infochat Backend

## Load extracted notices into Supabase

The loader reads `ingestion/data/notices*.json`, maps enriched notice fields to
`public.notices`, and uses the table's unique `content_hash` constraint to ignore
hashes already stored. Raw records without a `content_hash` are skipped.

Set the project URL and a Supabase service role key in the environment, then run
the loader from the repository root:

```powershell
$env:SUPABASE_URL = "https://eidcdillbicoseugihyp.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "<service-role-key>"
python ingestion/load_supabase.py
```

Keep the service role key private and out of source control. It bypasses Row Level
Security, so the loader is intended for trusted local ingestion only.

After notices are loaded and reviewed, create chunks with the same environment
variables:

```powershell
python -m pip install -r ingestion/requirements.txt
python ingestion/chunk_notices.py
```

The chunker selects notices whose `superseded_by` is null and whose
`review_status` is not `needs_review`. It uses 400-token chunks with a 50-token
overlap, stores the source `content_hash` on every chunk, and skips notices whose
complete chunks already match that hash. If a notice changes or becomes
ineligible, its old chunks are deleted before replacements are written. The
selected embedding model is `sentence-transformers/all-MiniLM-L6-v2` (384
dimensions); the chunker records that model on each chunk. It writes the 384-value
vector and sets `embedding_status` to `succeeded`;
if model loading or a chunk encoding fails, it still saves the text and metadata,
sets the status to `failed`, and records the error.

## Retrieve current notices

Apply `ingestion/search_chunks.sql` in Supabase to install the hybrid search RPCs.
They only search chunks whose notice has `review_status = 'ok'`, is not
superseded. Both RPCs accept optional department
and semester filters. Call the retrieval function from the repository root:

```python
from ingestion.retrieve import retrieve

results = retrieve("When is the makeup examination?", semester="5th semester")
```

`retrieve` embeds the question with the same FastEmbed model used during
ingestion, runs semantic and keyword search, and returns up to five chunks
ranked by reciprocal rank fusion. Each fused candidate must also have a vector
distance below `0.65`; keyword-only candidates without a measured vector
distance are excluded. If no candidates pass, it returns `"not found"`.
Each result includes `chunk_text`, `notice_id`, `title`, `published_at`, and
`source_url` for answer evidence and source cards.
For LangChain, use the adapter, which returns `Document` objects with the chunk
as `page_content` and source fields in `metadata`:

```python
from ingestion.langchain_retriever import NoticeRetriever

retriever = NoticeRetriever(department="Computer Science Engineering Department")
documents = retriever.invoke("When is the makeup examination?")
```

Use `NOTICE_ANSWER_PROMPT` to format a grounded answer request. Supply evidence
with stable numeric labels and each chunk's notice date, for example:

```python
from ingestion.prompt import NOTICE_ANSWER_PROMPT

messages = NOTICE_ANSWER_PROMPT.invoke(
    {
        "question": "When is the makeup examination?",
        "evidence": "[1] Notice date: 2026-08-29\nSchedule details ...",
    }
)
```

Set `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` in the environment (or `.env`).

## Groq chat model

Install the added `langchain-groq` dependency, then put your Groq API key in
the ignored local `.env` file as `GROQ_API_KEY=...` (see `.env.example`). The
configured model uses `openai/gpt-oss-120b`, temperature `0`, and a 30-second
request timeout. Create it and connect it to the answer prompt like this:

```python
from ingestion.answer_chain import build_answer_chain

answer_chain = build_answer_chain()
result = answer_chain.invoke({"question": "When is the makeup examination?"})
# result: {"answer": "... [1]", "sources": [{"title": ..., "source_url": ..., ...}]}
```

The chain checks that the answer contains at least one numeric citation and
that every citation points to a retrieved document. If not, it returns
`status="invalid_citation"`, `answer="not found"`, and the retrieved source cards.

## Ask API

Run the FastAPI service from the repository root:

```powershell
uvicorn api:app --reload
```

Send `POST /api/ask` with a JSON body such as
`{"question":"When is the makeup examination?"}`. Questions must contain
1–1000 characters. The response includes `answer`, `sources`, and a `status` of
`answered`, `not_found`, or `invalid_citation`; upstream failures return a
generic `502` error.

Each notice also records `index_status` (`indexed`, `failed`, or `skipped`), an
`index_status_reason` where applicable, the `index_version`, and
`index_status_at`. The chunking run marks superseded or `needs_review` notices as
skipped, successful text indexing as indexed, and database write failures as
failed.

## Step 5: Provider fallback

The answer chain currently uses Groq only. It fails closed on provider errors;
no second provider is configured. A fallback provider must be explicitly
selected and approved because it would receive the question and retrieved
notice text too. Keep model settings, timeouts, and citation validation the
same across providers, and record which provider answered each request.

## Step 6: Embedding cache and index versioning

Indexing and retrieval share `sentence-transformers/all-MiniLM-L6-v2` through
`ingestion/embedding_config.py`. FastEmbed caches model files under
`models_cache/` in the backend folder; this directory is ignored by Git. Chunk
size, overlap, dimensions, and model are represented by `INDEX_VERSION` in
`ingestion/chunk_notices.py`. Change that version and reindex when any of those
settings change. Do not change the embedding model without rebuilding stored
embeddings.

## Daily update job status

There is no scheduled job configured yet. The existing manual stages are:

1. `python .\ingestion\run_ingestion.py` fetches notices and writes the local
   raw and extracted JSON files.
2. `python .\ingestion\load_supabase.py` inserts new notices or refreshes
   changed content while preserving existing review decisions.
3. `python .\ingestion\chunk_notices.py` indexes eligible notices and embeds
   their chunks.

Schedule these stages in that order only after configuring the runtime and
approving the recurring notice-data transfer to Supabase. Newly fetched
unreviewed notices remain `needs_review` and are not returned by retrieval.
