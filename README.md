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
superseded, and is not marked withdrawn. Both RPCs accept optional department
and semester filters. Call the retrieval function from the repository root:

```python
from ingestion.retrieve import retrieve

results = retrieve("When is the makeup examination?", semester="5th semester")
```

`retrieve` embeds the question with the same FastEmbed model used during
ingestion, runs semantic and keyword search, and returns up to five chunks
ranked by reciprocal rank fusion. If keyword search has no matches and the
best semantic distance is above `0.75`, it returns `"not found"`. This
provisional cutoff is below the reported no-match distances (at least `0.777`);
measure correct-hit distances with `ingestion/test_search.py` before tuning it.
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
from ingestion.chat_model import create_chat_model
from ingestion.prompt import NOTICE_ANSWER_PROMPT

answer_chain = NOTICE_ANSWER_PROMPT | create_chat_model()
answer = answer_chain.invoke({"question": "...", "evidence": "[1] ..."})
```

Each notice also records `index_status` (`indexed`, `failed`, or `skipped`), an
`index_status_reason` where applicable, the `index_version`, and
`index_status_at`. The chunking run marks superseded or `needs_review` notices as
skipped, successful text indexing as indexed, and database write failures as
failed.
