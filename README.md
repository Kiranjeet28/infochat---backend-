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
overlap and upserts by `(notice_id, chunk_index)` when rerun. The
selected embedding model is `sentence-transformers/all-MiniLM-L6-v2` (384
dimensions); the chunker records that model on each chunk. It writes the 384-value
vector and sets `embedding_status` to `succeeded`;
if model loading or a chunk encoding fails, it still saves the text and metadata,
sets the status to `failed`, and records the error for retry.
