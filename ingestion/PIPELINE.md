# CascadeIQ ingestion pipeline

Repository settings: add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as Actions secrets. `NOTICES_API_URL` is optional and defaults to `https://infocascade-database.onrender.com/api/notices`; set it as an Actions variable to override the default. The workflow maps these only into its run step.

Scheduled runs are disabled unless repository variable `INGEST_ENABLED` is exactly `true`. Manual `workflow_dispatch` runs remain available regardless of that variable.

The pipeline exits with code `2` when ingestion completes with warnings; the workflow marks that run complete with a warning annotation and uploads the summary. Exit code `1` remains a hard failure. Failed notice summaries contain only notice IDs and safe failure codes, not notice text or exception details.

To run manually, open **Actions → CascadeIQ ingestion → Run workflow**, choose `incremental` or `full`, and leave dry run enabled for the first manual run. The pipeline summary JSON is uploaded as a seven day artifact; the step summary contains counts and review reason codes by notice ID. Warnings explain an incomplete fetch or a skipped full mode withdrawal (fetch incomplete, threshold exceeded, or missing `withdrawn` column).

For local dry run, set the three values in the process environment or in `backend/.env`, then run from the repository root:

```powershell
python .\ingestion\run_pipeline.py --mode incremental --dry-run
```

First-run order: manual full dry run, manual full real run, then let the six-hour schedule take over. The API warm-up and source text extraction need network access when run outside this offline development sandbox.

The minimum notice text length is configurable with `MIN_NOTICE_TEXT_LENGTH` (default `40`).

## Fetch safety (all notice fetches go through `ingestion/safe_fetch.py`)

- https only, no credentials in URLs, standard port, host must be `gndec.ac.in` (or a subdomain) or the host of `NOTICES_API_URL`. Extra hosts: `SAFE_FETCH_ALLOWED_HOSTS` (comma separated, exact or subdomain match).
- Every address the host resolves to must be public (private, loopback, link-local, multicast and reserved ranges are refused), re-checked on every redirect. At most 5 redirects, never https to http.
- Body caps: 25 MB for PDFs, 5 MB for HTML and JSON (decoded bytes). Content types allowed: `application/pdf`, `text/html`, `application/json`. A PDF served with another type is refused. PDFs are capped at 100 pages and 500000 characters.
- Errors never contain URLs, query strings or keys.
- If the official notice page cannot be read, the notice is marked failed (`official_page_unreachable`) and the API teaser text is never stored instead. If more than 20 percent of notices fail, the run exits 1.
- Warm-up of the notices API: 10 s connect timeout, 90 s read timeout, at most 4 attempts with exponential backoff, one overall deadline `WARMUP_DEADLINE_SECONDS` (default 240).

## Review gate and restricted notices

Notices that fail the gate stay `needs_review` and are never chunked or searched. Approval is always a manual decision (see `qa/review_queue.md` for the current list). Notices whose title or category matches `RESTRICTED_NOTICE_PATTERNS` (default `struck off,list of oe,readmission`) are shown to the model only as a privacy summary with the source link, even after approval.

## Backfilling chunks for named notices

After you approve notices, index only those (nothing else is touched):

```powershell
python -m ingestion.chunk_notices --ids <id1>,<id2> --dry-run   # shows what would be written
python -m ingestion.chunk_notices --ids <id1>,<id2>             # writes chunks and embeddings
```

Then bump the knowledge-base version once with `SELECT public.bump_kb_version();` so cached answers refresh.

## Settings (names; defaults in `.env.example`)

`NOTICES_API_URL`, `SAFE_FETCH_ALLOWED_HOSTS`, `ALLOWED_SOURCE_DOMAINS`, `MIN_NOTICE_TEXT_LENGTH`, `WARMUP_DEADLINE_SECONDS`, `RESTRICTED_NOTICE_PATTERNS`, `EMBEDDING_ALLOW_DOWNLOAD`.
