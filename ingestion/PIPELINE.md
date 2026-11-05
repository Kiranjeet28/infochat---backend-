# CascadeIQ ingestion pipeline

Repository settings: add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as Actions secrets, and `NOTICES_API_URL` as an Actions variable. The workflow maps these only into its run step.

To run manually, open **Actions → CascadeIQ ingestion → Run workflow**, choose `incremental` or `full`, and leave dry run enabled for the first manual run. The pipeline summary JSON is uploaded as a seven day artifact; the step summary contains counts and review reason codes by notice ID. Warnings explain an incomplete fetch or a skipped full mode withdrawal (fetch incomplete, threshold exceeded, or missing `withdrawn` column).

For local dry run, set the three values in the process environment or in `backend/.env`, then run from the repository root:

```powershell
python .\ingestion\run_pipeline.py --mode incremental --dry-run
```

First-run order: manual full dry run, manual full real run, then let the six-hour schedule take over. The API warm-up and source text extraction need network access when run outside this offline development sandbox.
