# CascadeIQ deployment (invite-only beta)

Backend: FastAPI in a Docker image (Render or similar). Frontend: static site (Vercel). Data: Supabase.
Nothing in this repo publishes anything by itself; every step below is yours to run.

## 1. Facts to plan around

- **Memory:** about 240 MB resident (measured: 226 MB after startup, 241 MB after live questions, peak 259 MB). A 512 MB host is enough for one instance.
- **One instance only for the beta.** Rate limits and the daily model cap live in memory per process: they reset on restart and each extra instance counts separately.
- **Python 3.12** in the image (the pinned `numpy` in `requirements-api.txt` needs 3.12 or newer). The ingestion workflow still uses 3.11 with older pins; see "Ingestion".
- **Cold start:** the API loads the embedding model at startup. Free hosts that sleep add a long first request; see keep-alive below.
- The Docker image was **not built or run here** (Docker is not installed on the development PC). The API itself was run with the venv and passed `qa/smoke_deployed.py`.

## 2. Environment variables (names only; values in the host's secret store)

| Name | Where | Purpose |
|---|---|---|
| `GROQ_API_KEY` | backend, secret | model provider |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | backend, secret | database, cache, feedback |
| `CORS_ALLOWED_ORIGINS` | backend | exact frontend origin(s), comma separated, https |
| `TRUST_PROXY_HEADERS` | backend | `1` behind the host's proxy so limits see the real client |
| `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY` | backend | per client (10 / 80) |
| `GLOBAL_DAILY_MODEL_CALLS` | backend | real model calls per IST day; set from your Groq limit |
| `FEEDBACK_PER_DAY` | backend | feedback per client (20) |
| `GROQ_ENABLED`, `GROQ_MODEL`, `GEMINI_ENABLED` | backend | providers (Gemini stays off without a key) |
| `CACHE_ENABLED`, `CACHE_TTL_SECONDS` | backend | answer cache |
| `RESTRICTED_NOTICE_PATTERNS` | backend | notices withheld from the model (student lists) |
| `LOG_FILE` | backend, optional | also write a rotating log file (stdout is always used) |
| `RATE_LIMIT_SALT` | backend, optional secret | fixed salt for client hashes |
| `PORT` | set by the host | uvicorn binds `0.0.0.0:$PORT` |
| Frontend API base URL | frontend (Vercel) | the variable name is in the frontend's own env example; set it to the backend's `https://` URL |

The full list with defaults is in `.env.example`. Never put a real value there.

## 3. First deploy checklist

1. Run `ingestion/feedback_table.sql` in the Supabase SQL editor (otherwise `/api/feedback` answers 503). Confirm `cache_tables.sql` was applied earlier.
2. Create the backend service from the `Dockerfile` (or `render.yaml`). Set the secret variables in the dashboard. Leave auto-deploy off.
3. Set `CORS_ALLOWED_ORIGINS` to the exact deployed frontend origin (scheme and host, no trailing slash, no wildcard).
4. Set `TRUST_PROXY_HEADERS=1` and `GLOBAL_DAILY_MODEL_CALLS` from your Groq daily limit.
5. Health check path `/health`. Wait until it returns `{"status":"ok"}`.
6. Deploy the frontend with the backend URL; use https only for both.
7. Run `python qa/smoke_deployed.py <API_URL> <FRONTEND_ORIGIN>`; all checks must pass (3 live questions).
8. Confirm in the data that no student-list notice is approved without your decision (`qa/review_queue.md`).
9. Keep-alive: ping `GET /health` every 10 minutes from an uptime monitor. It never calls the model.
10. Tell beta students it is an unofficial assistant and to check the linked official notice.

## 4. Rollback

- Backend: in the host dashboard redeploy the previous successful image/commit (keep auto-deploy off so you choose when). Settings stay as they are.
- Frontend: promote the previous deployment in Vercel.
- Bad answers after a data change: bump the knowledge-base version (section 6) so cached answers are recomputed; restore notice rows from the backup under `qa/backup/<timestamp>/` if needed (ask for help before restoring).
- If the feedback or cache SQL caused trouble, both scripts are additive (`IF NOT EXISTS`); nothing is dropped.

## 5. Key rotation

1. Create the new key at the provider (Groq console, or Supabase: project settings, API keys).
2. Update the host's environment variable and, for Supabase, the GitHub Actions secrets `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`; update local `.env`.
3. Redeploy or restart the backend; run the smoke script.
4. Revoke the old key. Rotate immediately if a key was ever committed, pasted in a chat or logged.

## 6. Operations runbook

- **Restart:** use the host's manual restart. Limits and the daily cap reset; the cache survives (it is in Supabase).
- **Read logs:** host log view (stdout). One line per event: `status=...`, never the question text. Search by `request_id` (the API returns it in `X-Request-ID`).
- **Clear or refresh the cache:** there is no `clear_cache.py` in the repo. In the Supabase SQL editor run `SELECT public.bump_kb_version();` (all cached answers become stale and are recomputed), or `DELETE FROM public.answer_cache;` to remove them.
- **Groq quota reached:** users see "busy today" with related notice cards once `GLOBAL_DAILY_MODEL_CALLS` is hit, or `ai_unavailable` if Groq itself errors. Check the Groq console; raise the cap only if the provider limit allows; otherwise wait for the next IST day. A Gemini fallback needs a key and acceptance of its terms before turning `GEMINI_ENABLED` on.
- **Ingestion failed:** open the run summary (Actions artifact `cascadeiq-ingest-summary`). `official_page_unreachable` means the college site was unreachable: nothing was written, rerun later. More than 20 percent failed makes the run exit 1 by design.
- **High error rate or slow answers:** check the Groq status and the `provider=` / `attempts=` fields in the logs.

## 7. Ingestion schedule

- The workflow `.github/workflows/cascadeiq-ingest.yml` runs on a schedule only when the repository variable `INGEST_ENABLED` is `true`; manual runs always work.
- Enable it only after **two successful manual real runs**: the first may change data; the second must report `new 0` and `changed 0`.
- Before enabling, align the workflow's Python version and `ingestion/requirements-ingest.txt` pins with the tested set (see `requirements-api.txt`); today they differ.

## 8. Docker (local test, when Docker is available)

```bash
docker build -t cascadeiq-api .
docker run --rm -p 8002:8000 --env-file .env cascadeiq-api
python qa/smoke_deployed.py http://127.0.0.1:8002 http://127.0.0.1:5173
```

`.env` is passed at run time and is excluded from the image by `.dockerignore`.
