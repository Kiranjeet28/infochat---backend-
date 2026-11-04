-- MANUAL ONLY: run manually, one statement at a time. Nothing runs by default.

-- Increment kb_version to prove old cache keys miss; expected one returned version.
-- SELECT public.bump_kb_version();

-- Inspect cache rows without selecting answer text.
-- SELECT cache_key, status, source_notice_ids, expires_at, hit_count
-- FROM public.answer_cache ORDER BY created_at DESC;

-- Delete only test entries. Replace the placeholder with the normalized_question_hash
-- for a question used exclusively in your manual cache test. This deletes cached
-- answers for that specific test question, across cache-key versions/dates.
-- DELETE FROM public.answer_cache
-- WHERE normalized_question_hash = '<test-question-sha256>';
