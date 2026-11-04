-- Review and run manually. Search RPCs exclude withdrawn notices.
DROP FUNCTION IF EXISTS public.search_chunks_similarity(extensions.vector, integer, text, text);
DROP FUNCTION IF EXISTS public.search_chunks_keyword(text, integer, text, text);

-- Keep the current RPC signatures and result columns from search_chunks.sql.
-- The only freshness predicate addition is: AND n.withdrawn IS NOT TRUE
-- Add that predicate to each function's WHERE clause after joining notices n.

CREATE OR REPLACE FUNCTION public.search_chunks_similarity(
    query_embedding extensions.vector(384),
    match_count integer DEFAULT 5,
    filter_department text DEFAULT NULL,
    filter_semester text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid, notice_id text, chunk_text text, title text,
    published_at text, source_url text, department text, semester text,
    content_hash text, score double precision
)
LANGUAGE sql STABLE SET search_path = public, extensions
AS $$
    SELECT c.chunk_id, c.notice_id, c.text::text, c.title::text,
           c.published_at::text, c.source_url::text, n.department::text,
           n.semester::text, c.content_hash::text,
           (c.embedding <=> query_embedding)::double precision
    FROM public.chunks c
    JOIN public.notices n ON n.notice_id = c.notice_id
    WHERE c.embedding IS NOT NULL AND n.review_status = 'ok'
      AND n.superseded_by IS NULL AND n.withdrawn IS NOT TRUE
      AND (filter_department IS NULL OR n.department = filter_department)
      AND (filter_semester IS NULL OR n.semester = filter_semester)
    ORDER BY c.embedding <=> query_embedding, c.chunk_id
    LIMIT greatest(match_count, 0);
$$;

CREATE OR REPLACE FUNCTION public.search_chunks_keyword(
    search_query text,
    match_count integer DEFAULT 5,
    filter_department text DEFAULT NULL,
    filter_semester text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid, notice_id text, chunk_text text, title text,
    published_at text, source_url text, department text, semester text,
    content_hash text, score double precision
)
LANGUAGE sql STABLE SET search_path = public, extensions
AS $$
    SELECT c.chunk_id, c.notice_id, c.text::text, c.title::text,
           c.published_at::text, c.source_url::text, n.department::text,
           n.semester::text, c.content_hash::text,
           ts_rank_cd(c.search_vector,
               websearch_to_tsquery('simple', search_query))::double precision
    FROM public.chunks c
    JOIN public.notices n ON n.notice_id = c.notice_id
    WHERE c.search_vector @@ websearch_to_tsquery('simple', search_query)
      AND n.review_status = 'ok' AND n.superseded_by IS NULL
      AND n.withdrawn IS NOT TRUE
      AND (filter_department IS NULL OR n.department = filter_department)
      AND (filter_semester IS NULL OR n.semester = filter_semester)
    ORDER BY ts_rank_cd(c.search_vector,
        websearch_to_tsquery('simple', search_query)) DESC, c.chunk_id
    LIMIT greatest(match_count, 0);
$$;

REVOKE EXECUTE ON FUNCTION public.search_chunks_similarity(extensions.vector, integer, text, text)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.search_chunks_similarity(extensions.vector, integer, text, text)
    TO service_role;
REVOKE EXECUTE ON FUNCTION public.search_chunks_keyword(text, integer, text, text)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.search_chunks_keyword(text, integer, text, text)
    TO service_role;

NOTIFY pgrst, 'reload schema';
