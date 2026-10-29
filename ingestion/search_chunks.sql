-- Required by ingestion/test_search.py for database-ranked search results.
-- Review this SQL before running it in Supabase; the script does not apply it.
-- Remove the prior signatures so PostgREST cannot see ambiguous RPC overloads.
DROP FUNCTION IF EXISTS public.search_chunks_similarity(extensions.vector, integer);
DROP FUNCTION IF EXISTS public.search_chunks_keyword(text, integer);

CREATE OR REPLACE FUNCTION public.search_chunks_similarity(
    query_embedding extensions.vector(384),
    match_count integer DEFAULT 5,
    filter_department text DEFAULT NULL,
    filter_semester text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid,
    notice_id text,
    chunk_text text,
    title text,
    published_at text,
    source_url text,
    department text,
    semester text,
    score double precision
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
    SELECT
        c.chunk_id,
        c.notice_id,
        c.text::text AS chunk_text,
        c.title::text,
        c.published_at::text,
        c.source_url::text,
        n.department::text,
        n.semester::text,
        (c.embedding <=> query_embedding)::double precision AS score
    FROM public.chunks AS c
    JOIN public.notices AS n ON n.notice_id = c.notice_id
    WHERE c.embedding IS NOT NULL
      AND n.review_status = 'ok'
      AND n.superseded_by IS NULL
      AND (filter_department IS NULL OR n.department = filter_department)
      AND (filter_semester IS NULL OR n.semester = filter_semester)
    ORDER BY c.embedding <=> query_embedding ASC, c.chunk_id
    LIMIT greatest(match_count, 0);
$$;

CREATE OR REPLACE FUNCTION public.search_chunks_keyword(
    search_query text,
    match_count integer DEFAULT 5,
    filter_department text DEFAULT NULL,
    filter_semester text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid,
    notice_id text,
    chunk_text text,
    title text,
    published_at text,
    source_url text,
    department text,
    semester text,
    score double precision
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
    SELECT
        c.chunk_id,
        c.notice_id,
        c.text::text AS chunk_text,
        c.title::text,
        c.published_at::text,
        c.source_url::text,
        n.department::text,
        n.semester::text,
        ts_rank_cd(
            c.search_vector,
            websearch_to_tsquery('simple', search_query)
        )::double precision AS score
    FROM public.chunks AS c
    JOIN public.notices AS n ON n.notice_id = c.notice_id
    WHERE c.search_vector @@ websearch_to_tsquery('simple', search_query)
      AND n.review_status = 'ok'
      AND n.superseded_by IS NULL
      AND (filter_department IS NULL OR n.department = filter_department)
      AND (filter_semester IS NULL OR n.semester = filter_semester)
    ORDER BY score DESC, c.chunk_id
    LIMIT greatest(match_count, 0);
$$;
