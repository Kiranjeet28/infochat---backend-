-- Required by ingestion/test_search.py for database-ranked search results.
-- Review this SQL before running it in Supabase; the script does not apply it.
CREATE OR REPLACE FUNCTION public.search_chunks_similarity(
    query_embedding extensions.vector(384),
    match_count integer DEFAULT 5
)
RETURNS TABLE (
    chunk_id uuid,
    title text,
    published_at text,
    source_url text,
    score double precision
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
    SELECT
        c.chunk_id,
        c.title::text,
        c.published_at::text,
        c.source_url::text,
        (c.embedding <=> query_embedding)::double precision AS score
    FROM public.chunks AS c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding ASC, c.chunk_id
    LIMIT greatest(match_count, 0);
$$;

CREATE OR REPLACE FUNCTION public.search_chunks_keyword(
    search_query text,
    match_count integer DEFAULT 5
)
RETURNS TABLE (
    chunk_id uuid,
    title text,
    published_at text,
    source_url text,
    score double precision
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
    SELECT
        c.chunk_id,
        c.title::text,
        c.published_at::text,
        c.source_url::text,
        ts_rank_cd(
            c.search_vector,
            websearch_to_tsquery('simple', search_query)
        )::double precision AS score
    FROM public.chunks AS c
    WHERE c.search_vector @@ websearch_to_tsquery('simple', search_query)
    ORDER BY score DESC, c.chunk_id
    LIMIT greatest(match_count, 0);
$$;
