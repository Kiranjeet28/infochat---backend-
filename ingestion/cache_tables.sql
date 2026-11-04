-- Review and run manually. This file is not executed by the application.
ALTER TABLE public.notices
    ADD COLUMN IF NOT EXISTS withdrawn boolean NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS public.answer_cache (
    cache_key text PRIMARY KEY,
    normalized_question_hash text NOT NULL,
    answer text NOT NULL,
    status text NOT NULL CHECK (status = 'answered'),
    sources jsonb NOT NULL,
    source_notice_ids text[] NOT NULL,
    source_content_hashes jsonb NOT NULL,
    kb_version bigint NOT NULL,
    prompt_version text NOT NULL,
    provider text,
    model text,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    hit_count bigint NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS answer_cache_cache_key_idx
    ON public.answer_cache (cache_key);
CREATE INDEX IF NOT EXISTS answer_cache_expires_at_idx
    ON public.answer_cache (expires_at);

CREATE TABLE IF NOT EXISTS public.kb_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    kb_version bigint NOT NULL DEFAULT 1,
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO public.kb_state (singleton, kb_version)
VALUES (true, 1)
ON CONFLICT (singleton) DO NOTHING;

ALTER TABLE public.answer_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.kb_state ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.answer_cache FROM PUBLIC, anon, authenticated;
REVOKE ALL ON public.kb_state FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.answer_cache TO service_role;
GRANT ALL ON public.kb_state TO service_role;

CREATE OR REPLACE FUNCTION public.bump_kb_version()
RETURNS bigint
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.kb_state (singleton, kb_version, updated_at)
    VALUES (true, 2, now())
    ON CONFLICT (singleton)
    DO UPDATE SET kb_version = public.kb_state.kb_version + 1, updated_at = now()
    RETURNING kb_version;
$$;

-- PostgreSQL grants EXECUTE on new functions to PUBLIC by default, and Supabase
-- exposes public-schema functions through RPC. Restrict this writer to service_role.
REVOKE EXECUTE ON FUNCTION public.bump_kb_version()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.bump_kb_version()
    TO service_role;

CREATE OR REPLACE FUNCTION public.get_kb_version()
RETURNS bigint
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT kb_version FROM public.kb_state WHERE singleton = true;
$$;

CREATE OR REPLACE FUNCTION public.verify_cached_sources(
    source_notice_ids text[],
    source_content_hashes jsonb
)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT cardinality(source_notice_ids) > 0
       AND NOT EXISTS (
           SELECT 1
           FROM unnest(source_notice_ids) AS requested(notice_id)
           LEFT JOIN public.notices AS n ON n.notice_id = requested.notice_id
           WHERE n.notice_id IS NULL
              OR n.content_hash::text IS DISTINCT FROM (source_content_hashes ->> requested.notice_id)
              OR n.superseded_by IS NOT NULL
              OR n.withdrawn IS TRUE
              OR n.review_status <> 'ok'
       );
$$;

REVOKE ALL ON FUNCTION public.bump_kb_version() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.get_kb_version() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.verify_cached_sources(text[], jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.bump_kb_version() TO service_role;
GRANT EXECUTE ON FUNCTION public.get_kb_version() TO service_role;
GRANT EXECUTE ON FUNCTION public.verify_cached_sources(text[], jsonb) TO service_role;
