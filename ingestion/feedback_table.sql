-- Answer feedback (thumbs up/down). Review and run this yourself in the Supabase SQL editor.
-- Idempotent: safe to run more than once. Stores NO question and NO answer text.

CREATE TABLE IF NOT EXISTS public.feedback (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  timestamptz NOT NULL DEFAULT now(),
    request_id  text NOT NULL CHECK (request_id ~ '^[0-9a-f]{1,64}$'),
    rating      text NOT NULL CHECK (rating IN ('up', 'down')),
    reason      text CHECK (reason IS NULL OR reason IN ('wrong_answer', 'outdated', 'not_relevant', 'other')),
    comment     text CHECK (comment IS NULL OR char_length(comment) <= 200)
);

CREATE INDEX IF NOT EXISTS feedback_created_at_idx ON public.feedback (created_at DESC);

-- Row level security on with no policies: only the service role (which bypasses RLS) can read or write.
ALTER TABLE public.feedback ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.feedback FROM PUBLIC, anon, authenticated;
GRANT ALL ON TABLE public.feedback TO service_role;
