-- Read-only verification. Run after cache_tables.sql and search_chunks_freshness.sql.

-- Expected: exactly 3 rows: both search RPCs and bump_kb_version; no other search_chunks% function.
SELECT p.oid::regprocedure AS signature
FROM pg_proc AS p
JOIN pg_namespace AS n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
  AND (p.proname LIKE 'search_chunks%' OR p.proname = 'bump_kb_version')
ORDER BY p.proname;

-- Expected: exactly those 3 rows; EXECUTE grantees contain only postgres and service_role.
SELECT p.oid::regprocedure AS signature,
       p.proacl,
       array_agg(DISTINCT CASE WHEN acl.grantee = 0 THEN 'PUBLIC' ELSE grantee.rolname END
                 ORDER BY CASE WHEN acl.grantee = 0 THEN 'PUBLIC' ELSE grantee.rolname END)
           FILTER (WHERE acl.privilege_type = 'EXECUTE') AS execute_grantees
FROM pg_proc AS p
JOIN pg_namespace AS n ON n.oid = p.pronamespace
CROSS JOIN LATERAL aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) AS acl
LEFT JOIN pg_roles AS grantee ON grantee.oid = acl.grantee
WHERE n.nspname = 'public'
  AND (p.proname LIKE 'search_chunks%' OR p.proname = 'bump_kb_version')
GROUP BY p.oid
ORDER BY p.proname;

-- Expected: 2 rows; relrowsecurity is true for both tables.
SELECT relname, relrowsecurity
FROM pg_class
WHERE relname IN ('answer_cache', 'kb_state')
ORDER BY relname;

-- Expected: service_role can appear; anon and authenticated must not appear.
SELECT table_name, grantee, privilege_type
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND table_name IN ('answer_cache', 'kb_state')
ORDER BY table_name, grantee, privilege_type;

-- Expected: one row; withdrawn exists and its default is false.
SELECT column_name, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'notices' AND column_name = 'withdrawn';

-- Expected: one row with current counts; use this to check that neither table lost rows.
SELECT (SELECT count(*) FROM public.notices) AS notices_count,
       (SELECT count(*) FROM public.chunks) AS chunks_count;

-- Expected: one singleton state row with the current kb_version.
SELECT singleton, kb_version, updated_at
FROM public.kb_state
ORDER BY singleton;

-- Expected: cache metadata only; answer text is intentionally not selected.
SELECT cache_key, status, source_notice_ids, expires_at, hit_count
FROM public.answer_cache
ORDER BY created_at DESC;
