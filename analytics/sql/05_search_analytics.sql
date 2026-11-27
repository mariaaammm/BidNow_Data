-- Purpose: Top search queries, authenticated user reach, and guest traffic share
-- Grain: Normalized search_query

SELECT
    LOWER(TRIM(search_query)) AS search_query,
    COUNT(*) AS total_searches,
    COUNT(DISTINCT user_id) AS unique_auth_users,
    ROUND(COUNT(CASE WHEN user_id IS NULL THEN 1 END) * 100.0 / COUNT(*), 1) AS guest_share_pct
FROM user_events
WHERE event_name = 'search_performed'
  AND search_query IS NOT NULL
  AND TRIM(search_query) != ''
GROUP BY LOWER(TRIM(search_query))
ORDER BY total_searches DESC
LIMIT 15;