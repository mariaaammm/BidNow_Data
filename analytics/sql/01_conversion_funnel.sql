SELECT
    event_name,
    COUNT(*) AS total_events,
    COUNT(DISTINCT user_id) AS unique_users
FROM user_events
WHERE event_name IN ('item_viewed', 'bid_placed', 'auction_won')
GROUP BY event_name
ORDER BY CASE event_name
    WHEN 'item_viewed' THEN 1
    WHEN 'bid_placed' THEN 2
    WHEN 'auction_won' THEN 3
END;