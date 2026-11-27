-- Purpose: Category-level engagement, bid volume, and won GMV breakdown
-- Grain: Item category
-- Notes: total_won_gmv sums bid_amount exclusively on auction_won events to avoid double counting

SELECT
    category,
    COUNT(CASE WHEN event_name = 'item_viewed' THEN 1 END) AS total_views,
    COUNT(CASE WHEN event_name = 'bid_placed' THEN 1 END) AS total_bids,
    COUNT(CASE WHEN event_name = 'auction_won' THEN 1 END) AS auctions_won,
    ROUND(AVG(CASE WHEN event_name = 'auction_won' THEN bid_amount END), 0) AS avg_winning_price,
    COALESCE(SUM(CASE WHEN event_name = 'auction_won' THEN bid_amount END), 0) AS total_won_gmv
FROM user_events
WHERE category IS NOT NULL
  AND category != ''
GROUP BY category
ORDER BY total_won_gmv DESC;