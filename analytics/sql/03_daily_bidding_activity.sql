SELECT
    DATE(timestamp) AS event_date,
    COUNT(CASE WHEN event_name = 'item_viewed' THEN 1 END) AS item_views,
    COUNT(CASE WHEN event_name = 'bid_placed' THEN 1 END) AS bids_placed,
    COUNT(CASE WHEN event_name = 'auction_won' THEN 1 END) AS auctions_won,
    COALESCE(SUM(CASE WHEN event_name = 'auction_won' THEN bid_amount END), 0) AS daily_won_gmv
FROM user_events
GROUP BY DATE(timestamp)
ORDER BY event_date ASC;