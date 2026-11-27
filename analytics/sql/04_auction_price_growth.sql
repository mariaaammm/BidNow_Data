-- Purpose: Price appreciation from starting price to final winning price on completed auctions
-- Grain: Item category (filtered to auctions with both item_created and auction_won events)

WITH auction_prices AS (
    SELECT
        auction_id,
        MAX(category) AS category,
        MAX(CASE WHEN event_name = 'item_created' THEN bid_amount END) AS start_price,
        MAX(CASE WHEN event_name = 'auction_won' THEN bid_amount END) AS final_price,
        COUNT(CASE WHEN event_name = 'bid_placed' THEN 1 END) AS bids_count
    FROM user_events
    WHERE auction_id IS NOT NULL
      AND auction_id != ''
    GROUP BY auction_id
)
SELECT
    category,
    COUNT(*) AS completed_auctions,
    ROUND(AVG(bids_count), 1) AS avg_bids_per_auction,
    ROUND(AVG(start_price), 0) AS avg_start_price,
    ROUND(AVG(final_price), 0) AS avg_final_price,
    ROUND(AVG((final_price - start_price) * 100.0 / NULLIF(start_price, 0)), 1) AS avg_price_growth_pct
FROM auction_prices
WHERE final_price IS NOT NULL
  AND start_price IS NOT NULL
GROUP BY category
ORDER BY avg_price_growth_pct DESC;