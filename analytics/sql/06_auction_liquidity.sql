-- Purpose: Marketplace liquidity — share of created listings receiving bids and closing with a winner
-- Grain: Item category

WITH auction_outcomes AS (
    SELECT
        auction_id,
        MAX(category) AS category,
        MAX(CASE WHEN event_name = 'item_created' THEN 1 ELSE 0 END) AS is_created,
        MAX(CASE WHEN event_name = 'bid_placed' THEN 1 ELSE 0 END) AS has_bids,
        MAX(CASE WHEN event_name = 'auction_won' THEN 1 ELSE 0 END) AS is_won
    FROM user_events
    WHERE auction_id IS NOT NULL
      AND auction_id != ''
    GROUP BY auction_id
)
SELECT
    category,
    SUM(is_created) AS created_auctions,
    SUM(has_bids) AS auctions_with_bids,
    SUM(is_won) AS won_auctions,
    ROUND(SUM(has_bids) * 100.0 / NULLIF(SUM(is_created), 0), 1) AS bid_rate_pct,
    ROUND(SUM(is_won) * 100.0 / NULLIF(SUM(is_created), 0), 1) AS win_rate_pct
FROM auction_outcomes
WHERE is_created = 1
  AND category IS NOT NULL
GROUP BY category
ORDER BY bid_rate_pct DESC, created_auctions DESC;