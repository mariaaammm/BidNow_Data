-- Dashboard queries for the simulated BidNow data. Plain PostgreSQL: paste each one into
-- Metabase as a new "SQL question" (or run with psql). Q1-Q7 work on any data with schema.sql.
-- Numbers about prices should use price_basis = 'data' unless you really mean the assumed categories.

-- Q1  Listings by item and status
SELECT item, status, count(*) AS listings
FROM listings
GROUP BY item, status
ORDER BY item, status;

-- Q2  Competition per closed auction: how many bids and bidders?
SELECT l.item,
       count(*)                                                       AS auctions,
       round(avg(l.n_bids), 1)                                        AS avg_bids,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY l.n_bids)          AS median_bids,
       round(avg(d.bidders), 1)                                       AS avg_bidders
FROM listings l
JOIN (SELECT listing_id, count(DISTINCT bidder_id) AS bidders FROM bids GROUP BY listing_id) d USING (listing_id)
WHERE l.status = 'closed'
GROUP BY l.item
ORDER BY auctions DESC;

-- Q3  Final price per item: typical value and spread (closed auctions)
SELECT item,
       count(*)                                                         AS auctions,
       percentile_cont(0.1) WITHIN GROUP (ORDER BY final_price)::numeric(12,2) AS p10,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY final_price)::numeric(12,2) AS median,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY final_price)::numeric(12,2) AS p90
FROM listings
WHERE status = 'closed' AND price_basis = 'data'
GROUP BY item
ORDER BY median DESC;

-- Q4  When do bids arrive? Share of bids by time left before the auction ends (closed auctions)
SELECT CASE WHEN left_h <= 1   THEN '1  last hour'
            WHEN left_h <= 6   THEN '2  1-6 hours before end'
            WHEN left_h <= 24  THEN '3  6-24 hours before end'
            WHEN left_h <= 72  THEN '4  1-3 days before end'
            ELSE                    '5  more than 3 days before end' END AS time_before_end,
       count(*)                                                          AS bids,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1)                AS pct_of_bids
FROM (SELECT extract(epoch FROM (l.ends_at - b.placed_at)) / 3600 AS left_h
      FROM bids b JOIN listings l USING (listing_id)
      WHERE l.status = 'closed') t
GROUP BY 1
ORDER BY 1;

-- Q5  Bids and new listings per day (activity over time)
SELECT d::date AS day,
       (SELECT count(*) FROM bids     WHERE placed_at::date = d::date) AS bids,
       (SELECT count(*) FROM listings WHERE starts_at::date = d::date) AS new_listings
FROM generate_series((SELECT min(starts_at) FROM listings)::date,
                     (SELECT max(placed_at) FROM bids)::date, interval '1 day') AS d
ORDER BY day;

-- Q6  Are a few bidders doing most of the bidding? Share of bids by bidder group
SELECT CASE WHEN decile = 1 THEN 'top 10% of bidders' WHEN decile <= 5 THEN 'next 40%' ELSE 'bottom 50%' END AS bidder_group,
       count(*)                                          AS bidders,
       sum(n)                                            AS bids,
       round(100.0 * sum(n) / sum(sum(n)) OVER (), 1)    AS pct_of_bids
FROM (SELECT bidder_id, count(*) AS n, ntile(10) OVER (ORDER BY count(*) DESC) AS decile
      FROM bids GROUP BY bidder_id) t
GROUP BY 1
ORDER BY min(decile);

-- Q7  Live auctions ending soon, with the current price
SELECT listing_id, item, current_price, n_bids,
       round((extract(epoch FROM (ends_at - now())) / 3600)::numeric, 1) AS hours_left
FROM listings
WHERE status = 'active' AND ends_at > now()
ORDER BY ends_at
LIMIT 20;

-- Q8  Does a low opening bid attract more bids? (closed auctions of measured items)
-- opening bid is compared with the item's typical (median) final price.
WITH typical AS (
    SELECT item, percentile_cont(0.5) WITHIN GROUP (ORDER BY final_price) AS med
    FROM listings WHERE status = 'closed' AND price_basis = 'data' GROUP BY item)
SELECT CASE WHEN l.start_price / t.med < 0.05 THEN '1  under 5%'
            WHEN l.start_price / t.med < 0.25 THEN '2  5-25%'
            WHEN l.start_price / t.med < 0.60 THEN '3  25-60%'
            WHEN l.start_price / t.med < 0.90 THEN '4  60-90%'
            ELSE                                   '5  over 90%' END AS opening_bid_vs_typical_price,
       count(*)                                                      AS auctions,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY l.n_bids)         AS median_bids,
       round((percentile_cont(0.5) WITHIN GROUP (ORDER BY l.final_price / t.med))::numeric, 2) AS median_final_vs_typical
FROM listings l JOIN typical t USING (item)
WHERE l.status = 'closed' AND l.price_basis = 'data'
GROUP BY 1
ORDER BY 1;
