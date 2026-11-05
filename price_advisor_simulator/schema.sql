-- DRAFT PostgreSQL schema for BidNow simulated data (Data team).
-- The Core team owns the real schema: agree on names/types with them, then adjust
-- this file and simulator.py so both match. Safe to run more than once.

CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE,
    rating      INTEGER NOT NULL DEFAULT 0,   -- feedback score (like eBay bidderrate)
    created_at  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS listings (
    listing_id     INTEGER PRIMARY KEY,
    seller_id      INTEGER NOT NULL REFERENCES users(user_id),
    title          TEXT NOT NULL,
    item           TEXT NOT NULL,               -- category-like label for now
    price_basis    TEXT NOT NULL CHECK (price_basis IN ('data', 'assumed')),  -- 'assumed' = placeholder prices, not measured
    start_price    NUMERIC(12,2) NOT NULL CHECK (start_price >= 0),
    duration_days  SMALLINT NOT NULL CHECK (duration_days > 0),
    starts_at      TIMESTAMPTZ NOT NULL,
    ends_at        TIMESTAMPTZ NOT NULL,
    current_price  NUMERIC(12,2) NOT NULL,      -- highest bid so far (start_price if no bids yet)
    final_price    NUMERIC(12,2),               -- NULL until the auction closes
    winner_id      INTEGER REFERENCES users(user_id),
    status         TEXT NOT NULL CHECK (status IN ('active', 'closed')),
    n_bids         INTEGER NOT NULL DEFAULT 0,
    CHECK (ends_at > starts_at)
);

CREATE TABLE IF NOT EXISTS bids (
    bid_id      BIGINT PRIMARY KEY,
    listing_id  INTEGER NOT NULL REFERENCES listings(listing_id),
    bidder_id   INTEGER NOT NULL REFERENCES users(user_id),
    amount      NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    placed_at   TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bids_listing   ON bids(listing_id, placed_at);
CREATE INDEX IF NOT EXISTS idx_bids_bidder    ON bids(bidder_id);
CREATE INDEX IF NOT EXISTS idx_listings_item  ON listings(item);
CREATE INDEX IF NOT EXISTS idx_listings_state ON listings(status, ends_at);
