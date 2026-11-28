#!/usr/bin/env bash
# Runs automatically the FIRST time the Docker database starts (after 01_schema.sql).
set -euo pipefail
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<'SQL'
\copy users (user_id, username, rating, created_at) FROM '/data/users.csv' CSV HEADER
\copy listings (listing_id, seller_id, title, item, price_basis, start_price, duration_days, starts_at, ends_at, current_price, final_price, winner_id, status, n_bids) FROM '/data/listings.csv' CSV HEADER
\copy bids (bid_id, listing_id, bidder_id, amount, placed_at) FROM '/data/bids.csv' CSV HEADER
SQL
