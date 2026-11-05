#!/usr/bin/env bash
# Load the simulator CSVs into any PostgreSQL (local install, Docker, ...).
# Usage: ./load_db.sh "postgresql://bidnow:bidnow@localhost:5432/bidnow" [csv-folder]
# Re-running replaces the old rows. Needs the `psql` command.
set -euo pipefail
URL="${1:?usage: ./load_db.sh <postgres-url> [csv-folder, default sim_out]}"
DIR="${2:-sim_out}"
cd "$(dirname "$0")"
psql "$URL" -v ON_ERROR_STOP=1 -q -f schema.sql
psql "$URL" -v ON_ERROR_STOP=1 -q <<SQL
TRUNCATE bids, listings, users;
\copy users (user_id, username, rating, created_at) FROM '$DIR/users.csv' CSV HEADER
\copy listings (listing_id, seller_id, title, item, price_basis, start_price, duration_days, starts_at, ends_at, current_price, final_price, winner_id, status, n_bids) FROM '$DIR/listings.csv' CSV HEADER
\copy bids (bid_id, listing_id, bidder_id, amount, placed_at) FROM '$DIR/bids.csv' CSV HEADER
SQL
psql "$URL" -At -c "SELECT 'loaded: ' || (SELECT count(*) FROM users) || ' users, ' || (SELECT count(*) FROM listings) || ' listings, ' || (SELECT count(*) FROM bids) || ' bids'"
