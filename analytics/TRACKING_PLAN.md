# BidNow — Analytics Tracking Plan (v1.2)

This document defines the event schema, data quality rules, and metric formulas for the BidNow analytics module (`user_events` table in SQLite). It serves as the contract between synthetic data generation (`simulator.py`), SQL analytics, Metabase dashboards, and future web application instrumentation.

---

## 1. General Rules & Conventions
- **Event naming (`event_name`):** Lowercase `snake_case` using the `object_action` format (e.g., `bid_placed`).
- **Timestamps (`timestamp`):** UTC string formatted as `YYYY-MM-DD HH:MM:SS` to ensure deterministic chronological sorting and date grouping in SQLite and BI tools.
- **Null values:** Attributes not applicable to a specific event (e.g., `bid_amount` on `item_viewed`, `item_id` on `search_performed`) must be stored as `NULL`.
- **Deduplication (`event_id`):** Every event must carry a globally unique UUID v4 (`PRIMARY KEY`). Client/network retries must not insert duplicate rows.
- **Currency consistency (`currency`):** Monetary aggregations (GMV, average price, price growth) must filter by a single currency (`AMD` in baseline) or normalize amounts to a single reporting currency prior to aggregation.

---

## 2. Events Table Schema (`user_events`)

| Column | Data Type | Constraint / Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `event_id` | `TEXT` | `PRIMARY KEY` (Always) | Unique event identifier (UUID v4) | `"550e8400-e29b-41d4-a716-446655440000"` |
| `user_id` | `TEXT` | Conditional* | Authenticated user ID (`NULL` allowed for unauthenticated guests) | `"user_0012"` |
| `session_id` | `TEXT` | Always | Client or backend session identifier | `"sess_0105"` |
| `event_name` | `TEXT` | Always | Event name defined in the Event Catalog | `"bid_placed"` |
| `timestamp` | `TEXT` | Always | Event timestamp in UTC (`YYYY-MM-DD HH:MM:SS`) | `"2026-11-27 12:30:00"` |
| `item_id` | `TEXT` | Conditional | Listing / item identifier (`NULL` for standalone search) | `"item_005"` |
| `auction_id` | `TEXT` | Conditional | Auction round identifier | `"auc_005"` |
| `category` | `TEXT` | Conditional | Item category or applied search filter | `"Electronics"` |
| `bid_amount` | `REAL` | Conditional | Starting price (`item_created`), bid value (`bid_placed`), or final price (`auction_won`) | `45000.0` |
| `currency` | `TEXT` | Conditional | Currency code (`AMD`, `USD`) | `"AMD"` |
| `search_query` | `TEXT` | Conditional | Raw search string entered by the user (`search_performed`) | `"iphone 15"` |

*\*`user_id` is strictly required for `item_created`, `bid_placed`, and `auction_won`.*

---

## 3. Event Catalog

### Browse & Discovery Events (Frontend)
- **`search_performed`**
  - **Trigger:** User submits a search query or applies a category filter.
  - **Required fields:** `event_id`, `session_id`, `event_name`, `timestamp`.
  - **Optional fields:** `user_id` (nullable for guests), `category`, `search_query`.
- **`item_viewed`**
  - **Trigger:** User opens an item / auction detail page.
  - **Required fields:** `event_id`, `session_id`, `event_name`, `timestamp`, `item_id`, `auction_id`, `category`.
  - **Optional fields:** `user_id` (nullable for guests).

### Auction & Transaction Events (Backend)
- **`item_created`**
  - **Trigger:** Seller publishes a new item for auction.
  - **Required fields:** `event_id`, `user_id` (seller), `session_id`, `event_name`, `timestamp`, `item_id`, `auction_id`, `category`, `bid_amount` (starting price `> 0`), `currency`.
- **`bid_placed`**
  - **Trigger:** Backend validates and accepts a bid from an authenticated buyer (`bidder_id != seller_id`).
  - **Required fields:** `event_id`, `user_id` (bidder), `session_id`, `event_name`, `timestamp`, `item_id`, `auction_id`, `category`, `bid_amount` (`> 0`), `currency`.
- **`auction_won`**
  - **Trigger:** Auction closes with at least one valid bid and a winner is recorded.
  - **Required fields:** `event_id`, `user_id` (winner), `session_id`, `event_name`, `timestamp`, `item_id`, `auction_id`, `category`, `bid_amount` (final winning price), `currency`.

---

## 4. Metric Definitions

- **Event-Level Conversion Funnel:** Total volume progression across `item_viewed` -> `bid_placed` -> `auction_won`.
- **User-Level Conversion Rate:** Unique authenticated users progressing from viewing an item (`item_viewed` with `user_id IS NOT NULL`) to placing a bid (`bid_placed`) and winning (`auction_won`).
- **Won GMV (Gross Merchandise Value):** Sum of `bid_amount` exclusively over `auction_won` events (excluding intermediate `bid_placed` events to avoid double counting).
- **Auction Price Growth (%):** `((final_price - start_price) / start_price) * 100` for completed auctions (`auction_won` matched with `item_created` on `auction_id`).
- **Auction Liquidity (Bid Rate & Win Rate):** Share of created auctions (`item_created`) that receive at least one bid (`bid_placed`), and share that close with a winner (`auction_won`).
- **Guest Traffic Share:** Percentage of `item_viewed` and `search_performed` events where `user_id IS NULL`.

---

## 5. Data Quality & QA Invariants
Before trusting dashboard outputs, the dataset must satisfy:
1. **Uniqueness:** `COUNT(*) = COUNT(DISTINCT event_id)`.
2. **Auth enforcement:** `user_id IS NOT NULL` for 100% of `item_created`, `bid_placed`, and `auction_won` events.
3. **No self-bidding:** For any `auction_id`, the `user_id` on `bid_placed` must never equal the `user_id` on `item_created`.
4. **Monotonic bid progression:** Within each `auction_id`, `bid_amount` increases strictly over `timestamp`.
5. **Chronological consistency:** `item_created.timestamp <= bid_placed.timestamp <= auction_won.timestamp`.