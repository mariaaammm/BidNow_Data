# BidNow — Analytics Tracking Plan (v1.0)

This document defines the product and auction events recorded in our analytics database.
The same schema is used first by the synthetic data generator (`simulator.py`) and later when integrating real tracking from the BidNow web application.

---

## 1. General Rules & Conventions
* **Event naming (`event_name`):** Lowercase `snake_case` using the `object_action` format (e.g., `bid_placed`).
* **Timestamps (`timestamp`):** UTC string formatted as `YYYY-MM-DD HH:MM:SS`.
* **Null values:** Fields not applicable to a specific event (such as `bid_amount` during a page view) are stored as `NULL`.

---

## 2. Events Table Schema (`user_events`)

All events are stored in a single table named `user_events` with the following columns:

| Column | Data Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `event_id` | `TEXT` | Yes | Unique event identifier (UUID v4) | `"550e8400-e29b-41d4..."` |
| `user_id` | `TEXT` | No* | User identifier (`NULL` for unauthenticated guests) | `"user_0012"` |
| `session_id` | `TEXT` | Yes | User session identifier | `"sess_0105"` |
| `event_name` | `TEXT` | Yes | Name of the event (see list below) | `"bid_placed"` |
| `timestamp` | `TEXT` | Yes | Date and time when the event occurred (UTC) | `"2026-11-27 12:30:00"` |
| `item_id` | `TEXT` | No | Listing / item identifier (`NULL` for search events) | `"item_005"` |
| `auction_id` | `TEXT` | No | Specific auction round identifier | `"auc_005"` |
| `category` | `TEXT` | No | Item category | `"Electronics"` |
| `bid_amount` | `REAL` | No | Bid value or starting price | `45000.0` |
| `currency` | `TEXT` | No | Currency code (`AMD`, `USD`) | `"AMD"` |
| `search_query` | `TEXT` | No | Text entered by the user in search | `"iphone 15"` |

*\*`user_id` is strictly required for creating listings, placing bids, and winning auctions.*

---

## 3. Event Catalog

### Browse & Search Events (Frontend)

| Event Name | Trigger | Populated Fields |
| :--- | :--- | :--- |
| `search_performed` | User submits a search query or applies category filters | `event_id`, `user_id`, `session_id`, `event_name`, `timestamp`, `category`, `search_query` |
| `item_viewed` | User opens a specific item / auction page | `event_id`, `user_id`, `session_id`, `event_name`, `timestamp`, `item_id`, `auction_id`, `category` |

### Auction & Transaction Events (Backend)

| Event Name | Trigger | Populated Fields |
| :--- | :--- | :--- |
| `item_created` | Seller publishes a new item for auction | `event_id`, `user_id`, `session_id`, `event_name`, `timestamp`, `item_id`, `auction_id`, `category`, `bid_amount` (starting price), `currency` |
| `bid_placed` | Backend validates and accepts a new bid from a buyer | `event_id`, `user_id`, `session_id`, `event_name`, `timestamp`, `item_id`, `auction_id`, `category`, `bid_amount`, `currency` |
| `auction_won` | Auction timer expires and a winner is recorded | `event_id`, `user_id` (winner), `session_id`, `event_name`, `timestamp`, `item_id`, `auction_id`, `category`, `bid_amount` (final price), `currency` |

---

## 4. Core Metrics & Dashboard Views (Metabase)
1. **Conversion Funnel:** `search_performed` -> `item_viewed` -> `bid_placed` -> `auction_won`.
2. **Category Performance:** Views, bids, and completed auctions broken down by category (`Real Estate`, `Vehicles`, `Electronics`, `Household`, `Fashion`).
3. **Bidding Activity:** Average and maximum `bid_amount`, daily bid volume, and price growth from `item_created` to `auction_won`.