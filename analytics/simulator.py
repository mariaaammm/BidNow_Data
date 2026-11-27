import sqlite3
import random
from datetime import datetime, timedelta
import uuid

# Configuration
DB_NAME = "auction_analytics.db"
NUM_USERS = 60
NUM_ITEMS = 30
DAYS_BACK = 7
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Step 1: Connect to SQLite database
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Step 2: Recreate table matching TRACKING_PLAN.md (v1.0)
cursor.execute("DROP TABLE IF EXISTS user_events")
cursor.execute("""
CREATE TABLE user_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    event_name TEXT,
    timestamp TEXT,
    item_id TEXT,
    auction_id TEXT,
    category TEXT,
    bid_amount REAL,
    currency TEXT,
    search_query TEXT
)
""")
conn.commit()

# Step 3: Reference data
categories_config = {
    "Real Estate": {"min_price": 20000000, "max_price": 80000000, "queries": ["apartment yerevan", "house center", "2 room flat"]},
    "Vehicles": {"min_price": 2500000, "max_price": 15000000, "queries": ["toyota camry", "bmw x5", "hybrid car"]},
    "Electronics": {"min_price": 30000, "max_price": 600000, "queries": ["iphone 15", "macbook pro", "ps5"]},
    "Household": {"min_price": 10000, "max_price": 250000, "queries": ["sofa", "coffee machine", "washing machine"]},
    "Fashion": {"min_price": 5000, "max_price": 120000, "queries": ["sneakers", "winter jacket", "watch"]}
}

categories = list(categories_config.keys())
users = [f"user_{i:04d}" for i in range(1, NUM_USERS + 1)]
base_start = datetime.now() - timedelta(days=DAYS_BACK)

events_to_insert = []

def make_session_id():
    return f"sess_{uuid.uuid4().hex[:8]}"

def fmt_time(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

print("Generating realistic auction simulation dataset...")

# Step 4: Simulate item lifecycles (creation -> views -> bids -> win)
for i in range(1, NUM_ITEMS + 1):
    item_id = f"item_{i:03d}"
    auction_id = f"auc_{i:03d}"
    category = random.choice(categories)
    cfg = categories_config[category]
    
    seller_id = random.choice(users)
    starting_price = round(random.uniform(cfg["min_price"], cfg["max_price"]), -2)
    
    # Item created within the first 5 days so auctions have time to progress
    created_at = base_start + timedelta(seconds=random.randint(0, 5 * 24 * 3600))
    
    events_to_insert.append((
        str(uuid.uuid4()), seller_id, make_session_id(), "item_created",
        fmt_time(created_at), item_id, auction_id, category,
        starting_price, "AMD", None
    ))
    
    # Generate views for this item (20 to 80 views)
    num_views = random.randint(20, 80)
    for _ in range(num_views):
        viewer_id = random.choice(users) if random.random() > 0.15 else None
        view_time = created_at + timedelta(seconds=random.randint(60, 48 * 3600))
        if view_time > datetime.now():
            view_time = datetime.now() - timedelta(minutes=random.randint(1, 30))
            
        events_to_insert.append((
            str(uuid.uuid4()), viewer_id, make_session_id(), "item_viewed",
            fmt_time(view_time), item_id, auction_id, category,
            None, None, None
        ))
    
    # Generate progressive bids (0 to 15 bids per auction)
    num_bids = random.choices([0, random.randint(1, 5), random.randint(6, 15)], weights=[15, 55, 30])[0]
    current_price = starting_price
    current_time = created_at
    last_bidder = None
    
    potential_bidders = [u for u in users if u != seller_id]
    
    for _ in range(num_bids):
        bidder_id = random.choice(potential_bidders)
        step_increase = round(current_price * random.uniform(0.03, 0.12), -2)
        if step_increase < 500:
            step_increase = 500.0
        current_price = round(current_price + step_increase, 2)
        
        current_time = current_time + timedelta(minutes=random.randint(5, 180))
        if current_time > datetime.now():
            break
            
        last_bidder = bidder_id
        events_to_insert.append((
            str(uuid.uuid4()), bidder_id, make_session_id(), "bid_placed",
            fmt_time(current_time), item_id, auction_id, category,
            current_price, "AMD", None
        ))
    
    # Close ~70% of auctions with at least 1 bid as won
    if last_bidder is not None and random.random() < 0.7:
        win_time = min(current_time + timedelta(minutes=random.randint(10, 120)), datetime.now())
        events_to_insert.append((
            str(uuid.uuid4()), last_bidder, make_session_id(), "auction_won",
            fmt_time(win_time), item_id, auction_id, category,
            current_price, "AMD", None
        ))

# Step 5: Simulate standalone search events
for _ in range(900):
    searcher_id = random.choice(users) if random.random() > 0.2 else None
    cat = random.choice(categories) if random.random() > 0.25 else None
    query = random.choice(categories_config[cat]["queries"]) if cat and random.random() > 0.2 else "auction"
    search_time = base_start + timedelta(seconds=random.randint(0, DAYS_BACK * 24 * 3600))
    
    events_to_insert.append((
        str(uuid.uuid4()), searcher_id, make_session_id(), "search_performed",
        fmt_time(search_time), None, None, cat,
        None, None, query
    ))

# Sort chronologically before inserting
events_to_insert.sort(key=lambda row: row[4])

# Step 6: Batch insert
cursor.executemany("""
INSERT INTO user_events (
    event_id, user_id, session_id, event_name, timestamp,
    item_id, auction_id, category, bid_amount, currency, search_query
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", events_to_insert)

conn.commit()
conn.close()

print(f"Success! '{DB_NAME}' generated with {len(events_to_insert)} logically consistent events.")