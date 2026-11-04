import sqlite3
import random
from datetime import datetime, timedelta
import uuid

# Step 1: Connect to the SQLite database (file is created automatically if it doesn't exist)
DB_NAME = "auction_analytics.db"
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Step 2: Create the tracking events table
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT,
    event_name TEXT,
    timestamp TEXT,
    item_id TEXT,
    bid_amount REAL,
    category TEXT
)
""")

# Clear table records in case the script is executed multiple times
cursor.execute("DELETE FROM user_events")
conn.commit()

# Step 3: Prepare reference data for randomized generation
categories = ["Real Estate", "Vehicles", "Electronics", "Household", "Fashion"]
users = [f"user_{i:04d}" for i in range(1, 51)] # 50 virtual users
items = [f"item_{i:03d}" for i in range(1, 21)]  # 20 unique items

# Set starting point exactly 7 days prior to current timestamp
start_time = datetime.now() - timedelta(days=7)
events_to_insert = []

print("Generating synthetic dataset...")

# Step 4: Loop to generate 3,000 event records
for _ in range(3000):
    event_id = str(uuid.uuid4())                  # Unique event ID
    user_id = random.choice(users)                # Random user ID
    item_id = random.choice(items)                # Random item ID
    category = random.choice(categories)          # Random category
    
    # Select event type based on weighted probability (higher views, lower wins)
    event_name = random.choices(
        ["item_viewed", "search_performed", "bid_placed", "auction_won"],
        weights=[60, 25, 13, 2]
    )[0]
    
    # Generate bid amount only for bid placement events, otherwise None
    bid_amount = round(random.uniform(5000, 250000), 2) if event_name == "bid_placed" else None
    
    # Generate random timestamp within the 7-day window
    random_seconds = random.randint(0, 7 * 24 * 3600)
    timestamp = (start_time + timedelta(seconds=random_seconds)).strftime("%Y-%m-%d %H:%M:%S")
    
    # Append event tuple to batch list
    events_to_insert.append((
        event_id, user_id, event_name, timestamp, item_id, bid_amount, category
    ))

# Step 5: Execute batch insert of 3,000 records into SQLite database
cursor.executemany("""
INSERT INTO user_events (event_id, user_id, event_name, timestamp, item_id, bid_amount, category)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", events_to_insert)

# Commit transactions and close connection
conn.commit()
conn.close()

print(f"Success! Database file '{DB_NAME}' created with 3,000 simulated event records.")