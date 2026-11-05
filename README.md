# BidNow Data
## Scope

| # | Area | Where |
|---|---|---|
| 1 | **Market research**: what sells and for how much (Kaggle Online Auctions dataset) | `notebooks/eda.ipynb` |
| 2 | **Price advisor**: expected final price and suggested opening bid | branch `price-advisor/data-simulator` |
| 3 | **Category map**: listing title category (keyword based) | `category_map/` |
| 4 | **Data simulator**: synthetic users and listings and bids | branch `price-advisor/data-simulator` |
| 5 | **Tracking plan**: analytics events for product features | branch `feature/analytics-baseline` |
| 6 | **Dashboards (Metabase)**: simulator data first and real data later | branch `feature/analytics-baseline` |

## Setup

Requires Python 3.10 or newer.

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the script run `Set-ExecutionPolicy -Scope Process RemoteSigned` first.

**macOS and Linux**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the tests

From the project root with the virtual environment active:

```bash
pytest -q
```

The suite covers the category classifier (53 cases).

## Market research (EDA)

`notebooks/eda.ipynb` analyses the Online Auctions dataset (10,681 bids in 628 eBay auctions of 3 items). It covers data quality and prices and opening bids and auction length and competition.

The dataset is stored in `data/auction.csv`. Source: [Kaggle](https://www.kaggle.com/datasets/onlineauctions/online-auctions-dataset) Open the notebook from the `notebooks/` folder and choose **Run All**.

Key points:

- The data is stored per bid. Market questions use a table with one row per auction.
- Compare prices inside one item only. The three items have very different price levels.
- Sellers choose the opening bid and the auction length themselves. Results show links and not causes.

## Category map

```python
from category_preprocessing import classify

classify('iphone 13')            # 'Electronics > Phones and Tablets'
classify('3 room apartment')     # 'Real Estate'
classify('qwerty')               # 'Other'
```

- `categories.json`: list.am category tree (3 levels)
- `category_keywords.json`: keywords per sub category (lowercase and singular)
- `category_preprocessing.py`: `clean()` and `classify()`

Every keyword found in a title gives points to its sub category and a longer phrase gives more points. A tie inside one main category returns only the main category.

To change categories or keywords edit the JSON files and run `pytest -q`.

Known limits: keyword matching only (accuracy on real titles is not measured) and no Armenian keywords yet.

## Price advisor, data simulator

Available on branch `price-advisor/data-simulator`.

```bash
git checkout price-advisor/data-simulator
cd price_advisor_simulator
pip install pandas numpy scikit-learn joblib
python simulator.py
python price_advisor.py train
```

The simulator writes users and listings and bids to `sim_out/`. `queries.sql` has the dashboard queries for Metabase and `docker-compose.yml` starts PostgreSQL with Metabase. See the folder README for results and limits.

## Tracking plan

Available on branch `feature/analytics-baseline`. `simulator.py` creates a SQLite table `user_events` with 3,000 synthetic events of four types: `item_viewed`, `search_performed`, `bid_placed`, `auction_won`. A written tracking plan (event names, properties and owners) is still to be added.

## Limits

- The dataset is old US eBay data in dollars for 3 products. Prices do not apply to an Armenian market. The behaviour (bid timing and effect of the opening bid) is the useful part.
- Simulator data is synthetic. Dashboards must be rebuilt on real BidNow data later.

## Repository layout (`dev`)

```
├── category_map/
│   ├── categories.json
│   ├── category_keywords.json
│   ├── category_preprocessing.py
│   └── test_category_preprocessing.py
├── data/auction.csv
├── notebooks/eda.ipynb
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
