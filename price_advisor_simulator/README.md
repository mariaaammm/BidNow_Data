# BidNow – Data team: price advisor + data simulator (all local)

Built from `auction.csv`: 10,681 bids in 628 real auctions of 3 items (Cartier wristwatch,
Palm Pilot M515 PDA, Xbox game console). No server or API is needed; everything runs from the command line.

## What is in this folder
| File | What it is |
|---|---|
| `price_advisor.py` | Predicts a listing's final price, a price range, a suggested opening bid and a tip |
| `live_forecast.py` | Predicts the final price of a RUNNING auction from the bids so far |
| `simulator.py` | Generates fake users, listings and bids that behave like the real data |
| `categories.json` | Optional extra categories for the simulator (ASSUMED prices, edit them) |
| `schema.sql` | DRAFT PostgreSQL schema (agree on it with the Core team) |
| `queries.sql` | 8 ready dashboard queries for Metabase |
| `load_db.sh`, `docker-compose.yml`, `db/` | Load the simulator output into PostgreSQL; optional Docker + Metabase |
| `local_prices_template.csv` | Header for your own price data (see below) |
| `.gitignore` | Keeps generated files (trained models, big CSVs) out of git |
| `sim_out/` | Created by `simulator.py`: `users.csv`, `listings.csv`, `bids.csv`, `validation.csv` (only `validation.csv` is kept in git) |
| `price_advisor_report.json`, `live_forecast_report.json` | Saved evaluation results |

## Quick start
```bash
pip install pandas numpy scikit-learn joblib

python price_advisor.py train
python price_advisor.py predict --item "Xbox game console" --days 7 --openbid 9.99
python live_forecast.py train
python live_forecast.py predict --item "Palm Pilot M515 PDA" --days 7 --openbid 1 --elapsed 5 --high 180 --bids 9
python simulator.py                       # writes sim_out/*.csv, checks them, compares with the real data
```
`train` takes about a minute because it also tests the model. Add `--no-eval` to skip the test.

### If something fails on Windows
- `pip` is not recognized: use `py -m pip install pandas numpy scikit-learn joblib`, and write `py` instead of `python` in the commands.
- NumPy error mentioning `X86_V2` (older processor): run `py -m pip install "numpy<2.4"` and try again.

## 1. Price advisor
Input: item, auction length in days, opening bid. Output: expected final price, an 80% range,
a suggested opening bid, and a tip.

**How good is it?** Tested only on auctions the model had not seen (5-fold cross-validation, repeated twice):

| | Baseline: median price of the item | Price advisor |
|---|---|---|
| Mean error, all items | $142.92 | **$111.84** |
| Typical error, all items | 10.9% | **9.4%** |
| Cartier wristwatch | 58.1% | **43.9%** |
| Palm Pilot | 6.6% | **5.4%** |
| Xbox | 18.5% | **18.0%** |

The 80% range contained the real price 79.9% of the time (Cartier 80.5%, Palm Pilot 79.6%, Xbox 79.9%),
so the range is honest. It is very different per item: the high end is about 1.3x the low end for
Palm Pilot, 2.4x for Xbox and 6.4x for Cartier.

**Where the improvement comes from.** A model with only item and length does no better than the
baseline. The opening bid is what helps, mostly for Cartier: a high opening bid hints that this
is a more valuable watch. So it is a useful hint, not proof that the opening bid causes the price.

**What the data says about opening bids** (from `price_advisor_report.json`; opening bid compared
with the item's typical price):

| Opening bid | Auctions | Median bids | Final price vs typical | Only one bid |
|---|---|---|---|---|
| under 5% | 291 | 23 | 1.00 | 0.3% |
| 5-25% | 73 | 19 | 0.94 | 0% |
| 25-60% | 81 | 13 | 0.98 | 0% |
| 60-90% | 111 | 7 | 0.93 | 3.6% |
| over 90% | 72 | 3.5 | 1.13 | 22.2% |

A low opening bid attracted about 3 times more bids and the final price was not lower. That is why
the advisor suggests opening at about 10% of the expected price. Careful: this is observed data, not
an experiment. The "over 90%" row is high mostly because sellers of expensive items start high.

**Using your own data (for example list.am prices).** One row per auction, one currency per item:
```
item,duration_days,openbid,price,n_bids        (n_bids is optional; see local_prices_template.csv)
```
```bash
python price_advisor.py train --data auction.csv my_prices.csv
python price_advisor.py items
```
Items with fewer than 10 rows are dropped with a warning. Aim for a few hundred rows per item. Without
`n_bids`, the opening-bid tip uses only the rows that have it.

## 2. Live forecast (for running auctions)
Uses the highest bid so far, the number of bids and how much time has passed. Median error of the
final-price guess, on auctions not used for training:

| Time passed | Guess = current high bid | Guess = item median | Live forecast |
|---|---|---|---|
| 10% | 59.3% | 11.0% | 10.7% |
| 25% | 51.0% | 11.0% | 9.8% |
| 50% | 36.1% | 11.0% | 9.1% |
| 75% | 23.9% | 11.0% | 8.7% |
| 90% | 13.9% | 11.0% | 7.9% |
| 97% | 5.4% | 11.0% | 5.4% |

It helps most in the middle of an auction. At the very end the current bid is already the best guess,
and the code automatically uses it. It needs bid-level data, so for now it only trains on `auction.csv`
(later it could train on BidNow's own `bids` table).

## 3. Data simulator
Learns from the real data: item and length mix, opening bids, price spread per item, bid counts, who
bids, and **when** bids arrive. Features:
- **Live site look:** some auctions are closed and some are still running "now" (about 19%).
- **Opening bid matters, like in the real data:** low opening bids attract many more bids.
- **Concentrated bidders:** a few bidders place many bids, many bid only once.
- **Built-in checks:** 11 rules are verified every run (bids rise, sellers never bid on their own listing,
  final price = highest bid, ...). The run stops if any fails. These checks found and fixed a real bug during development.
- **Same `--seed` and `--now` = same output.** Options: `--listings`, `--days`, `--users`, `--seed`, `--now`, `--out`.

**Real vs simulated** (`sim_out/validation.csv`, closed auctions of the 3 real items):

| Metric | Real | Simulated |
|---|---|---|
| Median final price: Cartier / Palm Pilot / Xbox | 510 / 231.5 / 123.2 | 532 / 227 / 115 |
| Mean bids per auction | 17.0 | 17.9 |
| Share of bids in the last day / last hour | 50.3% / 16.0% | 50.2% / 15.5% |
| Median bids if opening bid < 25% / > 60% of typical | 22 / 6 | 22 / 7 |
| Share of bids from the top 10% of bidders | 38.1% | 36.5% |
| Bids per bidder / bidders who bid only once | 3.15 / 42.8% | 3.46 / 34.1% |

Prices and timing match well. Bidder behaviour is only approximate: the real sample is small
(628 auctions), so a perfect match is not possible.

**More categories.** `python simulator.py --categories categories.json` adds Smartphone, Laptop, Bicycle, etc.
Their prices in `categories.json` are **assumed placeholders, not measured**. Their listings are marked
`price_basis = 'assumed'`, so dashboards can separate them. Replace the numbers with real ones from your research.

**Limits to say out loud:** bid amounts only go up (the real data has proxy-bidding noise); behaviour is learned
from 3 old US items; activity ramps up during the first days of the window because the simulated site "launches"
at its start; the sample data is a snapshot, so rerun `simulator.py` to refresh "now".

## 4. Database and dashboards
```bash
python simulator.py
./load_db.sh "postgresql://USER:PASSWORD@localhost:5432/DBNAME"      # needs psql; creates tables and loads sim_out
psql "postgresql://USER:PASSWORD@localhost:5432/DBNAME" -f queries.sql
```
Or with Docker: `docker compose up -d`, then open http://localhost:3000 (Metabase) and add PostgreSQL
(host `db`, database/user/password `bidnow`). Paste the queries from `queries.sql` as SQL questions.

**What was tested:** `schema.sql`, `load_db.sh` and all 8 queries ran on a real PostgreSQL 18.4 with the sample
data (32,359 users, 2,000 listings, 31,338 bids; all constraints passed). **Not tested:** `docker-compose.yml`
and Metabase, because Docker is not available in my workspace. If Docker complains, check the file paths first.

## Be honest about this when presenting
- The data is a US eBay dataset with 3 items, in dollars, from an old period. Prices are not valid for an Armenian
  marketplace; the **behaviour** (bid timing, effect of opening bid) is the useful part.
- The price advisor only knows items it was trained on. For real BidNow use, collect local data and retrain.
- Everything here is a first version to agree on with Core (schema) and DevOps (database).
