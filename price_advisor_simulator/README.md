# BidNow – Data team: price advisor + data simulator (all local)

Built from `auction.csv`: 10,681 bids in 628 real auctions of 3 items (Cartier wristwatch,
Palm Pilot M515 PDA, Xbox game console). No server or API is needed; everything runs from the command line.

## What is in this folder
| File | What it is |
|---|---|
| `price_advisor.py` | Predicts a listing's final price, a price range, a suggested opening bid and a tip |
| `live_forecast.py` | Predicts the final price of a RUNNING auction from the bids so far |
| `simulator.py` | Generates fake users, listings and bids that behave like the real data (2 modes: machine learning or statistics) |
| `sim_ml.py` | The 3 machine learning models used by the simulator; `python sim_ml.py` tests them |
| `categories.json` | Optional extra categories for the simulator (ASSUMED prices, edit them) |
| `schema.sql` | DRAFT PostgreSQL schema (agree on it with the Core team) |
| `queries.sql` | 8 ready dashboard queries for Metabase |
| `load_db.sh`, `docker-compose.yml`, `db/` | Load the simulator output into PostgreSQL; optional Docker + Metabase |
| `local_prices_template.csv` | Header for your own price data (see below) |
| `test_project.py` | 27 automatic checks of everything (run `python test_project.py`, about 2 minutes) |
| `.gitignore`, `.gitattributes` | Keep generated files out of git; keep Linux line endings in `.sh`/`.sql` files on Windows |
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
python simulator.py --mode stats         # same without machine learning (faster)
python sim_ml.py                          # tests the 3 simulator models on auctions they have not seen
python test_project.py                    # runs 27 automatic checks (inputs, files, reproducibility, quality)
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
CSV files saved by Excel work too (`;` separator, decimal comma like `9,99`, BOM, Windows encodings). Items with fewer than 10 rows are dropped with a warning. Aim for a few hundred rows per item. Without
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
Learns from the real data (item and length mix, opening bids, price spread, bid counts, who bids, **when** bids
arrive) and generates fake users, listings and bids. Some auctions are closed and some are still running "now"
(about 19%). Same `--seed` and `--now` give the same output. Options: `--mode`, `--listings`, `--days`, `--users`,
`--seed`, `--now`, `--out`, `--categories`, `--real-share`. Wrong values (for example `--users 5` or a broken `categories.json`) stop with a clear message; `--users` must be at least 100.

**Two modes**
- `--mode ml` (default) uses three machine learning models trained on the real data (`sim_ml.py`):
  1. **Price:** the price advisor's gradient boosting model, plus a real past error of similar auctions, gives the final price.
  2. **Number of bids:** gradient boosting with Poisson loss, with a negative-binomial spread.
  3. **Price over time:** gradient boosting *quantile* models learn how fast the highest bid climbs during an auction
     (each fake auction gets its own speed, so its price path is smooth).
  The length and the opening bid of an auction are also drawn together from a real auction, because in the real data
  short auctions start much higher (3-day median opening bid 99, 7-day 9.99).
- `--mode stats` uses only statistics and random sampling. It is faster and needs no scikit-learn, but its prices climb too slowly.

**Why the machine learning mode exists (an honest finding).** While adding it I compared the first version with the
real data on something I had not checked before: how the highest bid grows during an auction. The first version was wrong.
Halfway through an auction the real highest bid is about 64% of the final price, but the statistical mode gave about 25%,
and it gave 3-day auctions a 10 instead of a 99 opening bid. The validation now checks both. Do not use `--mode stats`
for any chart of price over time.

**Do the models learn something real?** Tested on auctions not used for training (`python sim_ml.py`):
- Price over time (lower error is better): 0.0760 with no model, 0.0525 knowing only how far the auction has gone, 0.0480 with length, item, opening bid and number of bids.
- Number of bids (lower is better): 6.54 with a constant average, 3.02 with the model.
- Price: see section 1 (typical error 9.4% against 10.9% for the baseline).

**Real vs simulated** (`sim_out/validation.csv`, closed auctions of the 3 real items; statistical mode on the right for comparison):

| Metric | Real | ML mode | Statistical mode |
|---|---|---|---|
| Highest bid / final price at 25% of the time | 0.49 | 0.58 | 0.15 |
| ... at 50% | 0.64 | 0.66 | 0.25 |
| ... at 75% | 0.76 | 0.73 | 0.36 |
| ... at 90% | 0.86 | 0.82 | 0.52 |
| Median opening bid, 3-day / 7-day auctions | 99 / 9.99 | 99 / 9.99 | 9.99 / 10 |
| Median final price: Cartier / Palm Pilot / Xbox | 510 / 231.5 / 123.2 | 570 / 234 / 129 | 532 / 227 / 115 |
| Mean bids per auction | 17.0 | 17.0 | 17.9 |
| Share of bids in the last day / last hour | 50.3% / 16.0% | 50.0% / 15.7% | 50.2% / 15.5% |
| Median bids if opening bid < 25% / > 60% of typical | 22 / 6 | 22 / 7 | 22 / 7 |
| Share of bids from the top 10% of bidders | 38.1% | 35.9% | 36.5% |
| Bids per bidder / bidders who bid only once | 3.15 / 42.8% | 3.44 / 33.9% | 3.46 / 34.1% |

Where it is still off: the Cartier price runs about 7% to 15% above the real median (the real median itself is uncertain,
about 450 to 630, because there are only 136 Cartier auctions); the link between opening bid and final price is weaker than real for the Xbox
(correlation 0.02 against 0.17); bidder behaviour is only approximate (the real sample is small).

**Other features:** 11 built-in checks run every time (bids rise, sellers never bid on their own listing, final price = highest bid, ...)
and the run stops if any fails. `python simulator.py --categories categories.json` adds Smartphone, Laptop, Bicycle, etc. Their prices in
`categories.json` are **assumed placeholders, not measured**; their listings are marked `price_basis = 'assumed'`. Replace the numbers
with real ones from your research.

**Limits to say out loud:** bid amounts only go up (the real data has proxy-bidding noise); behaviour is learned from 3 old US items;
activity ramps up during the first days of the window because the simulated site "launches" at its start; the sample data is a
snapshot, so rerun `simulator.py` to refresh "now". ML mode takes about 15 seconds to train before it generates, and about 2 minutes
for 20,000 listings.

## 4. Database and dashboards
```bash
python simulator.py
./load_db.sh "postgresql://USER:PASSWORD@localhost:5432/DBNAME"      # needs psql; creates tables and loads sim_out
psql "postgresql://USER:PASSWORD@localhost:5432/DBNAME" -f queries.sql
```
Or with Docker: `docker compose up -d`, then open http://localhost:3000 (Metabase) and add PostgreSQL
(host `db`, database/user/password `bidnow`). Paste the queries from `queries.sql` as SQL questions.

**What was tested:** `schema.sql`, `load_db.sh` and all 8 queries ran on a real PostgreSQL 18.4 with the sample
data (32,359 users, 2,000 listings, about 30,000 bids; all constraints passed). Query 7 treats the time of the newest bid as "now", so it also works on an older snapshot. **Not tested:** `docker-compose.yml`
and Metabase, because Docker is not available in my workspace. If Docker complains, check the file paths first.

## Be honest about this when presenting
- The data is a US eBay dataset with 3 items, in dollars, from an old period. Prices are not valid for an Armenian
  marketplace; the **behaviour** (bid timing, effect of opening bid) is the useful part.
- The price advisor only knows items it was trained on. For real BidNow use, collect local data and retrain.
- Everything here is a first version to agree on with Core (schema) and DevOps (database).
