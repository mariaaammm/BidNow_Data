# BidNow – Data team: price advisor + data simulator (all local)

Built from `auction.csv`: 10,681 bids in 628 real auctions of 3 items (Cartier wristwatch,
Palm Pilot M515 PDA, Xbox game console). No server or API is needed; everything runs from the command line.

## What is in this folder
| File | What it is |
|---|---|
| `price_advisor.py` | Predicts a listing's final price, a price range, a suggested opening bid and a tip |
| `benchmark_price_advisor.py`, `price_advisor_benchmark.csv` | Reproducible comparison of 6 model types x 5 feature sets under 2 validation schemes (see section 1) |
| `live_forecast.py` | Predicts the final price of a RUNNING auction from the bids so far |
| `simulator.py` | Generates fake users, listings and bids that behave like the real data (2 modes: machine learning or statistics) |
| `sim_ml.py` | The 3 machine learning models used by the simulator; `python sim_ml.py` tests them |
| `categories.json` | Optional extra categories for the simulator (ASSUMED prices, edit them) |
| `schema.sql` | DRAFT PostgreSQL schema (agree on it with the Core team) |
| `queries.sql` | 8 ready dashboard queries for Metabase |
| `load_db.sh`, `docker-compose.yml`, `db/` | Load the simulator output into PostgreSQL; optional Docker + Metabase |
| `local_prices_template.csv` | Header for your own price data (see below) |
| `test_project.py` | 35 automatic checks of everything (run `python test_project.py`, about 3 minutes) |
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
python benchmark_price_advisor.py         # re-runs the whole model comparison of section 1 (3-6 minutes)
python test_project.py                    # runs 35 automatic checks (inputs, files, reproducibility, quality)
```
`train` takes about a minute because it also tests the model twice (ordinary and strict test, see section 1). Add `--no-eval` to skip the test.

### If something fails on Windows
- `pip` is not recognized: use `py -m pip install pandas numpy scikit-learn joblib`, and write `py` instead of `python` in the commands.
- NumPy error mentioning `X86_V2` (older processor): run `py -m pip install "numpy<2.4"` and try again.

## 1. Price advisor
Input: item, auction length in days, opening bid. Output: expected final price, an 80% range,
a suggested opening bid, and a tip. The commands and the saved model file are unchanged.

**The model (changed in the November review).** Two simple models fitted on log(price) and averaged:
a shallow gradient-boosting model (depth 1, Huber loss) and a linear Ridge model in which every item has its own
slopes for duration and opening bid. Before, it was one depth-2 gradient-boosting model.

### How it was tested
- One row per **auction** (628), never per bid. Same reproducible 5-fold splits for every model, balanced by item.
  The item's "typical price" and every range are computed from the training part of each split only.
- **Ordinary test:** the usual 5-fold cross-validation. 86% of the test auctions have an (item, opening bid) pair
  that also appears in training (many sellers use 1, 9.99, ...).
- **Strict test:** auctions with the same item and opening bid always stay on the same side of a split, so the model
  never saw that opening bid for that item. It is harsher than real use but shows overfitting that the ordinary test hides.
  Real use lies between the two.
- Models were compared and chosen with split seed 42, then **confirmed with a different seed (777)**; the numbers below are from seed 777 (5-fold x 4).

| Ordinary test | Item-median baseline | Previous model | **Price advisor** |
|---|---|---|---|
| Mean error (MAE) | $142.48 | $112.20 | **$111.30** |
| RMSE (punishes big misses) | 424.5 | 322.9 | **307.1** |
| Median absolute error | $22.68 | **$20.48** | $21.42 |
| Typical % error | 10.8% | **9.6%** | 10.2% |
| 80% range contains the real price | - | 80.0% | 82.0% |

| Strict test (unseen opening bids) | Item-median baseline | Previous model | **Price advisor** |
|---|---|---|---|
| Mean error (MAE) | $145.06 | $121.06 | **$113.82** |
| RMSE | 427.1 | 349.7 | **320.5** |
| Median absolute error | $24.25 | $26.74 | **$22.41** |
| Typical % error | 11.1% | 12.9% | **10.8%** |
| 80% range contains the real price | - | 69.1% | **76.9%** |

Typical % error per item (baseline -> previous model -> price advisor):

| | Ordinary test | Strict test |
|---|---|---|
| Cartier wristwatch (136) | 58.4 -> 45.0 -> 47.8 | 58.4 -> 42.3 -> 42.6 |
| Palm Pilot (343) | 6.9 -> 5.3 -> 5.9 | 7.0 -> 8.7 -> 6.9 |
| Xbox (149) | 18.7 -> 18.7 -> 17.1 | 18.6 -> 17.6 -> 17.2 |

**Honest reading.** On opening bids the old model had already seen, the new model is *not* more accurate: its typical error is
slightly worse (10.2% vs 9.6%) and the scale-free score (mean absolute log error) is a statistical tie
(0.239 vs 0.236, difference +0.002, 95% interval -0.003 to +0.008). It is better on mean error and RMSE. On opening bids
it has not seen, it is clearly better (0.246 vs 0.268, difference -0.022, interval -0.030 to -0.014). On Palm Pilot, where prices barely vary,
the old model overfitted (8.7% typical error against 7.0% for the plain item median); the new one is about level with the median (6.9%), i.e. it adds nothing there but no longer hurts.
So the gain is **robustness**, not a higher score on familiar listings. The older figure of "9.4%" came from different splits; on the same
splits the previous model scores 9.6%. `python price_advisor.py train` prints both tests with a different seed (numbers move by about 0.5 points).

### Model comparison (best feature set of each type; mean absolute log error, lower is better)
| Model | Ordinary: MAE / typical % / score | Strict: MAE / typical % / score |
|---|---|---|
| Item-median baseline | 142.5 / 10.8% / 0.278 | 145.1 / 11.1% / 0.284 |
| Gradient boosting depth 2 (previous) | 111.5 / 9.6% / 0.236 | 119.7 / 12.2% / 0.264 |
| Random forest | 113.0 / 9.9% / 0.240 | 123.3 / 11.5% / 0.281 |
| Extra trees | 116.4 / 10.1% / 0.244 | 129.8 / 11.2% / 0.267 |
| HistGradientBoosting | 114.9 / 9.9% / 0.240 | 120.1 / 12.1% / 0.263 |
| Ridge (no per-item slopes) | 121.5 / 15.2% / 0.278 | 145.4 / 11.4% / 0.285 |
| **Price advisor (GB depth 1 + per-item Ridge)** | **111.3 / 10.2% / 0.239** | **113.8 / 10.8% / 0.246** |

No model beats the previous one on the ordinary test (all differences are inside the noise; see the confidence intervals in
`price_advisor_benchmark.csv`). Random forest, extra trees and HistGradientBoosting were no better and sometimes worse, so they were not adopted.
Plain Ridge without item slopes cannot follow Cartier and is no better than the baseline. The chosen pair is the only one that is
good in both tests. The price of that choice is the small loss in typical error on familiar opening bids and a second model to maintain.
About 20 variants were tried in total (tree depth, leaf size, trees, loss, blends); picking the best of many on 628 auctions is a risk
in itself, which is why the final choice was re-checked on splits it was not selected on. That does not remove the risk of the small sample.

### Features
- **Item** matters most, then **opening bid**. **Duration is almost irrelevant** in this data (only 3, 5 and 7-day auctions, 60% are 7-day).
  Permutation importance on unseen auctions (rise in mean absolute log error when the feature is shuffled): item 0.30, opening bid 0.08,
  duration 0.002. The advisor still takes the length as an input, because removing it changes nothing measurable and other markets
  have other lengths; do not expect it to move the price here.
- Item only: score 0.280. Item + duration: 0.281 (no gain). Item + opening bid: 0.236 (the whole gain).
- **Opening bid relative to the item's typical price** (computed on training data only) did not help the boosted trees
  (0.240 vs 0.236 ordinary; 0.284 vs 0.268 strict) and is used only inside the Ridge part, together with a "very low opening bid" flag
  (at most 1% of the typical price, so it works in any currency).
- The data holds no other facts that are known before an auction ends (no seller, description, photos or start day). The live forecast
  (section 2) uses the bids so far and is kept separate on purpose: nothing from after the start is used here.

### Price range (80%)
The range is the prediction plus the 10% and 90% quantiles of the model's **out-of-fold errors, per item** (items with fewer than 30 auctions share one pooled range).
Compared in the same nested way (the errors never include the auction being tested; seed 42, 5-fold x 6):

| Range method | Coverage, ordinary test | Coverage, strict test |
|---|---|---|
| Errors per item (before) | 79.5% | 74.1% |
| One range for all items | 80.3% (but Cartier 37%, Palm Pilot 100%) | 80.9% (same problem) |
| Per item, finite-sample correction | 80.8% | 76.1% |
| Quantile gradient boosting | 80.1% | 76.0% |
| **Per item, errors from both kinds of fold (adopted)** | 81.3% | 76.9% |
| Per item, errors from unseen-opening-bid folds only | 82.0% | 78.5% |

The adopted method (errors from ordinary folds plus errors from unseen-opening-bid folds) raised strict coverage with almost no loss of sharpness.
On the confirmation splits, old model with old ranges: 80.0% ordinary / 69.1% strict; new: 82.0% / 76.9%.
What this does and does not show: ranges are **well calibrated on familiar opening bids and still too narrow (77% instead of 80%) on
unseen ones**. Per item the width is very different: the high end is about 1.3x the low end for Palm Pilot, 2.4x for Xbox and 7x for Cartier. Coverage
for 3-day auctions is lower (about 63% to 74% depending on the method; only 148 such auctions). With 136 Cartier auctions a coverage figure has
an uncertainty of about 3 to 4 points, so do not read more into small differences.

### Limits of this data (say them out loud)
- 628 auctions, 3 items, one old US marketplace, dollars. Nothing is validated for other items, other countries or today's prices.
- Cartier (price spread 200 to 1,850) is where most of the money error is; its median error is still about 45%.
- Xbox: the model is about 1.5 points better than the item median (17.1% vs 18.7%), which is within the noise for 149 auctions. There is no solid evidence that it predicts Xbox prices well.
- The opening bid is a *hint* about value (sellers of valuable watches start higher), not a cause of the price.
- Always retrain on your own data before using the numbers for decisions. The benchmark script reruns everything in a few minutes.

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
  1. **Price:** the price advisor's model (section 1), plus a real past error of similar auctions, gives the final price.
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
- Price: see section 1.

**Real vs simulated** (`sim_out/validation.csv`, closed auctions of the 3 real items; statistical mode on the right for comparison):

| Metric | Real | ML mode | Statistical mode |
|---|---|---|---|
| Highest bid / final price at 25% of the time | 0.49 | 0.58 | 0.15 |
| ... at 50% | 0.64 | 0.67 | 0.25 |
| ... at 75% | 0.76 | 0.74 | 0.36 |
| ... at 90% | 0.86 | 0.82 | 0.52 |
| Median opening bid, 3-day / 7-day auctions | 99 / 9.99 | 99 / 9.99 | 9.99 / 10 |
| Median final price: Cartier / Palm Pilot / Xbox | 510 / 231.5 / 123.2 | 555 / 235 / 132 | 532 / 227 / 115 |
| Mean bids per auction | 17.0 | 16.3 | 17.9 |
| Share of bids in the last day / last hour | 50.3% / 16.0% | 51.2% / 16.2% | 50.2% / 15.5% |
| Median bids if opening bid < 25% / > 60% of typical | 22 / 6 | 20 / 7 | 22 / 7 |
| Share of bids from the top 10% of bidders | 38.1% | 36.8% | 36.5% |
| Bids per bidder / bidders who bid only once | 3.15 / 42.8% | 3.35 / 34.9% | 3.46 / 34.1% |

Each run uses a different "now", so the ML column moves a little from run to run. Where it is still off: the Cartier price runs about 9% and the Xbox price about 7% above the real
median (the real Cartier median is itself uncertain, about 450 to 630, because there are only 136 auctions); the link between opening bid and
final price now matches the real data for the Xbox (0.17 against 0.17) but is too strong for Cartier (0.38 against 0.28); bidder behaviour is only approximate
(the real sample is small).

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
