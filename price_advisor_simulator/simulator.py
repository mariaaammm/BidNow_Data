"""
Auction data simulator for BidNow (Data team) - runs locally, writes CSV files.

It learns simple statistics from the real eBay-style auction.csv (prices, opening
bids, number of bids, WHEN bids arrive, how concentrated bidders are) and
generates fake users, listings and bids that behave similarly. Some auctions
are already closed and some are still running "now", like a live site.

Run:
    python simulator.py                                     # 2000 listings, last 30 days
    python simulator.py --listings 5000 --days 60 --seed 7
    python simulator.py --categories categories.json        # also add ASSUMED extra categories

Output (CSV, matches schema.sql): users.csv, listings.csv, bids.csv, validation.csv

Simplifications (say these when presenting):
  * Bid amounts always increase (the real data has proxy-bidding noise).
  * Real behaviour is learned from only 3 item types; extra categories (categories.json)
    borrow that behaviour and use ASSUMED prices, marked price_basis='assumed'.
  * Users only bid and sell; no messages, favorites, or views.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
# Tuned by comparing with the real data (see README): how unequal bidder activity is, and how big the user pool is.
BUCKET_EDGES = [0.05, 0.25, 0.6, 0.9]  # opening bid as a share of the item's typical price (same buckets as price_advisor.py)
ACTIVITY_POWER = 1.5
USER_POOL_FACTOR = 3.0


# ---------------------------------------------------------------- learning
def load_real(path):
    df = pd.read_csv(path)
    df["duration_days"] = df.auction_type.str.extract(r"(\d+)")[0].astype(int)
    return df


def fit_params(df):
    a = (df.groupby("auctionid")
         .agg(item=("item", "first"), dur=("duration_days", "first"), openbid=("openbid", "first"),
              price=("price", "first"), n_bids=("bid", "count"), n_bidders=("bidder", "nunique"))
         .reset_index())
    p = {"n_auctions": len(a)}
    p["items"] = {}
    for it, g in a.groupby("item"):
        p["items"][it] = {"n": len(g), "median_price": float(g.price.median()),
                          "mu": float(np.log(g.price).mean()), "sigma": float(np.log(g.price).std()),
                          "openbids": g.openbid.values, "durs": g.dur.value_counts(normalize=True).sort_index()}
    p["nbids_by_dur"] = {d: g.n_bids.values for d, g in a.groupby("dur")}
    # Real finding: a low opening bid attracts many more bids. Keep bid counts per (opening-bid bucket, duration).
    typical = a.item.map({it: v["median_price"] for it, v in p["items"].items()})
    a["bucket"] = np.searchsorted(BUCKET_EDGES, (a.openbid / typical).values, side="left")
    p["nbids_pool"] = {k: g.n_bids.values for k, g in a.groupby(["bucket", "dur"])}
    p["nbids_bucket"] = {b: g.n_bids.values for b, g in a.groupby("bucket")}
    p["nbids_dur_factor"] = {d: float(v.mean() / a.n_bids.mean()) for d, v in p["nbids_by_dur"].items()}
    p["bidder_ratio"] = (a.n_bidders / a.n_bids).values
    p["u_by_dur"] = {d: (g.bidtime / d).clip(0, 0.999999).values for d, g in df.groupby("duration_days")}
    p["ratings"] = df.bidderrate.dropna().clip(lower=0).astype(int).values
    per_bidder = df.groupby("bidder").size()
    p["bidder_counts"] = per_bidder.values
    p["bids_per_bidder"] = len(df) / df.bidder.nunique()
    p["mean_bids"] = float(a.n_bids.mean())
    p["p_single_at_open"] = float((a.price <= a.openbid).mean())
    return p


def build_profiles(p, categories, real_share):
    """One profile per real item and per assumed category: price model, opening bids, durations, weight."""
    profiles = []
    total_n = sum(v["n"] for v in p["items"].values())
    for it, v in p["items"].items():
        profiles.append({"name": it, "basis": "data", "typical": v["median_price"], "mu": v["mu"], "sigma": v["sigma"],
                         "openbids": v["openbids"], "durs": v["durs"], "weight": real_share * v["n"] / total_n})
    if categories:
        wsum = sum(c["weight"] for c in categories.values())
        for name, c in categories.items():
            like = p["items"][c["like"]]
            scale = c["median_price"] / like["median_price"]
            profiles.append({"name": name, "basis": "assumed", "typical": float(c["median_price"]),
                             "mu": float(np.log(c["median_price"])),
                             "sigma": c["price_sigma"], "openbids": like["openbids"] * scale,
                             "durs": like["durs"], "weight": (1 - real_share) * c["weight"] / wsum})
    w = np.array([pr["weight"] for pr in profiles])
    return profiles, w / w.sum()


# ---------------------------------------------------------------- helpers
def min_increment(x):
    for limit, inc in [(1, .05), (5, .25), (25, .5), (100, 1), (250, 2.5), (500, 5),
                       (1000, 10), (2500, 25), (5000, 50)]:
        if x < limit:
            return inc
    return 100


def bid_path(rng, n, openbid, price):
    """n increasing bid amounts starting near openbid and ending at ~price."""
    if n == 1:
        return np.array([round(max(openbid, price, 0.01), 2)])
    power = rng.uniform(1.2, 3.0)  # >1 => small early bids, bigger late jumps
    g = np.sort(np.clip((np.arange(1, n + 1) / n) ** power + rng.normal(0, 0.03, n), 0, 1))
    g[-1] = 1.0
    x = openbid + (price - openbid) * g
    x[0] = max(openbid, x[0], 0.01)
    for i in range(1, n):
        x[i] = max(x[i], x[i - 1] + min_increment(x[i - 1]))
    return np.round(x, 2)


def draw_unique(rng, cdf, k, banned):
    """k distinct users, drawn in proportion to their activity, never the seller."""
    chosen, seen = [], set(banned)
    while len(chosen) < k:
        for u in np.searchsorted(cdf, rng.random(max(8, 2 * k))):
            u = int(min(u, len(cdf) - 1))
            if u not in seen:
                seen.add(u)
                chosen.append(u)
                if len(chosen) == k:
                    break
    return chosen


def assign_bidders(rng, n, bidders):
    """Every chosen bidder bids at least once; nobody outbids themselves twice in a row."""
    seq = list(rng.permutation(bidders))[:n]
    while len(seq) < n:
        c = bidders[rng.integers(len(bidders))]
        if len(bidders) > 1 and c == seq[-1]:
            continue
        seq.append(c)
    return seq


# ---------------------------------------------------------------- simulate
def simulate(p, n_listings, days, seed, now, n_users=None, categories=None, real_share=1.0):
    rng = np.random.default_rng(seed)
    n_users = n_users or max(300, int(USER_POOL_FACTOR * n_listings * p["mean_bids"] / p["bids_per_bidder"]))
    uid = np.arange(1, n_users + 1)
    users = pd.DataFrame({"user_id": uid, "username": [f"user_{i:06d}" for i in uid],
                          "rating": rng.choice(p["ratings"], n_users)})
    users["created_at"] = (now - pd.Timedelta(days=days + 7)) - pd.to_timedelta(rng.uniform(1, 400, n_users), unit="D")
    activity = rng.choice(p["bidder_counts"], n_users).astype(float) ** ACTIVITY_POWER  # from real bids-per-bidder
    cdf = np.cumsum(activity) / activity.sum()
    sellers = rng.choice(n_users, size=max(30, n_users // 8), replace=False)

    profiles, pw = build_profiles(p, categories, real_share)
    listings, bids = [], []
    bid_id = 1
    for lid in range(1, n_listings + 1):
        pr = profiles[rng.choice(len(profiles), p=pw)]
        dur = int(rng.choice(pr["durs"].index.values, p=pr["durs"].values))
        price = float(np.exp(rng.normal(pr["mu"], pr["sigma"])))
        if rng.random() < p["p_single_at_open"]:  # rare auction where only the opening bid is ever placed
            openbid = float(rng.choice(pr["openbids"]))
            price, n = openbid, 1
        else:
            cands = pr["openbids"][pr["openbids"] < price * 0.98]
            openbid = float(rng.choice(cands)) if len(cands) else price * 0.3
            bucket = int(np.searchsorted(BUCKET_EDGES, openbid / pr["typical"], side="left"))
            pool = p["nbids_pool"].get((bucket, dur))
            if pool is not None and len(pool) >= 8:
                n = int(rng.choice(pool))
            else:  # too few real examples: use the bucket's bid counts, adjusted for auction length
                n = int(round(rng.choice(p["nbids_bucket"][bucket]) * p["nbids_dur_factor"][dur]))
            n = max(2, n)
        openbid = max(round(openbid, 2), 0.01)
        amounts = bid_path(rng, n, openbid, price)

        seller = int(rng.choice(sellers))
        starts = now - pd.Timedelta(days=float(rng.uniform(0, days)))
        ends = starts + pd.Timedelta(days=dur)
        u = np.sort(rng.choice(p["u_by_dur"][dur], n))
        placed = (starts + pd.to_timedelta(u * dur, unit="D")).floor("us")

        ratio = float(rng.choice(p["bidder_ratio"]))
        n_bidders = int(np.clip(round(n * ratio), 1 if n == 1 else 2, n))
        seq = assign_bidders(rng, n, draw_unique(rng, cdf, n_bidders, {seller}))
        seq = [s + 1 for s in seq]

        closed = ends <= now
        k = n if closed else int((placed <= now).sum())  # bids already visible at "now"
        for j in range(k):
            bids.append((bid_id, lid, int(seq[j]), float(amounts[j]), placed[j]))
            bid_id += 1
        listings.append((lid, seller + 1, pr["name"], pr["name"], pr["basis"], openbid, dur, starts.floor("us"),
                         ends.floor("us"), float(amounts[k - 1]) if k else openbid,
                         float(amounts[-1]) if closed else None, int(seq[-1]) if closed else None,
                         "closed" if closed else "active", k))

    listings = pd.DataFrame(listings, columns=[
        "listing_id", "seller_id", "title", "item", "price_basis", "start_price", "duration_days", "starts_at",
        "ends_at", "current_price", "final_price", "winner_id", "status", "n_bids"])
    bids = pd.DataFrame(bids, columns=["bid_id", "listing_id", "bidder_id", "amount", "placed_at"])
    listings["winner_id"] = listings.winner_id.astype("Int64")  # whole numbers (or empty), never "123.0"
    users["created_at"] = users.created_at.dt.floor("us")
    return users, listings, bids


# ---------------------------------------------------------------- checks
def check_integrity(users, listings, bids, now):
    """Rules every row of the output must satisfy. Returns a list of (rule, passed)."""
    b = bids.merge(listings, on="listing_id").sort_values(["listing_id", "placed_at", "bid_id"])
    g = b.groupby("listing_id")
    L = listings.set_index("listing_id")
    closed, active = L[L.status == "closed"], L[L.status == "active"]
    last_bid = g.amount.last()
    checks = [
        ("bid ids are unique", bids.bid_id.is_unique),
        ("every bidder and seller exists", bids.bidder_id.isin(users.user_id).all() and L.seller_id.isin(users.user_id).all()),
        ("bid amounts rise within each auction", bool((g.amount.diff().dropna() > 0).all())),
        ("bids are placed inside the auction window and not in the future",
         bool(((b.placed_at >= b.starts_at) & (b.placed_at <= b.ends_at) & (b.placed_at <= now)).all())),
        ("sellers never bid on their own listing", bool((b.seller_id != b.bidder_id).all())),
        ("nobody outbids themselves in a row", bool((g.bidder_id.apply(lambda s: (s == s.shift()).sum()) == 0).all())),
        ("n_bids equals the number of bid rows", bool((L.n_bids == g.size().reindex(L.index).fillna(0)).all())),
        ("bids are never below the start price", bool((b.amount >= b.start_price).all())),
        ("closed: ended in the past, final price = highest bid, winner = last bidder",
         bool((closed.ends_at <= now).all() and (closed.final_price == last_bid.reindex(closed.index)).all()
              and (closed.winner_id == g.bidder_id.last().reindex(closed.index)).all())),
        ("active: ends in the future, no final price or winner",
         bool((active.ends_at > now).all() and active.final_price.isna().all() and active.winner_id.isna().all())),
        ("current price = highest bid so far (or start price if none)",
         bool((L.current_price == last_bid.reindex(L.index).fillna(L.start_price)).all())),
    ]
    return checks


def real_vs_sim(real_df, p, listings, bids, users):
    """Compare the closed simulated auctions of REAL item types with the real data."""
    a = real_df.groupby("auctionid").agg(item=("item", "first"), price=("price", "first"),
                                         n_bids=("bid", "count"), n_bidders=("bidder", "nunique"))
    real_left = real_df.duration_days - real_df.bidtime
    cl = listings[(listings.status == "closed") & (listings.price_basis == "data")]
    cb = bids[bids.listing_id.isin(cl.listing_id)].merge(cl[["listing_id", "ends_at"]], on="listing_id")
    left = (cb.ends_at - cb.placed_at).dt.total_seconds() / 86400
    rows = [(f"median final price - {it}", a[a.item == it].price.median(), cl[cl.item == it].final_price.median())
            for it in sorted(a.item.unique())]
    rows += [("mean bids per auction", a.n_bids.mean(), cl.n_bids.mean()),
             ("median distinct bidders per auction", a.n_bidders.median(),
              cb.groupby("listing_id").bidder_id.nunique().median()),
             ("share of bids in the last day", (real_left <= 1).mean(), (left <= 1).mean()),
             ("share of bids in the last hour", (real_left <= 1 / 24).mean(), (left <= 1 / 24).mean())]
    typ = {it: v["median_price"] for it, v in p["items"].items()}
    ra = real_df.groupby("auctionid").agg(item=("item", "first"), openbid=("openbid", "first"), n_bids=("bid", "count"))
    ra["ratio"] = ra.openbid / ra.item.map(typ)
    cl = cl.assign(ratio=cl.start_price / cl.item.map(typ))
    rows += [("median bids when opening bid < 25% of typical price", ra[ra.ratio < .25].n_bids.median(),
              cl[cl.ratio < .25].n_bids.median()),
             ("median bids when opening bid > 60% of typical price", ra[ra.ratio > .6].n_bids.median(),
              cl[cl.ratio > .6].n_bids.median())]
    rb = real_df.groupby("bidder").size().sort_values(ascending=False)
    sb = bids.groupby("bidder_id").size().sort_values(ascending=False)
    top = lambda s: s.iloc[:max(1, int(len(s) * .1))].sum() / s.sum()
    rows += [("bids per bidder (average)", len(real_df) / real_df.bidder.nunique(), len(bids) / bids.bidder_id.nunique()),
             ("share of bids from the top 10% of bidders", top(rb), top(sb)),
             ("share of bidders who bid only once", (rb == 1).mean(), (sb == 1).mean())]
    return pd.DataFrame(rows, columns=["metric", "real", "simulated"]).round(3)


def main():
    ap = argparse.ArgumentParser(description="BidNow auction data simulator (local)")
    ap.add_argument("--data", default=str(HERE / "auction.csv"))
    ap.add_argument("--listings", type=int, default=2000)
    ap.add_argument("--days", type=int, default=30, help="listings start at random times in the last N days")
    ap.add_argument("--users", type=int, default=None, help="default: sized to match real bidder concentration")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "sim_out"))
    ap.add_argument("--now", default=None, help="ISO date treated as 'now' (default: current time, UTC)")
    ap.add_argument("--categories", default=None, help="JSON file of ASSUMED extra categories (see categories.json)")
    ap.add_argument("--real-share", type=float, default=0.3,
                    help="with --categories: share of listings from the 3 real item types")
    args = ap.parse_args()

    now = pd.Timestamp(args.now, tz="UTC") if args.now else pd.Timestamp.now(tz="UTC").floor("s")
    real = load_real(args.data)
    p = fit_params(real)
    cats = json.loads(Path(args.categories).read_text())["categories"] if args.categories else None
    users, listings, bids = simulate(p, args.listings, args.days, args.seed, now, args.users, cats,
                                     args.real_share if cats else 1.0)

    checks = check_integrity(users, listings, bids, now)
    bad = [name for name, ok in checks if not ok]
    if bad:
        raise SystemExit("integrity checks FAILED: " + "; ".join(bad))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    users.to_csv(out / "users.csv", index=False)
    listings.to_csv(out / "listings.csv", index=False)
    bids.to_csv(out / "bids.csv", index=False)
    table = real_vs_sim(real, p, listings, bids, users)
    table.to_csv(out / "validation.csv", index=False)
    st = listings.status.value_counts().to_dict()
    print(f"wrote {len(users)} users, {len(listings)} listings ({st.get('closed', 0)} closed, "
          f"{st.get('active', 0)} active), {len(bids)} bids -> {out}")
    print(f"integrity checks: all {len(checks)} passed\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
