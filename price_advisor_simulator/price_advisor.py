"""
Price advisor for BidNow (Data team) - runs locally, no server needed.

What it does
  Given an item, the auction length (days) and the opening bid, it predicts
  the expected FINAL price, an 80% range, a suggested opening bid and a short
  tip based on how similar auctions behaved in your data.

Commands (run from this folder)
  python price_advisor.py train                      # evaluate (ordinary + strict test) and train on auction.csv
  python price_advisor.py train --data auction.csv my_local_prices.csv
  python price_advisor.py predict --item "Xbox game console" --days 7 --openbid 9.99
  python price_advisor.py items                      # what the trained model knows

Data formats accepted by --data (you can pass several files at once)
  1. Bid-level eBay format (like auction.csv): auctionid, bid, bidtime, bidder,
     openbid, price, item, auction_type ...  (one row per bid)
  2. Simple local format, one row per auction, columns:
        item, duration_days, openbid, price      (optional: n_bids)
     See local_prices_template.csv. Keep one currency per item.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DEFAULT_DATA = HERE / "auction.csv"
MODEL_PATH = HERE / "model.joblib"
REPORT_PATH = HERE / "price_advisor_report.json"
REQUIRED = ["item", "duration_days", "openbid", "price"]
MIN_ITEM_ROWS = 10          # items with fewer auctions are dropped (too little to learn from)
POOLED_RANGE_MIN_ROWS = 30  # items with fewer rows borrow the pooled price range
EDGES = [-1, 0.05, 0.25, 0.6, 0.9, 1e9]
LABELS = ["under 5%", "5-25%", "25-60%", "60-90%", "over 90%"]
SUGGESTED_START_SHARE = 0.10  # suggest opening at ~10% of the expected price (see README for the evidence)


# ------------------------------------------------------------------ loading
def read_table(path) -> pd.DataFrame:
    """Read a CSV the way Excel on Windows may have saved it: BOM, ';' or tab separators, old Windows encodings."""
    last = None
    for enc in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                head = f.readline()
            sep = max([",", ";", "\t"], key=head.count)
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise ValueError(f"{path}: cannot read the file ({last})")


def load_auctions(*paths) -> pd.DataFrame:
    """Return one row per auction: item, duration_days, openbid, price, n_bids."""
    frames = []
    for p in paths or [DEFAULT_DATA]:
        df = read_table(p)
        if {"auctionid", "bid", "auction_type"} <= set(df.columns):  # eBay bid-level file
            a = df.groupby("auctionid").agg(
                item=("item", "first"), auction_type=("auction_type", "first"),
                openbid=("openbid", "first"), price=("price", "first"), n_bids=("bid", "count"))
            a["duration_days"] = a.auction_type.str.extract(r"(\d+)")[0].astype(float)
            a = a.drop(columns="auction_type").reset_index(drop=True)
        else:
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                raise ValueError(f"{p}: missing columns {missing}. Need at least {REQUIRED}")
            a = df.copy()
            if "n_bids" not in a:
                a["n_bids"] = np.nan
            a = a[REQUIRED + ["n_bids"]]
        frames.append(a)
    frames = [f for f in frames if len(f)]  # e.g. the header-only template has no rows
    if not frames:
        raise ValueError("no data rows found in the given file(s)")
    a = pd.concat(frames, ignore_index=True)
    for c in ["duration_days", "openbid", "price", "n_bids"]:
        if not pd.api.types.is_numeric_dtype(a[c]):  # Excel with a decimal comma: "9,99" -> 9.99
            a[c] = a[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        a[c] = pd.to_numeric(a[c], errors="coerce")  # text in a number column becomes "missing"
    n_all = len(a)
    a = a.dropna(subset=REQUIRED)
    a = a[(a.price > 0) & (a.openbid >= 0) & (a.duration_days > 0)]
    if len(a) < n_all:
        print(f"warning: ignored {n_all - len(a)} rows with missing, non-numeric or invalid values", file=sys.stderr)
    counts = a.item.value_counts()
    small = counts[counts < MIN_ITEM_ROWS]
    if len(small):
        print(f"warning: dropping items with < {MIN_ITEM_ROWS} auctions: {({k: int(v) for k, v in small.items()})}",
              file=sys.stderr)
        a = a[~a.item.isin(small.index)]
    return a.reset_index(drop=True)


def nice(x: float) -> float:
    """Round to 2 significant digits so suggestions look human."""
    return float(f"{x:.2g}")


LOW_OPENBID_SHARE = 0.01  # "very low opening bid" = at most 1% of the item's typical price (works in any currency)


class _RidgeInteractions:
    """Linear model on log(price) in which every item gets its own slopes for duration and opening bid.

    It extrapolates smoothly to opening bids it has never seen (trees cannot), which is what the boosted trees lack.
    The item's typical price is learned inside fit() from the training rows only, so cross-validation stays clean."""

    def __init__(self, alpha=3.0):
        self.alpha = alpha

    def _design(self, X):
        ob = np.expm1(X["log_openbid"].values)
        typ = sum(X[c].values * self.typical_[c] for c in self.item_cols_)
        rel = ob / typ
        base = pd.DataFrame({"dur": X["duration_days"].values, "lob": X["log_openbid"].values,
                             "rel": np.log1p(rel), "low": (rel <= LOW_OPENBID_SHARE).astype(float)})
        D = pd.concat([base, X[self.item_cols_].reset_index(drop=True)], axis=1)
        for c in self.item_cols_:
            for k in ("dur", "lob", "rel", "low"):
                D[f"{c}*{k}"] = D[c] * D[k]
        return D.values

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        self.item_cols_ = [c for c in X.columns if c.startswith("is_")]
        price = np.exp(np.asarray(y))
        self.typical_ = {c: float(np.median(price[X[c].values == 1])) for c in self.item_cols_}
        D = self._design(X)
        self.scaler_ = StandardScaler().fit(D)
        self.ridge_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(D), y)
        return self

    def predict(self, X):
        return self.ridge_.predict(self.scaler_.transform(self._design(X.reset_index(drop=True))))


def _legacy_model():
    """The model used before the 2026-11 review (single depth-2 boosted-tree model). Kept only as a comparison baseline."""
    return GradientBoostingRegressor(loss="huber", n_estimators=150, max_depth=2,
                                     learning_rate=0.05, subsample=0.8, random_state=0)


def _new_models():
    """Two simple models whose log-price predictions are averaged (see README: why this pair)."""
    return [
        GradientBoostingRegressor(loss="huber", n_estimators=150, max_depth=1,   # depth 1 = additive, no feature interactions
                                  learning_rate=0.05, subsample=0.8, random_state=0),
        _RidgeInteractions(alpha=3.0),
    ]


# ------------------------------------------------------------------ model
class PriceAdvisor:
    """Average of a shallow gradient-boosting model (robust Huber loss) and a per-item linear model, fitted on
    log(price), plus per-item price ranges from out-of-fold errors."""

    def _X(self, item, duration_days, openbid):
        item = pd.Series(item).reset_index(drop=True)
        dur = pd.Series(duration_days, dtype=float).reset_index(drop=True)
        ob = pd.Series(openbid, dtype=float).reset_index(drop=True)
        X = pd.DataFrame({"duration_days": dur, "log_openbid": np.log1p(ob)})
        for it in self.items:
            X["is_" + it] = (item == it).astype(float)
        return X

    def _fit_predict(self, Xtr, ytr, Xte):
        return np.mean([m.fit(Xtr, ytr).predict(Xte) for m in _new_models()], axis=0)

    def _predict_log(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def fit(self, a: pd.DataFrame):
        a = a.reset_index(drop=True)
        self.items = sorted(a.item.unique())
        self.medians = a.groupby("item").price.median().to_dict()
        self.counts = a.item.value_counts().to_dict()
        X = self._X(a.item, a.duration_days, a.openbid)
        y = np.log(a.price.values)
        self.models = [m.fit(X, y) for m in _new_models()]

        # Honest price range: out-of-fold errors, per item (items with little data share the pooled range)
        oof = np.zeros(len(a))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(a):
            oof[te] = self._fit_predict(X.iloc[tr], y[tr], X.iloc[te])
        resid = y - oof
        # Second set of errors from a STRICTER test: folds in which an item's opening bid never appears in training.
        # Prices for opening bids the model has not seen are less accurate, so ranges built from both sets stay honest
        # in both situations (validated in the README: ordinary 81% / strict 77% coverage vs 79.5% / 74% before).
        oof_strict = oof
        groups = (a.item + "|" + a.openbid.astype(str)).values
        if len(set(groups)) >= 5:
            oof_strict = np.zeros(len(a))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(a, a.item, groups):
                oof_strict[te] = self._fit_predict(X.iloc[tr], y[tr], X.iloc[te])
        resid_strict = y - oof_strict
        both = np.concatenate([resid, resid_strict])
        self.pooled_q = (float(np.quantile(both, 0.10)), float(np.quantile(both, 0.90)))
        # Estimates made WITHOUT having seen each auction (used by simulator.py --mode ml: it must not use predictions that
        # are too good because the model has already seen that auction)
        self.oof_mu = (pd.DataFrame({"item": a.item.values, "dur": a.duration_days.astype(float).values,
                                     "ob": a.openbid.astype(float).values, "mu": oof})
                       .groupby(["item", "dur", "ob"]).mu.mean().to_dict())
        self.resid_q, self.resid_by_item, self.resid_pool = {}, {}, {}
        bucket = np.searchsorted(EDGES[1:-1], (a.openbid / a.item.map(self.medians)).values, side="left")
        for it in self.items:
            r = resid[(a.item == it).values]
            self.resid_by_item[it] = r  # real out-of-fold errors, used by simulator.py --mode ml to draw realistic prices
            for b in range(len(LABELS)):  # the error is smaller when the opening bid is already high: keep them apart
                self.resid_pool[(it, b)] = resid[((a.item == it).values) & (bucket == b)]
            r_both = np.concatenate([r, resid_strict[(a.item == it).values]])
            self.resid_q[it] = ((float(np.quantile(r_both, 0.10)), float(np.quantile(r_both, 0.90)))
                                if len(r) >= POOLED_RANGE_MIN_ROWS else self.pooled_q)
        self._fit_evidence(a)
        return self

    def _fit_evidence(self, a):
        """How did the opening bid (relative to the item's typical price) relate to bids and price?"""
        self.evidence, self.low_start_bids = [], None
        if "n_bids" not in a:
            return
        d = a.dropna(subset=["n_bids"]).copy()
        if d.empty:
            return
        med = d.item.map(self.medians)
        d["ratio"], d["rel_final"] = d.openbid / med, d.price / med
        d["bucket"] = pd.cut(d.ratio, EDGES, labels=LABELS)
        for b, g in d.groupby("bucket", observed=True):
            self.evidence.append({"opening_bid_vs_typical_price": str(b), "auctions": int(len(g)),
                                  "median_final_vs_typical": round(float(g.rel_final.median()), 2),
                                  "median_bids": float(g.n_bids.median()),
                                  "share_with_single_bid_pct": round(float((g.n_bids == 1).mean() * 100), 1)})
        low = d[d.ratio < 0.25]
        if len(low) >= 20:
            self.low_start_bids = (float(low.n_bids.median()), int(len(low)), float(low.rel_final.median()))

    def predict(self, item, duration_days, openbid):
        if item not in self.items:
            raise ValueError(f"unknown item '{item}'. Known items: {self.items}. "
                             "Add local data with: python price_advisor.py train --data auction.csv yourfile.csv")
        if not duration_days > 0:
            raise ValueError("auction length (--days) must be greater than 0")
        if not openbid >= 0:
            raise ValueError("opening bid (--openbid) cannot be negative")
        X = self._X([item], [duration_days], [openbid])
        mu = float(self._predict_log(X)[0])
        q_lo, q_hi = self.resid_q[item]
        expected, low, high = np.exp(mu), np.exp(mu + q_lo), np.exp(mu + q_hi)
        low, high = min(low, expected), max(high, expected)
        ratio = openbid / self.medians[item]
        suggested = nice(SUGGESTED_START_SHARE * expected)
        return {"item": item, "duration_days": duration_days, "opening_bid": openbid,
                "expected_final_price": round(float(expected), 2),
                "range_80pct": [round(float(low), 2), round(float(high), 2)],
                "suggested_opening_bid": suggested,
                "opening_bid_vs_typical_price_pct": round(float(ratio * 100), 1),
                "tip": self._tip(ratio, suggested)}

    def sample_price(self, item, duration_days, openbid, rng):
        """One random but realistic final price: the model's estimate plus a real past error for this item."""
        mu = self.oof_mu.get((item, float(duration_days), float(openbid)))
        if mu is None:  # a combination not seen in the training data: use the model itself
            mu = float(self._predict_log(self._X([item], [duration_days], [openbid]))[0])
        b = int(np.searchsorted(EDGES[1:-1], openbid / self.medians[item], side="left"))
        r = self.resid_pool.get((item, b))
        if r is None or len(r) < 10:  # too few similar auctions: use all of this item's errors
            r = self.resid_by_item[item]
        return float(np.exp(mu + r[rng.integers(len(r))]))

    def _tip(self, ratio, suggested):
        if not self.evidence or self.low_start_bids is None:
            return "No bid-count data in the training file, so no opening-bid advice."
        row = next((e for e in self.evidence
                    if e["opening_bid_vs_typical_price"] == str(pd.cut([ratio], EDGES, labels=LABELS)[0])), None)
        low_bids, low_n, low_final = self.low_start_bids
        if ratio < 0.25:
            return (f"Low opening bid. In the training data, {low_n} auctions started under 25% of their item's "
                    f"typical price: median {low_bids:.0f} bids, final price about {low_final:.0%} of typical.")
        if row:
            return (f"This opening bid is {ratio:.0%} of the typical price. Auctions in that range got a median of "
                    f"{row['median_bids']:.0f} bids (final about {row['median_final_vs_typical']:.0%} of typical), versus "
                    f"{low_bids:.0f} bids when starting under 25%. Consider opening near {suggested:g}.")
        return "Opening bid is outside the range seen in training data."


# ------------------------------------------------------------------ evaluation
def _cv_splits(a, repeats, seed, grouped):
    """Reproducible 5-fold splits over AUCTIONS (never bid rows), balanced by item.
    grouped=True also keeps auctions with the same item and opening bid on the same side, so every test auction
    has an opening bid the model never saw for that item (a stricter test than the usual one)."""
    if grouped:
        g = (a.item + "|" + a.openbid.astype(str)).values
        for r in range(repeats):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(a, a.item, g):
                yield r, tr, te
    else:
        for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(a, a.item)):
            yield f // 5, tr, te


def evaluate(a: pd.DataFrame, repeats=2, seed=1, grouped=False):
    """Repeated 5-fold cross-validation (always tested on auctions the model has not seen)."""
    a = a.reset_index(drop=True)
    n, items = len(a), sorted(a.item.unique())
    kinds = ["base_item", "base_item_dur", "legacy_gb", "model"]
    P = {k: np.zeros((repeats, n)) for k in kinds}
    LO, HI = np.zeros((repeats, n)), np.zeros((repeats, n))
    for r, tr, te in _cv_splits(a, repeats, seed, grouped):
        trd, ted = a.iloc[tr], a.iloc[te]
        adv = PriceAdvisor().fit(trd)
        Xte = adv._X(ted.item, ted.duration_days, ted.openbid)
        mu = adv._predict_log(Xte)
        q = np.array([adv.resid_q[i] for i in ted.item])
        P["model"][r, te], LO[r, te], HI[r, te] = np.exp(mu), np.exp(mu + q[:, 0]), np.exp(mu + q[:, 1])
        old = _legacy_model().fit(adv._X(trd.item, trd.duration_days, trd.openbid), np.log(trd.price.values))
        P["legacy_gb"][r, te] = np.exp(old.predict(Xte))
        m1 = trd.groupby("item").price.median()
        m2 = trd.groupby(["item", "duration_days"]).price.median()
        P["base_item"][r, te] = ted.item.map(m1).values
        P["base_item_dur"][r, te] = [m2.get((i, d), m1[i]) for i, d in zip(ted.item, ted.duration_days)]
    y = a.price.values

    def stats(mask, k):
        yy = y[mask]
        e = [P[k][r][mask] - yy for r in range(repeats)]
        return {"MAE": round(float(np.mean([np.abs(x).mean() for x in e])), 2),
                "RMSE": round(float(np.mean([np.sqrt((x ** 2).mean()) for x in e])), 2),
                "MedAE": round(float(np.mean([np.median(np.abs(x)) for x in e])), 2),
                "median_abs_pct_err": round(float(np.mean([np.median(np.abs(x) / yy) for x in e]) * 100), 1)}

    def rng(mask):
        yy = y[mask]
        cov = np.mean([((yy >= LO[r][mask]) & (yy <= HI[r][mask])).mean() for r in range(repeats)])
        wid = np.mean([np.median(HI[r][mask] / LO[r][mask]) for r in range(repeats)])
        return {"range_coverage_pct": round(float(cov * 100), 1), "typical_high_over_low": round(float(wid), 2)}

    allm = np.ones(n, bool)
    rep = {"n_auctions": n, "cv": f"5-fold x {repeats}" + (", grouped by item+opening bid" if grouped else ""),
           "overall": {**{k: stats(allm, k) for k in kinds}, **rng(allm)}, "by_item": {}}
    for it in items:
        m = (a.item == it).values
        rep["by_item"][it] = {"auctions": int(m.sum()), **{k: stats(m, k) for k in ("base_item", "legacy_gb", "model")}, **rng(m)}
    return rep


def print_report(rep):
    o = rep["overall"]
    print(f"\nTested on unseen auctions ({rep['cv']} cross-validation, {rep['n_auctions']} auctions)")
    print(f"{'':32}{'mean err':>9}{'RMSE':>9}{'median':>9}{'typical %':>11}")
    for k, label in [("base_item", "baseline: item median"), ("base_item_dur", "baseline: item+days median"),
                     ("legacy_gb", "previous model (1 boosted GB)"), ("model", "price advisor")]:
        print(f"{label:32}{o[k]['MAE']:>9.2f}{o[k]['RMSE']:>9.2f}{o[k]['MedAE']:>9.2f}{o[k]['median_abs_pct_err']:>10.1f}%")
    print(f"80% range contains the real price {o['range_coverage_pct']}% of the time")
    print("By item (typical % error: baseline -> previous model -> price advisor):")
    for it, r in rep["by_item"].items():
        print(f"  {it} ({r['auctions']} auctions): {r['base_item']['median_abs_pct_err']}% -> "
              f"{r['legacy_gb']['median_abs_pct_err']}% -> {r['model']['median_abs_pct_err']}%, "
              f"range covers {r['range_coverage_pct']}% (high/low x{r['typical_high_over_low']})")


# ------------------------------------------------------------------ command line
def _load_model():
    sys.path.insert(0, str(HERE))
    if not MODEL_PATH.exists():
        sys.exit("No trained model yet. Run: python price_advisor.py train")
    return joblib.load(MODEL_PATH)


def main():
    ap = argparse.ArgumentParser(description="BidNow price advisor (local)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="evaluate and train, then save model.joblib")
    t.add_argument("--data", nargs="+", default=[str(DEFAULT_DATA)])
    t.add_argument("--no-eval", action="store_true", help="skip cross-validation (faster)")
    p = sub.add_parser("predict", help="price for one listing")
    p.add_argument("--item", required=True)
    p.add_argument("--days", type=float, required=True)
    p.add_argument("--openbid", type=float, required=True)
    sub.add_parser("items", help="list items the model knows")
    args = ap.parse_args()

    if args.cmd == "train":
        sys.path.insert(0, str(HERE))
        from price_advisor import PriceAdvisor as PA  # so the saved file refers to price_advisor.PriceAdvisor

        try:
            a = load_auctions(*args.data)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(f"error: {e}")
        print(f"{len(a)} auctions, items: { {k: int(v) for k, v in a.item.value_counts().items()} }")
        if not args.no_eval:
            rep = evaluate(a)
            print_report(rep)
            rep["strict_test"] = evaluate(a, grouped=True)  # opening bids the model has never seen for that item
            print_report(rep["strict_test"])
        else:
            rep = {"n_auctions": len(a)}
        adv = PA().fit(a)
        rep["opening_bid_evidence"] = adv.evidence
        REPORT_PATH.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        joblib.dump(adv, MODEL_PATH)
        print(f"\nsaved {MODEL_PATH.name} and {REPORT_PATH.name}")
    elif args.cmd == "predict":
        try:
            print(json.dumps(_load_model().predict(args.item, args.days, args.openbid), indent=2))
        except ValueError as e:
            sys.exit(f"error: {e}")
    else:
        adv = _load_model()
        for it in adv.items:
            print(f"{it}: {adv.counts[it]} auctions, typical price {adv.medians[it]:g}")


if __name__ == "__main__":
    main()
