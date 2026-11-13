"""
Live forecast for a RUNNING auction (local, no server).

price_advisor.py answers "what will this listing sell for?" before it starts.
This one answers it while the auction is running, using what has happened so far:
the highest bid now, how many bids, and how much of the auction time has passed.

Needs bid-level data (like auction.csv: auctionid, bid, bidtime, openbid, price, item, auction_type).

  python live_forecast.py train
  python live_forecast.py predict --item "Palm Pilot M515 PDA" --days 7 --openbid 1 \
         --elapsed 5 --high 180 --bids 9
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent
DEFAULT_DATA = HERE / "auction.csv"
MODEL_PATH = HERE / "live_model.joblib"
REPORT_PATH = HERE / "live_forecast_report.json"
FRACS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.97]  # moments in the auction's life used for training/testing


def build_snapshots(path) -> pd.DataFrame:
    """For each auction, look at it at several moments: what did the bids look like then?"""
    df = pd.read_csv(path)
    df["dur"] = df.auction_type.str.extract(r"(\d+)")[0].astype(float)
    rows = []
    for aid, g in df.groupby("auctionid"):
        dur, ob, final = float(g.dur.iloc[0]), float(g.openbid.iloc[0]), float(g.price.iloc[0])
        t, b = g.bidtime.values, g.bid.values
        for f in FRACS:
            m = t <= f * dur
            rows.append({"auctionid": aid, "item": g["item"].iloc[0], "duration_days": dur, "openbid": ob,
                         "frac": f, "high": max(float(b[m].max()), ob) if m.any() else ob,
                         "n_bids_so_far": int(m.sum()), "final": final})
    return pd.DataFrame(rows)


class LiveForecaster:
    def _X(self, d: pd.DataFrame):
        med = d["item"].map(self.medians).astype(float)
        X = pd.DataFrame({"frac": d.frac.values, "duration_days": d.duration_days.values,
                          "log_openbid": np.log1p(d.openbid.values), "log_high": np.log(d.high.values),
                          "log_n_bids": np.log1p(d.n_bids_so_far.values),
                          "high_vs_typical": np.log(d.high.values / med.values)})
        for it in self.items:
            X["is_" + it] = (d["item"].values == it).astype(float)
        return X

    @staticmethod
    def _model():
        return GradientBoostingRegressor(loss="huber", n_estimators=150, max_depth=3,
                                         learning_rate=0.05, subsample=0.8, random_state=0)

    def fit(self, s: pd.DataFrame, final_by_auction: pd.DataFrame):
        s = s.reset_index(drop=True)
        self.items = sorted(s["item"].unique())
        self.medians = final_by_auction.groupby("item").final.median().to_dict()
        X, y = self._X(s), np.log(s.final.values / s.high.values)
        self.model = self._model().fit(X, y)
        oof = np.zeros(len(s))
        for tr, te in GroupKFold(5).split(X, y, s.auctionid):
            oof[te] = self._model().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        # At each moment, use the model only if it beat "final = current high bid" on unseen auctions.
        self.use_model, self.resid_q = {}, {}
        for f in FRACS:
            m = (s.frac == f).values
            final_rel = np.exp(y[m])  # final / high
            err_model = np.median(np.abs(final_rel - np.exp(oof[m])) / final_rel)
            err_high = np.median(np.abs(final_rel - 1.0) / final_rel)
            self.use_model[f] = bool(err_model < err_high)
            resid = y[m] - (oof[m] if self.use_model[f] else 0.0)
            self.resid_q[f] = (float(np.quantile(resid, .1)), float(np.quantile(resid, .9)))
        return self

    def _log_ratio(self, X, fracs):
        """Predicted log(final/high); 0 where the plain current-high-bid guess is better."""
        mu = self.model.predict(X)
        keep = np.array([self.use_model[min(FRACS, key=lambda g: abs(g - f))] for f in fracs])
        return np.where(keep, mu, 0.0)

    def predict(self, item, duration_days, openbid, elapsed_days, current_high, n_bids):
        if item not in self.items:
            raise ValueError(f"unknown item '{item}'. Known items: {self.items}")
        frac = float(np.clip(elapsed_days / duration_days, 0.0, 0.995))
        row = pd.DataFrame([{"item": item, "duration_days": duration_days, "openbid": openbid, "frac": frac,
                             "high": max(current_high, openbid), "n_bids_so_far": n_bids}])
        mu = float(self._log_ratio(self._X(row), [frac])[0])
        nearest = min(FRACS, key=lambda f: abs(f - frac))
        lo, hi = self.resid_q[nearest]
        base = float(row.high.iloc[0])
        return {"item": item, "share_of_time_elapsed_pct": round(frac * 100, 1), "current_high_bid": base,
                "expected_final_price": round(base * np.exp(mu), 2),
                "range_80pct": [round(base * np.exp(mu + lo), 2), round(base * np.exp(mu + hi), 2)]}


def evaluate(s: pd.DataFrame, auctions: pd.DataFrame):
    """Out-of-fold errors by moment in the auction's life, versus two simple guesses."""
    s = s.reset_index(drop=True)
    pred = np.zeros(len(s))
    for tr, te in GroupKFold(5).split(s, groups=s.auctionid):
        lf = LiveForecaster().fit(s.iloc[tr], auctions[auctions.auctionid.isin(s.auctionid.iloc[tr])])
        pred[te] = s.high.values[te] * np.exp(lf._log_ratio(lf._X(s.iloc[te]), s.frac.values[te]))
    s["pred"] = pred
    med = s["item"].map(auctions.groupby("item").final.median())
    s["guess_high"], s["guess_median"] = s.high, med
    rows = []
    for f, g in s.groupby("frac"):
        pe = lambda c: round(float((np.abs(g[c] - g.final) / g.final).median() * 100), 1)
        rows.append({"share_of_time_elapsed_pct": int(round(f * 100)), "auctions": len(g),
                     "guess: current high bid": pe("guess_high"), "guess: item median": pe("guess_median"),
                     "live forecast": pe("pred")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Live final-price forecast (local)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default=str(DEFAULT_DATA))
    p = sub.add_parser("predict")
    p.add_argument("--item", required=True)
    p.add_argument("--days", type=float, required=True, help="auction length in days")
    p.add_argument("--openbid", type=float, required=True)
    p.add_argument("--elapsed", type=float, required=True, help="days since the auction started")
    p.add_argument("--high", type=float, required=True, help="highest bid so far")
    p.add_argument("--bids", type=int, required=True, help="number of bids so far")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))

    if args.cmd == "train":
        from live_forecast import LiveForecaster as LF  # so the saved file refers to live_forecast.LiveForecaster
        s = build_snapshots(args.data)
        auctions = s.groupby("auctionid").agg(item=("item", "first"), final=("final", "first")).reset_index()
        table = evaluate(s, auctions)
        print("Median % error of the final-price guess, on auctions not used for training:\n")
        print(table.to_string(index=False))
        lf = LF().fit(s, auctions)
        joblib.dump(lf, MODEL_PATH)
        REPORT_PATH.write_text(json.dumps(table.to_dict("records"), indent=2))
        print(f"\nsaved {MODEL_PATH.name} and {REPORT_PATH.name}")
    else:
        if not MODEL_PATH.exists():
            sys.exit("No trained model yet. Run: python live_forecast.py train")
        try:
            out = joblib.load(MODEL_PATH).predict(args.item, args.days, args.openbid, args.elapsed, args.high, args.bids)
        except ValueError as e:
            sys.exit(f"error: {e}")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
