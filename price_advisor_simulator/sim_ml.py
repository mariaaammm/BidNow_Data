"""
Learned (machine learning) parts of the simulator: used by `python simulator.py --mode ml`.

Three models, all learned from the real auction.csv:
  1. PRICE        the price advisor's gradient boosting model + a real past error  -> final price
  2. BID COUNT    gradient boosting with Poisson loss + negative-binomial spread     -> how many bids
  3. PRICE PATH   gradient boosting QUANTILE models: how fast the highest bid climbs  -> bid amounts over time

Why the third one matters: the statistical simulator made prices climb much too slowly
(halfway through an auction the real highest bid is ~64% of the final price; the old
simulator gave ~25%). The path model learns this curve from the 10,681 real bids.

Test how well each model works on auctions it has not seen:
    python sim_ml.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from price_advisor import PriceAdvisor, load_auctions  # noqa: E402

LEVELS = np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98])  # quantile levels the path model learns
MAX_BIDS = 90  # a little above the most bids seen in the real data (75)


def _dummies(names, items):
    names = np.asarray(names)
    return np.column_stack([(names == it).astype(float) for it in items])


def _interp_levels(P, q):
    """Pick, for each row, the value at quantile q by interpolating between the learned levels."""
    q = np.clip(q, LEVELS[0], LEVELS[-1])
    j = np.clip(np.searchsorted(LEVELS, q, side="right") - 1, 0, len(LEVELS) - 2)
    t = (q - LEVELS[j]) / (LEVELS[j + 1] - LEVELS[j])
    r = np.arange(len(q))
    return P[r, j] * (1 - t) + P[r, j + 1] * t


def _quantile_gb(alpha):
    return GradientBoostingRegressor(loss="quantile", alpha=float(alpha), n_estimators=80, max_depth=3,
                                     learning_rate=0.1, subsample=0.8, min_samples_leaf=40, random_state=0)


class LearnedModels:
    def __init__(self, real_df: pd.DataFrame, typical: dict, data_path):
        self.items = sorted(typical)
        self.typical = typical
        self.auctions = (real_df.groupby("auctionid")
                         .agg(item=("item", "first"), dur=("duration_days", "first"), openbid=("openbid", "first"),
                              price=("price", "first"), n=("bid", "count")).reset_index())
        self.advisor = PriceAdvisor().fit(load_auctions(data_path))      # 1. price
        self._fit_counts()                                               # 2. number of bids
        self._fit_path(real_df)                                          # 3. price over time

    # ---------------------------------------------------------------- 2. number of bids
    def _count_X(self, items, dur, openbid, price, typical):
        typical = np.asarray(typical, dtype=float)
        return np.column_stack([np.asarray(dur, float), np.log1p(np.asarray(openbid, float) / typical),
                                np.log(np.asarray(price, float) / typical), _dummies(items, self.items)])

    @staticmethod
    def _count_model():
        return HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05, max_iter=150,
                                             min_samples_leaf=15, random_state=0)

    def _count_training(self):
        a = self.auctions[self.auctions.price > self.auctions.openbid]  # auctions with real competition
        X = self._count_X(a.item, a.dur, a.openbid, a.price, a["item"].map(self.typical))
        return a, X, a.n.values.astype(float)

    def _fit_counts(self):
        a, X, y = self._count_training()
        self.count_model = self._count_model().fit(X, y)
        mu = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            mu[te] = self._count_model().fit(X[tr], y[tr]).predict(X[te])
        # negative binomial: variance = mu + mu^2 / k  ->  estimate k from unseen-data errors
        self.count_k = float(np.clip(np.sum(mu ** 2) / max(np.sum((y - mu) ** 2 - mu), 1e-6), 1.0, 60.0))

    def sample_count(self, rng, item, dur, openbid, price, typical):
        mu = float(self.count_model.predict(self._count_X([item], [dur], [openbid], [price], [typical]))[0])
        n = int(rng.negative_binomial(self.count_k, self.count_k / (self.count_k + mu)))
        return int(np.clip(n, 2, MAX_BIDS))

    # ---------------------------------------------------------------- 3. price over time
    def _path_X(self, u, dur, rel_open, log_n, items):
        return np.column_stack([u, dur, rel_open, log_n, _dummies(items, self.items)])

    def _path_training(self, real_df):
        d = real_df.sort_values(["auctionid", "bidtime"]).copy()
        d["run_max"] = d.groupby("auctionid").bid.cummax()  # highest bid so far
        d["span"] = d.price - d.openbid
        d = d[d.span > 0]
        d["g"] = ((d.run_max - d.openbid) / d.span).clip(0, 1)  # 0 = still at the opening bid, 1 = at the final price
        d["u"] = (d.bidtime / d.duration_days).clip(0, 1)
        d["n"] = d.groupby("auctionid").bid.transform("size")
        d["rel_open"] = np.log1p(d.openbid / d["item"].map(self.typical))
        X = self._path_X(d.u, d.duration_days, d.rel_open, np.log(d.n), d["item"])
        return d, X, d.g.values

    def _fit_path(self, real_df):
        _, X, g = self._path_training(real_df)
        self.path_models = [_quantile_gb(a).fit(X, g) for a in LEVELS]

    def _predict_levels(self, X):
        P = np.clip(np.column_stack([m.predict(X) for m in self.path_models]), 0, 1)
        return np.maximum.accumulate(P, axis=1)  # quantile curves must not cross

    def path_progress(self, specs, rng):
        """specs: dicts with item, dur, rel_open, u (sorted bid times as share of the auction).
        Returns for each auction the share of the climb (0..1) reached at each bid: rising, ending at 1."""
        lens = [len(s["u"]) for s in specs]
        X = self._path_X(np.concatenate([s["u"] for s in specs]),
                         np.repeat([s["dur"] for s in specs], lens),
                         np.repeat([s["rel_open"] for s in specs], lens),
                         np.repeat(np.log(lens), lens),
                         np.repeat([s["item"] for s in specs], lens))
        # one random "how fast does THIS auction climb" level per auction keeps each price path smooth
        g = _interp_levels(self._predict_levels(X), np.repeat(rng.random(len(specs)), lens))
        out = []
        for part in np.split(g, np.cumsum(lens)[:-1]):
            part = np.maximum.accumulate(part)
            part[-1] = 1.0
            out.append(part)
        return out


# ---------------------------------------------------------------- how well do the models generalise?
def _pinball(y, pred, alpha):
    d = y - pred
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def evaluate(real_df, data_path):
    typical = real_df.groupby("item").price.median().to_dict()
    lm = LearnedModels.__new__(LearnedModels)
    lm.items, lm.typical = sorted(typical), typical
    lm.auctions = (real_df.groupby("auctionid").agg(item=("item", "first"), dur=("duration_days", "first"),
                   openbid=("openbid", "first"), price=("price", "first"), n=("bid", "count")).reset_index())

    # --- price path: does knowing more than the time help? (pinball loss on auctions not used for training; lower = better)
    d, X, g = lm._path_training(real_df)
    groups = d.auctionid.values
    levels = [0.1, 0.5, 0.9]
    cols = {"time only": [0], "time + length + item + opening bid + bids": list(range(X.shape[1]))}
    res = {k: [] for k in ["no model (same curve for all times)"] + list(cols)}
    for tr, te in GroupKFold(5).split(X, g, groups):
        res["no model (same curve for all times)"].append(np.mean([_pinball(g[te], np.quantile(g[tr], a), a) for a in levels]))
        for name, c in cols.items():
            res[name].append(np.mean([_pinball(g[te], _quantile_gb(a).fit(X[tr][:, c], g[tr]).predict(X[te][:, c]), a)
                                      for a in levels]))
    print("PRICE PATH model: pinball loss on unseen auctions (lower is better)")
    for k, v in res.items():
        print(f"   {k:48s} {np.mean(v):.4f}")

    # --- bid count: Poisson deviance on unseen auctions
    a, Xc, y = lm._count_training()
    ga = a.auctionid.values
    dev = {"constant average": [], "item + length + opening bid + price (model)": []}
    def pdev(y_, m_): m_ = np.maximum(m_, 1e-6); return float(np.mean(2 * (y_ * np.log(np.maximum(y_, 1e-9) / m_) - (y_ - m_))))
    for tr, te in GroupKFold(5).split(Xc, y, ga):
        dev["constant average"].append(pdev(y[te], np.full(len(te), y[tr].mean())))
        dev["item + length + opening bid + price (model)"].append(pdev(y[te], lm._count_model().fit(Xc[tr], y[tr]).predict(Xc[te])))
    print("\nBID COUNT model: Poisson deviance on unseen auctions (lower is better)")
    for k, v in dev.items():
        print(f"   {k:48s} {np.mean(v):.3f}")


if __name__ == "__main__":
    from simulator import load_real
    data = HERE / "auction.csv"
    evaluate(load_real(str(data)), data)
    print("\nPRICE model: see `python price_advisor.py train` (typical error 9.4% vs 10.9% for the median baseline).")
