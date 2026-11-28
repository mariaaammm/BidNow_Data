"""
Reproducible benchmark for the price advisor: several model types x feature sets, same splits for everyone.

    python benchmark_price_advisor.py                 # about 3-6 minutes, writes price_advisor_benchmark.csv
    python benchmark_price_advisor.py --repeats 8     # steadier numbers, slower

What is compared (every row is one auction, never one bid; target = log of the final price):
  baselines     item median, item+days median
  models        GradientBoosting (Huber, the PREVIOUS advisor), RandomForest, ExtraTrees, HistGradientBoosting,
                Ridge, Ridge with per-item slopes, and the CURRENT price advisor (shallow GB + per-item Ridge, averaged)
  features      item | item+days | item+opening bid (+days) | + opening bid relative to the item's typical price
  validation    "ordinary": 5-fold, balanced by item
                "strict":   same, but auctions with the same item AND opening bid stay on one side, so every test
                            auction has an opening bid that item never had in training
Typical price of an item is always computed from the TRAINING part of each split only (no leakage).
Metrics: MAE, RMSE, MedAE (money), MedAPE (median % error), MALE (mean absolute log error, scale-free; main score).
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import price_advisor as pa

HERE = Path(__file__).parent
warnings.filterwarnings("ignore")

SETS = {
    "item": [],
    "item+days": ["dur"],
    "item+openbid": ["lob"],
    "item+days+openbid": ["dur", "lob"],
    "item+days+openbid+relative openbid": ["dur", "lob", "rel"],
}
MODELS = {
    "GradientBoosting Huber depth2 (previous advisor)": lambda: pa._legacy_model(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0),
    "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=0),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                  min_samples_leaf=15, random_state=0),
    "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
}
FINAL = "CURRENT price advisor (GB depth1 + per-item Ridge)"


def features(df, typ, items):
    f = pd.DataFrame({"dur": df.duration_days.values.astype(float), "lob": np.log1p(df.openbid.values),
                      "rel": np.log1p(df.openbid.values / df.item.map(typ).values)})
    for it in items:
        f["is_" + it] = (df.item.values == it).astype(float)
    return f


def one_split(a, items, r, tr, te):
    trd, ted = a.iloc[tr], a.iloc[te]
    ytr = np.log(trd.price.values)
    typ = trd.groupby("item").price.median().to_dict()
    Ftr, Fte = features(trd, typ, items), features(ted, typ, items)
    icols = ["is_" + i for i in items]
    out = {("baseline: item median", "-"): ted.item.map(trd.groupby("item").price.median()).values}
    m2 = trd.groupby(["item", "duration_days"]).price.median()
    out[("baseline: item+days median", "-")] = np.array([m2.get((i, d), typ[i]) for i, d in zip(ted.item, ted.duration_days)])
    for mname, mk in MODELS.items():
        for sname, cols in SETS.items():
            out[(mname, sname)] = np.exp(mk().fit(Ftr[cols + icols], ytr).predict(Fte[cols + icols]))
    adv = pa.PriceAdvisor().fit(trd)
    out[(FINAL, "item+days+openbid")] = np.exp(adv._predict_log(adv._X(ted.item, ted.duration_days, ted.openbid)))
    return r, te, out


def metrics(P, y, mask):
    yy = y[mask]
    e = [p[mask] - yy for p in P]
    return {"MAE": np.mean([np.abs(x).mean() for x in e]), "RMSE": np.mean([np.sqrt((x ** 2).mean()) for x in e]),
            "MedAE": np.mean([np.median(np.abs(x)) for x in e]),
            "MedAPE_%": 100 * np.mean([np.median(np.abs(x) / yy) for x in e]),
            "MALE": np.mean([np.abs(np.log(p[mask] / yy)).mean() for p in P])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=[str(pa.DEFAULT_DATA)])
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777, help="split seed (the model was chosen with seed 42, then confirmed with 777)")
    ap.add_argument("--out", default=str(HERE / "price_advisor_benchmark.csv"))
    args = ap.parse_args()
    a = pa.load_auctions(*args.data).reset_index(drop=True)
    items, y = sorted(a.item.unique()), a.price.values
    rng = np.random.default_rng(0)
    rows = []
    for scheme, grouped in (("ordinary", False), ("strict", True)):
        jobs = [(r, tr, te) for r, tr, te in pa._cv_splits(a, args.repeats, args.seed, grouped)]
        res = Parallel(n_jobs=-1)(delayed(one_split)(a, items, r, tr, te) for r, tr, te in jobs)
        P = {}
        for r, te, out in res:
            for k, p in out.items():
                P.setdefault(k, np.full((args.repeats, len(a)), np.nan))[r, te] = p
        ref = np.abs(np.log(P[("GradientBoosting Huber depth2 (previous advisor)", "item+days+openbid")] / y)).mean(0)
        for (m, s), p in P.items():
            row = {"validation": scheme, "model": m, "features": s, "auctions": len(a), **metrics(p, y, np.ones(len(a), bool))}
            d = np.abs(np.log(p / y)).mean(0) - ref  # per auction, against the previous advisor
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
            row.update({"MALE_diff_vs_previous": d.mean(), "diff_ci_low": np.percentile(bs, 2.5), "diff_ci_high": np.percentile(bs, 97.5)})
            for it in items:
                mm = metrics(p, y, (a.item == it).values)
                row[f"{it} MedAPE_%"], row[f"{it} MAE"] = mm["MedAPE_%"], mm["MAE"]
            rows.append(row)
    T = pd.DataFrame(rows).round(4)
    T.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    for scheme in ("ordinary", "strict"):
        t = T[T.validation == scheme].sort_values("MALE")
        print(f"\n=== {scheme} validation, {len(a)} auctions, 5-fold x {args.repeats}: best 8 by MALE (lower is better) ===")
        print(t[["model", "features", "MAE", "RMSE", "MedAPE_%", "MALE", "MALE_diff_vs_previous"]].head(8).to_string(index=False))
        b = t[t.model.str.startswith("baseline")].iloc[0]
        print(f"baseline item median: MAE {b.MAE:.1f}  MedAPE {b['MedAPE_%']:.1f}%  MALE {b.MALE:.3f}")
    print(f"\nfull table (all models, per-item results, paired confidence intervals) -> {args.out}")


if __name__ == "__main__":
    main()
