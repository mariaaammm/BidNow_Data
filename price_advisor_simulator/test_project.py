"""
Automatic checks for the whole project. Run:   python test_project.py
Takes about 1-2 minutes. Prints PASS/FAIL per check and exits with an error if anything fails.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PY = sys.executable
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail and not ok else ""))


def run(*args, cwd=HERE):
    return subprocess.run([PY, *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=900)


def main():
    # the tests retrain with --no-eval, which rewrites these report files: put the real ones back afterwards
    keep = {f: f.read_bytes() for f in (HERE / "price_advisor_report.json", HERE / "live_forecast_report.json") if f.exists()}
    try:
        run_checks()
    finally:
        for f, data in keep.items():
            f.write_bytes(data)


def run_checks():
    tmp = Path(tempfile.mkdtemp())

    # ---- loading real and Excel-style data
    import price_advisor as pa
    a = pa.load_auctions(HERE / "auction.csv")
    check("real data: 628 auctions, 3 items", len(a) == 628 and a.item.nunique() == 3, f"{len(a)}")

    rng = np.random.default_rng(0)
    rows = "\r\n".join(f"Bike;{rng.choice([3, 5, 7])};{str(round(rng.uniform(1, 20), 2)).replace('.', ',')};"
                       f"{str(round(rng.uniform(80, 200), 2)).replace('.', ',')}" for _ in range(30))
    f = tmp / "excel.csv"
    f.write_text("﻿item;duration_days;openbid;price\r\n" + rows, encoding="utf-8")
    e = pa.load_auctions(f)
    check("Excel CSV (BOM, ';', decimal comma) loads all rows", len(e) == 30 and e.price.mean() > 80, f"{len(e)}")
    (tmp / "empty.csv").write_text("item,duration_days,openbid,price\n")
    try:
        pa.load_auctions(tmp / "empty.csv")
        check("header-only file gives a clear error", False)
    except ValueError:
        check("header-only file gives a clear error", True)

    # ---- price advisor
    r = run("price_advisor.py", "train", "--no-eval")
    check("price advisor trains", r.returncode == 0, r.stderr[-300:])
    out = json.loads(run("price_advisor.py", "predict", "--item", "Xbox game console", "--days", 7, "--openbid", 9.99).stdout)
    lo, hi = out["range_80pct"]
    check("prediction is sensible (range contains estimate, tip present)",
          lo <= out["expected_final_price"] <= hi and out["suggested_opening_bid"] > 0 and out["tip"])
    for bad in (["--days", "-3", "--openbid", "5"], ["--days", "7", "--openbid", "-1"]):
        r = run("price_advisor.py", "predict", "--item", "Xbox game console", *bad)
        check(f"rejects bad input {bad}", r.returncode != 0 and "Traceback" not in r.stderr)
    r = run("price_advisor.py", "predict", "--item", "Nonexistent", "--days", 7, "--openbid", 5)
    check("unknown item gives a clear error", r.returncode != 0 and "unknown item" in r.stderr)

    # ---- price advisor quality, honesty of ranges, new model internals (one repeat of each validation, ~30 s)
    full = pa.load_auctions(HERE / "auction.csv")
    for grouped in (False, True):
        tag = "strict (unseen opening bids)" if grouped else "ordinary"
        rep = pa.evaluate(full, repeats=1, seed=5, grouped=grouped)
        o = rep["overall"]
        check(f"advisor beats the item-median baseline on mean error and RMSE, {tag}",
              o["model"]["MAE"] < o["base_item"]["MAE"] and o["model"]["RMSE"] < o["base_item"]["RMSE"],
              f"{o['model']['MAE']} vs {o['base_item']['MAE']}")
        lo_ok = 68 <= o["range_coverage_pct"] <= 90 if grouped else 74 <= o["range_coverage_pct"] <= 90
        check(f"80% range coverage is in a plausible band, {tag}", lo_ok, f"{o['range_coverage_pct']}%")
    adv = pa.PriceAdvisor().fit(full)
    new_ob = adv.predict("Cartier wristwatch", 7, 123.45)  # an opening bid never seen in the data
    check("opening bid never seen in training still gives an ordered, positive range",
          0 < new_ob["range_80pct"][0] <= new_ob["expected_final_price"] <= new_ob["range_80pct"][1])
    lo_p, hi_p = adv.predict("Palm Pilot M515 PDA", 7, 1), adv.predict("Cartier wristwatch", 7, 1)
    check("Cartier range is much wider than Palm Pilot range (items have their own spread)",
          hi_p["range_80pct"][1] / hi_p["range_80pct"][0] > 2 * lo_p["range_80pct"][1] / lo_p["range_80pct"][0])
    import joblib
    joblib.dump(adv, tmp / "m.joblib")
    check("trained advisor survives save and load with identical predictions",
          joblib.load(tmp / "m.joblib").predict("Xbox game console", 7, 9.99) == adv.predict("Xbox game console", 7, 9.99))
    one_group = pd.DataFrame({"item": ["X"] * 40, "duration_days": [7] * 40, "openbid": [1.0] * 40,
                              "price": np.linspace(10, 30, 40)})
    check("advisor fits when every auction has the same opening bid (no crash in the strict-range step)",
          pa.PriceAdvisor().fit(one_group).predict("X", 7, 1.0)["expected_final_price"] > 0)

    # ---- live forecast
    r = run("live_forecast.py", "train")
    check("live forecast trains", r.returncode == 0, r.stderr[-300:])
    r = run("live_forecast.py", "predict", "--item", "Xbox game console", "--days", 7, "--openbid", 0,
            "--elapsed", 3, "--high", 0, "--bids", 0)
    check("live forecast handles zero bids / zero opening bid", r.returncode == 0 and "expected_final_price" in r.stdout, r.stderr[-300:])
    r = run("live_forecast.py", "predict", "--item", "Xbox game console", "--days", 7, "--openbid", 1,
            "--elapsed", -1, "--high", 5, "--bids", 2)
    check("live forecast rejects negative input", r.returncode != 0 and "Traceback" not in r.stderr)

    # ---- simulator: integrity (the simulator itself stops if its 11 checks fail), reproducibility, bad arguments
    outs = {}
    for name, extra in {"stats": ["--mode", "stats"], "ml": ["--mode", "ml", "--categories", "categories.json"]}.items():
        for rep in (1, 2):
            d = tmp / f"{name}{rep}"
            r = run("simulator.py", *extra, "--listings", 400, "--seed", 5, "--now", "2026-11-27", "--out", d)
            check(f"simulator {name}: runs and passes its 11 integrity checks (run {rep})",
                  r.returncode == 0 and "all 11 passed" in r.stdout, (r.stderr or r.stdout)[-300:])
            outs[(name, rep)] = d
        same = all((outs[(name, 1)] / x).read_bytes() == (outs[(name, 2)] / x).read_bytes()
                   for x in ("users.csv", "listings.csv", "bids.csv"))
        check(f"simulator {name}: same seed gives identical files", same)
    L = pd.read_csv(outs[("ml", 1)] / "listings.csv")
    check("assumed categories are marked price_basis='assumed'",
          set(L.price_basis) == {"data", "assumed"} and (L[L.price_basis == "assumed"].item.nunique() >= 3))
    check("closed auctions have winner and final price; active ones do not",
          L[L.status == "closed"].final_price.notna().all() and L[L.status == "active"].final_price.isna().all())
    for bad in (["--users", 5], ["--listings", 0], ["--days", 0], ["--real-share", 2]):
        r = run("simulator.py", "--mode", "stats", *bad, "--out", tmp / "bad")
        check(f"simulator rejects {bad}", r.returncode != 0 and "Traceback" not in r.stderr)
    d = tmp / "small"
    r = run("simulator.py", "--mode", "stats", "--users", 100, "--listings", 200, "--out", d)
    check("simulator works with the smallest allowed user pool (no endless loop)", r.returncode == 0, r.stderr[-300:])
    badcat = tmp / "badcat.json"
    badcat.write_text(json.dumps({"categories": {"X": {"like": "Nope", "median_price": 5, "price_sigma": .5, "weight": 1}}}))
    r = run("simulator.py", "--mode", "stats", "--categories", badcat, "--out", tmp / "bad2")
    check("bad categories.json gives a clear error", r.returncode != 0 and "Traceback" not in r.stderr and "like" in r.stderr)

    # ---- model quality guards (stop silent regressions). Needs a bigger run: with few listings the medians are noise.
    d = tmp / "quality"
    r = run("simulator.py", "--mode", "ml", "--listings", 2000, "--seed", 5, "--now", "2026-11-27", "--out", d)
    v = pd.read_csv(d / "validation.csv").set_index("metric")
    r50 = v.loc["highest bid / final price at 50% of the time"]
    check("ML mode: price-over-time is close to real at 50% of the time", abs(r50.real - r50.simulated) < 0.12,
          f"{r50.real} vs {r50.simulated}")
    v3 = v.loc["median opening bid, 3-day auctions"]
    check("ML mode: 3-day opening bids are realistic (not ~10)", v3.simulated > 50, f"{v3.simulated}")

    n_bad = results.count(False)
    print(f"\n{len(results) - n_bad}/{len(results)} checks passed")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
