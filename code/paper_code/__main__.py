"""One entry point: verify, metrics, fit, replay, or test."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/paper_code"
PERIODS = {"validation": ("2018-01-01", "2021-12-31"),
           "evaluation": ("2022-01-01", "2025-08-29")}


def score_proposals():
    import numpy as np
    from . import data, policy
    from .signals import FEATURES

    rows = data.proposals()
    states = data.training_rows(FEATURES)[["date", "pair"] + FEATURES]
    rows = rows.merge(states, on=["date", "pair"], validate="one_to_one")
    if len(rows) != 608 or rows[FEATURES].isna().any().any():
        raise ValueError("The paper requires 608 proposals with complete archived inputs")
    for month, indices in rows.groupby(rows.date.dt.to_period("M")).groups.items():
        model = data.saved_model(str(month.start_time.date()))
        np.testing.assert_array_equal(policy.predict(model, rows.loc[indices]),
                                      rows.loc[indices, "learned_level"])
    print("All 608 archived sizing decisions reproduced from their monthly models.")


def performance(write=False):
    import numpy as np
    import pandas as pd
    from . import data, metrics

    tables, years, later = [], [], {}
    expected = pd.read_csv(ROOT / "reports/p25_validation_split/periods.csv")
    for case, folder in data.CASES.items():
        nav = data.read_parquet(f"{folder}/daily_nav.parquet")
        decisions = data.read_decisions(case)
        for period, (start, stop) in PERIODS.items():
            result = metrics.period_metrics(nav, decisions, start, stop)
            reference_case = "screened_selected" if case == "learned" else "screened_" + case
            reference_period = "retrospective_holdout" if period == "evaluation" else period
            reference = expected.loc[expected.case.eq(reference_case)
                                     & expected.period.eq(reference_period)].iloc[0]
            for key in ("annualized_excess_sharpe", "profit_above_cash", "entries",
                        "annualized_total_account_return_pct",
                        "annualized_growth_relative_to_cash_pct"):
                np.testing.assert_allclose(result[key], reference[key], rtol=1e-10, atol=1e-7)
            tables.append({"policy": case, "period": period, **result})
        for year in sorted(nav.index.year.unique()):
            result = metrics.period_metrics(nav, decisions, f"{year}-01-01", f"{year}-12-31")
            years.append({"policy": case, "year": int(year), **result})
        later[case] = nav.loc["2022-01-01":, "excess_return"]
    interval = metrics.paired_interval(later["learned"], later["fixed100"])
    reference = pd.read_csv(ROOT / "reports/p25_validation_split/holdout_intervals.csv")
    reference = reference.loc[reference.pool.eq("screened")].iloc[0]
    for key in ("point", "lower_95", "upper_95"):
        np.testing.assert_allclose(interval[key], reference[key], atol=1e-12)
    table = pd.DataFrame(tables)
    print(table[["policy", "period", "annualized_excess_sharpe", "entries",
                 "profit_above_cash"]].to_string(index=False))
    print("Paired Sharpe-gain interval:", json.dumps(interval))
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT / "periods.csv", index=False)
        pd.DataFrame(years).to_csv(OUT / "annual.csv", index=False)
        (OUT / "sharpe_interval.json").write_text(json.dumps(interval, indent=2) + "\n")


def fit(month):
    import joblib
    import pandas as pd
    from . import data, policy
    from .signals import FEATURES

    months = list(pd.period_range("2018-01-01", "2025-06-01", freq="M")) if month == "all" else [pd.Period(month, freq="M")]
    meta = data.training_rows(FEATURES)
    calendar = pd.read_parquet(data.MARKET / "rates.parquet", use_threads=False).index
    folder = OUT / "models"
    folder.mkdir(parents=True, exist_ok=True)
    for period in months:
        stamp = str(period.start_time.date())
        model, log = policy.fit_month(meta, calendar, data.LABELS, stamp)
        policy.verify_model(model, data.saved_model(stamp))
        joblib.dump(model, folder / f"{stamp}.joblib")
        (folder / f"{stamp}.json").write_text(json.dumps(log, indent=2) + "\n")
        print(f"{stamp}: exact archived fit reproduced; {log}", flush=True)


def replay(case):
    from io import BytesIO
    import pandas as pd
    from . import data, ledger, metrics
    from .contracts import Settings

    cfg = Settings(**json.loads((Path(__file__).parent / "execution.json").read_text()))
    events = json.loads((ROOT / "config/p25_submission.json").read_text())["corporate_events"]
    market = data.Market()
    rows = data.proposals()
    if case != "learned":
        rows["learned_level"] = {"fixed50": 2, "fixed75": 3, "fixed100": 4}[case]
    print(f"Replaying {case}: {len(rows)} proposals, one shared account.", flush=True)
    nav, trades, decisions, audit = ledger.simulate(
        market, rows, market.rates.index[market.rates.index >= "2018-01-01"], cfg,
        pair_capital=1_000_000, wing_capital=1_000_000,
        policy=lambda row, context: int(row["learned_level"]), corporate_events=events)
    metrics.mark_benchmark(nav, market.rates, cfg.capital)
    audit.update(metrics.verify_ledger(nav, trades, decisions, rows))
    folder = data.CASES[case]
    expected = data.read_parquet(f"{folder}/daily_nav.parquet")
    pd.testing.assert_frame_equal(nav, expected, check_dtype=False, rtol=1e-10, atol=1e-7)
    saved = pd.read_csv(BytesIO(data.read_bytes(f"{folder}/trades.csv")),
                        parse_dates=["date"])
    pd.testing.assert_frame_equal(trades.reset_index(drop=True), saved,
                                  check_dtype=False, rtol=1e-10, atol=1e-7)
    saved = data.read_decisions(case)
    pd.testing.assert_frame_equal(decisions.reset_index(drop=True), saved,
                                  check_dtype=False, rtol=1e-10, atol=1e-7)
    destination = OUT / "replays" / case
    destination.mkdir(parents=True, exist_ok=True)
    nav.to_parquet(destination / "daily_nav.parquet")
    trades.to_csv(destination / "trades.csv", index=False)
    decisions.to_csv(destination / "decisions.csv", index=False)
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"{case}: daily NAV, fills and admissions match the archived portfolio.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "metrics", "fit", "replay", "test"])
    parser.add_argument("--month", default="2018-01-01", help="Fit one month or 'all' (90 fits)")
    parser.add_argument("--policy", choices=["learned", "fixed50", "fixed75", "fixed100"], default="learned")
    args = parser.parse_args()
    if args.command in ("fit", "replay") and os.environ.get("PAPER_CODE_WORKER") != "1":
        from .guard import run
        raise SystemExit(run(sys.argv[1:], ROOT))
    from .signals import FEATURES
    specification = json.loads((Path(__file__).parent / "strategy.json").read_text())["screened"]
    if (specification["inputs"] != FEATURES or specification["lookback_months"] != 18
            or specification["max_leaves"] != 8 or specification["actions"] != [.5, .75, 1.]):
        raise ValueError("This package implements the frozen paper specification")
    import pyarrow
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    if args.command == "verify":
        from .data import selection_evidence
        selection_evidence()
        score_proposals()
        performance()
    elif args.command == "metrics":
        performance(write=True)
    elif args.command == "fit":
        fit(args.month)
    elif args.command == "replay":
        replay(args.policy)
    else:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
