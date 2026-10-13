"""Read the existing submission data without restoring the research workspace."""
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile

import pandas as pd

from .contracts import MarketRow

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/submission"
ARCHIVE = ROOT / "data/p25_validation_split_experiment.zip"
FEATURE = "artifacts/contracts/p25_feature_search"
LABELS = BUNDLE / "artifacts/contracts/uniform_smile/pairs"
MARKET = BUNDLE / "artifacts/contracts/p25_signal_transfer/market"
CASES = {
    "learned": "artifacts/contracts/p25_validation_split/replays/screened_selected",
    "fixed50": "artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed50",
    "fixed75": "artifacts/contracts/p25_validation_split/replays/screened_fixed75",
    "fixed100": "artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed100",
}


def read_bytes(relative):
    """Prefer committed bundles; never depend on untracked artifacts/."""
    path = BUNDLE / relative
    if path.is_file():
        return path.read_bytes()
    with zipfile.ZipFile(ARCHIVE) as archive:
        return archive.read(relative)


def read_parquet(relative, columns=None):
    return pd.read_parquet(BytesIO(read_bytes(relative)), columns=columns,
                           use_threads=False)


def read_decisions(case):
    return pd.read_csv(BytesIO(read_bytes(f"{CASES[case]}/decisions.csv")),
                       parse_dates=["entry_date", "signal_date"])


def training_rows(features):
    columns = features + ["date", "end", "pair", "signal_eligible",
                          "finite_label", "target", "position", "label_row",
                          "direction"]
    return read_parquet(f"{FEATURE}/training_rows.parquet", columns)


def proposals():
    return read_parquet(f"{CASES['learned']}/scored_allocations.parquet")


def saved_model(month):
    import joblib
    return joblib.load(BytesIO(read_bytes(
        f"{FEATURE}/models/control/{month}.joblib")))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_evidence():
    """Re-rank the archived grid on validation scores, not evaluation scores."""
    grid = pd.read_csv(ROOT / "reports/p25_validation_split/validation_candidates.csv")
    grid = grid.loc[grid.pool.eq("screened")].sort_values(
        ["validation_sharpe", "candidate"], ascending=[False, True])
    locked = json.loads((ROOT / "reports/p25_validation_split/selection.json").read_text())
    if grid.iloc[0].candidate != locked["selected"]["screened"]["candidate"]:
        raise ValueError("Validation winner differs from the saved selection")
    return grid


class Market:
    """Read-only quotes with a two-year/two-session bounded cache."""

    def __init__(self):
        stock = pd.read_parquet(MARKET / "stocks.parquet",
                               columns=["date", "secid", "close", "cfadj"],
                               use_threads=False)
        self.spots = stock.pivot(index="date", columns="secid", values="close")
        self.adjustments = stock.pivot(index="date", columns="secid", values="cfadj")
        self.rates = pd.read_parquet(MARKET / "rates.parquet", use_threads=False).rate
        self.divs = pd.read_parquet(MARKET / "dividends.parquet", use_threads=False)
        self.years, self.days = OrderedDict(), OrderedDict()

    def stock(self, date, sid):
        # Documented 2025 corporate-event exit proxy; preserve the paper ledger.
        if sid == 101322 and date == pd.Timestamp("2025-07-18"):
            return (1.025 * self.spots.at[date, 102968],
                    self.adjustments.at[pd.Timestamp("2025-07-17"), sid])
        return self.spots.at[date, sid], self.adjustments.at[date, sid]

    def rate(self, date):
        return float(self.rates.loc[date])

    def quote(self, date, optionid):
        if date not in self.days:
            if date.year not in self.years:
                quotes = pd.read_parquet(MARKET / f"{date.year}_quotes.parquet",
                                         use_threads=False)
                self.years[date.year] = quotes.set_index("date", drop=False).sort_index()
                while len(self.years) > 2:
                    self.years.popitem(last=False)
            self.years.move_to_end(date.year)
            quotes = self.years[date.year]
            today = quotes.loc[[date]] if date in quotes.index else quotes.iloc[:0]
            self.days[date] = {int(row["optionid"]): MarketRow(row)
                               for row in today.to_dict("records")}
            while len(self.days) > 2:
                self.days.popitem(last=False)
        self.days.move_to_end(date)
        return self.days[date].get(int(optionid))

    def dividend_pv(self, date, sid, expiry, rate):
        import numpy as np
        dividends = self.divs.loc[
            self.divs.secid.eq(sid) & self.divs.declare_date.le(date)
            & self.divs.ex_date.gt(date) & self.divs.ex_date.le(expiry)]
        return float((dividends.amount * np.exp(
            -rate * (dividends.ex_date - date).dt.days / 365)).sum())
