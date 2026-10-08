"""Chronological, fixed-size contextual entry rules.

The target is lifetime standalone cash-relative net payoff divided by reference
entry vega 450, not actual gross vega. Archived labels use $0.65 commission,
whereas the funded replay uses $0.03. Validation aggregates lifetime payoffs by
signal date: this is an unfunded surrogate, never daily P&L or portfolio Sharpe.
The caller supplies point-in-time candidates and independently replays selections.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


SETTINGS = dict(months=18, validation_months=6, fold_months=3,
                alpha_multipliers=[0.1, 1.0, 10.0], quantiles=[0.5, 0.7, 0.9],
                minimum_training_rows=100, minimum_validation_entries=20,
                spline_knots=3, spline_degree=2, economic_floor=0.0)


def _valid_outcomes(rows, target):
    valid = np.isfinite(pd.to_numeric(rows[target], errors="coerce"))
    for flag in ("finite_label", "train_eligible"):
        if flag in rows:
            valid &= rows[flag].fillna(False).astype(bool)
    return valid


def completed_window(rows, cutoff, months=18, target="target"):
    """Both signal and completed outcome must precede the fitting instant."""
    cutoff = pd.Timestamp(cutoff)
    date = pd.to_datetime(rows.date)
    end = pd.to_datetime(rows.end)
    valid = _valid_outcomes(rows, target)
    return rows.loc[(date >= cutoff - pd.DateOffset(months=months)) &
                    (date < cutoff) & (end < cutoff) & valid].copy()


def make_pipeline(kind, alpha_multiplier, n):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("scale", StandardScaler())]
    if kind == "spline":
        steps += [("spline", SplineTransformer(n_knots=3, degree=2,
                                               include_bias=False,
                                               extrapolation="linear")),
                  ("spline_scale", StandardScaler())]
    elif kind != "ridge":
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline(steps + [("ridge", Ridge(alpha=alpha_multiplier * n,
                                              solver="svd"))])


def _x(rows, features):
    return rows.loc[:, features].replace([np.inf, -np.inf], np.nan)


def best_per_pair_date(frame):
    """One candidate direction; exact score ties use ascending direction."""
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(["date", "pair", "score", "direction"],
                                ascending=[True, True, False, True], kind="stable")
    return ordered.drop_duplicates(["date", "pair"], keep="first").copy()


def tune_hurdle(oof, calendar, target="target", minimum_entries=20):
    winners = best_per_pair_date(oof)
    scores = winners.score.to_numpy(float)
    thresholds = sorted(set([0.0] + [max(0.0, float(v)) for v in
                                      np.quantile(scores, SETTINGS["quantiles"])]))
    records = []
    for threshold in thresholds:
        chosen = winners.loc[winners.score > threshold]
        daily = chosen.groupby("date")[target].sum().reindex(calendar, fill_value=0.)
        sd = float(daily.std(ddof=1))
        objective = (float(daily.mean()) / sd if len(chosen) >= minimum_entries
                     and np.isfinite(sd) and sd > 0 else None)
        records.append(dict(hurdle=threshold, entries=len(chosen),
                            active_dates=int((daily != 0).sum()),
                            mean_daily_payoff=float(daily.mean()),
                            daily_payoff_sd=sd if np.isfinite(sd) else None,
                            objective=objective))
    feasible = [r for r in records if r["objective"] is not None and
                r["mean_daily_payoff"] > 0]
    # Lower threshold wins exact objective ties; all-negative validation skips.
    winner = max(feasible, key=lambda r: (r["objective"], -r["hurdle"])) if feasible else None
    return (winner["hurdle"] if winner else np.inf), records


@dataclass
class MonthlyRule:
    features: list
    model: Optional[Pipeline]
    hurdle: float
    metadata: dict

    def predict_scores(self, frame):
        if self.model is None:
            return np.full(len(frame), np.nan)
        return self.model.predict(_x(frame, self.features))

    def select(self, frame):
        """Return original rows with score/selected; never consult target/end."""
        result = frame.copy()
        result["score"] = self.predict_scores(frame)
        result["prediction"] = result["score"]
        result["hurdle"] = self.hurdle
        result["selected"] = False
        # Positional row keys avoid duplicated caller indices selecting two sides.
        result["_row_key"] = np.arange(len(result))
        winners = best_per_pair_date(result)
        ids = winners.loc[winners.score > self.hurdle, "_row_key"].to_numpy(int)
        result.iloc[ids, result.columns.get_loc("selected")] = True
        return result.drop(columns="_row_key")


def fit_month(rows, features, cutoff, kind="ridge", target="target", calendar=None,
              tick: Optional[Callable] = None, budget=None):
    """Fit with two causal OOF folds, choose alpha by MSE then entry hurdle.

    Calendar should contain all signal trading dates, including zero-entry days.
    If omitted, a business-day calendar is used and this is recorded explicitly.
    The returned metadata is JSON serializable. No outer-month outcomes are read.
    """
    cutoff = pd.Timestamp(cutoff)
    if budget is not None and tick is not None:
        raise ValueError("Supply budget or tick, not both")
    if budget is not None:
        tick = budget.rest
    features = list(features)
    forbidden = {target, "date", "end", "pair", "label_row", "selected", "score"}
    if set(features) & forbidden:
        raise ValueError("Outcome, identity, or output columns cannot be predictors")
    if not features:
        raise ValueError("At least one feature is required")
    if rows.duplicated(["date", "pair", "direction"]).any():
        raise ValueError("Candidate date/pair/direction keys must be unique")
    outer = completed_window(rows, cutoff, target=target)
    metadata = dict(cutoff=cutoff.isoformat(), kind=kind, features=features,
                    training_rows=len(outer), training_pair_dates=len(
                        outer.drop_duplicates(["date", "pair"])),
                    latest_training_outcome=(pd.to_datetime(outer.end).max().isoformat()
                                             if len(outer) else None),
                    settings=SETTINGS.copy(), folds=[], candidates=[],
                    calendar_source="supplied" if calendar is not None else "business_days",
                    validation_objective="mean/std of signal-date aggregated lifetime payoff; not daily P&L or funded Sharpe",
                    status="insufficient_training_rows", alpha_multiplier=None,
                    hurdle=None)
    empty = MonthlyRule(features, None, np.inf, metadata)
    if len(outer) < SETTINGS["minimum_training_rows"]:
        return empty
    start = cutoff - pd.DateOffset(months=6)
    dates = pd.to_datetime(rows.date)
    ends = pd.to_datetime(rows.end)
    valid_target = _valid_outcomes(rows, target)
    folds = []
    for offset in (6, 3):
        first = cutoff - pd.DateOffset(months=offset)
        last = first + pd.DateOffset(months=3)
        train = completed_window(rows, first, target=target)
        validation = rows.loc[(dates >= first) & (dates < last) &
                              (ends < cutoff) & valid_target].copy()
        usable = len(train) >= SETTINGS["minimum_training_rows"] and len(validation) > 0
        metadata["folds"].append(dict(start=first.isoformat(), stop=last.isoformat(),
            train_rows=len(train), validation_rows=len(validation), usable=usable,
            latest_training_outcome=(pd.to_datetime(train.end).max().isoformat()
                                     if len(train) else None)))
        if usable:
            folds.append((first, last, train, validation))
    if len(folds) != 2:
        metadata["status"] = "insufficient_chronological_validation"
        return empty
    if calendar is None:
        cal = pd.bdate_range(start, cutoff - pd.Timedelta(days=1))
    else:
        cal = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
        cal = cal[(cal >= start) & (cal < cutoff)]
    if len(cal) < 2:
        raise ValueError("Validation calendar must contain at least two dates")
    candidates = []
    for alpha in SETTINGS["alpha_multipliers"]:
        predictions = []
        for first, last, train, validation in folds:
            model = make_pipeline(kind, alpha, len(train))
            model.fit(_x(train, features), train[target].to_numpy(float))
            pred = validation[["date", "pair", "direction", target]].copy()
            pred["score"] = model.predict(_x(validation, features))
            predictions.append(pred)
            if tick:
                tick()
        oof = pd.concat(predictions, ignore_index=True)
        if not pd.DatetimeIndex(oof.date).isin(cal).all():
            raise ValueError("Validation calendar omits candidate signal dates")
        mse = float(np.mean((oof.score - oof[target]) ** 2))
        metadata["candidates"].append(dict(alpha_multiplier=alpha, oof_mse=mse,
                                           oof_rows=len(oof)))
        candidates.append((mse, -alpha, alpha, oof))
    _, _, alpha, oof = min(candidates, key=lambda v: (v[0], v[1]))
    hurdle, choices = tune_hurdle(oof, cal, target=target,
                                 minimum_entries=SETTINGS["minimum_validation_entries"])
    final_model = make_pipeline(kind, alpha, len(outer))
    final_model.fit(_x(outer, features), outer[target].to_numpy(float))
    if tick:
        tick()
    metadata.update(status="fitted" if np.isfinite(hurdle) else "validation_skip_all",
                    alpha_multiplier=alpha, hurdle=float(hurdle) if np.isfinite(hurdle) else None,
                    hurdle_candidates=choices, oof_rows=len(oof),
                    oof_pair_dates=len(oof.drop_duplicates(["date", "pair"])),
                    validation_calendar_days=len(cal))
    return MonthlyRule(features, final_model, hurdle, metadata)
