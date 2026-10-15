"""Cash-relative performance, continuing books, and paired uncertainty."""
import numpy as np
import pandas as pd


def sharpe(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sharpe requires finite daily returns")
    std = values.std(ddof=1)
    return float(np.sqrt(252) * values.mean() / std) if std > 1e-12 else 0.


def mark_benchmark(nav, rates, capital):
    """Accrue previous-close rates over calendar days, including weekends."""
    elapsed = nav.index.to_series().diff().dt.days.fillna(0) / 365
    cash_return = rates.reindex(nav.index).shift(1).fillna(0) * elapsed
    nav["cash_benchmark"] = capital * (1 + cash_return).cumprod()
    nav["excess_return"] = nav.return_ - cash_return
    return nav


def period_metrics(nav, decisions, start, stop):
    """Same definitions as the paper, including the preceding boundary close."""
    part = nav.loc[pd.Timestamp(start):pd.Timestamp(stop)]
    if part.empty:
        raise ValueError("Empty reporting period")
    first = nav.index.get_loc(part.index[0])
    before = nav.iloc[max(0, first - 1)]
    years = (part.index[-1] - nav.index[max(0, first - 1)]).days / 365.25
    returns = part.excess_return.iloc[1:] if first == 0 else part.excess_return
    relative = part.nav / part.cash_benchmark
    total_growth = float(part.nav.iloc[-1] / before.nav)
    relative_growth = float(relative.iloc[-1] / (before.nav / before.cash_benchmark))
    in_period = decisions.entry_date.between(part.index[0], part.index[-1])
    entered = decisions.status.eq("entered") & in_period
    drawdown_path = pd.concat([
        pd.Series([before.nav / before.cash_benchmark]), relative.reset_index(drop=True)
    ], ignore_index=True)
    return {
        "annualized_excess_sharpe": sharpe(returns),
        "annualized_total_account_return_pct": 100 * (total_growth**(1 / years) - 1),
        "annualized_growth_relative_to_cash_pct": 100 * (relative_growth**(1 / years) - 1),
        "total_account_return_pct": 100 * (total_growth - 1),
        "return_above_cash_pp": 100 * (total_growth - part.cash_benchmark.iloc[-1] / before.cash_benchmark),
        "profit_above_cash": float((part.nav.iloc[-1] - part.cash_benchmark.iloc[-1])
                                   - (before.nav - before.cash_benchmark)),
        "entries": int(entered.sum()), "proposals": int(in_period.sum()),
        "average_reserve_pct": 100 * float((part.capital_reserve / part.nav).mean()),
        "peak_reserve_pct": 100 * float((part.capital_reserve / part.nav).max()),
        "max_cash_relative_drawdown_pct": 100 * float(
            (drawdown_path / drawdown_path.cummax() - 1).min()),
    }


def paired_interval(learned, fixed, block=126, draws=1000):
    """Circular blocks with identical resampling indices for both policies.

    Conditional on fitted policies; no adjustment for retrospective selection.
    """
    if not learned.index.equals(fixed.index):
        raise ValueError("Paired returns must have identical dates")
    a, b = learned.to_numpy(), fixed.to_numpy()
    rng, differences = np.random.default_rng(2026), []
    for _ in range(draws):
        starts = rng.integers(0, len(a), size=int(np.ceil(len(a) / block)))
        indices = ((starts[:, None] + np.arange(block)[None]) % len(a)).ravel()[:len(a)]
        differences.append(sharpe(a[indices]) - sharpe(b[indices]))
    return {"point": sharpe(a) - sharpe(b),
            "lower_95": float(np.quantile(differences, .025)),
            "upper_95": float(np.quantile(differences, .975)),
            "block_sessions": block, "replications": draws}


def verify_ledger(nav, trades, decisions, signals):
    """Reconcile cash flows independently of the main simulation loop."""
    if len(decisions) != len(signals):
        raise AssertionError("Each proposal needs a decision")
    entered = decisions.loc[decisions.status.eq("entered")]
    imbalance = (entered.vega_a - entered.vega_b).abs() / entered[["vega_a", "vega_b"]].max(axis=1)
    if not imbalance.le(.1 + 1e-12).all():
        raise AssertionError("Initial vega mismatch exceeds 10%")
    if not trades.groupby(["cohort", "optionid"]).quantity.sum().eq(0).all():
        raise AssertionError("Open option inventory remains")
    cashflow = (-trades.quantity * 100 * trades.price - trades.commission
                - trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index, fill_value=0)
    expected = (cashflow + nav.option_value.diff().fillna(nav.option_value)
                + nav.hedge_gross_pnl - nav.hedge_cost + nav.interest
                - nav.stock_borrow_cost + nav.dividend_cash)
    error = float((expected - nav.pnl).abs().max())
    if error >= 1e-5 or abs(nav.option_value.iloc[-1]) + abs(nav.stock_value.iloc[-1]) >= 1e-7:
        raise AssertionError("Funded ledger does not reconcile or inventory remains")
    return {"independent_ledger_error": error, "entries": len(entered)}
