"""Render completed direct-threshold experiments without choosing new policies."""
import os
for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"]:
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.diagnose_p25_pca import Budget


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/p25_threshold_search"
DEFAULT_ARTIFACTS = ROOT / "artifacts/contracts/p25_threshold_search"
ORDER = ["q20", "selected_fixed", "selected_rolling", "selected_context"]
NAMES = {"q20": "Original 20% rule", "selected_fixed": "Learned fixed cutoff",
         "selected_rolling": "Monthly cutoff selection", "selected_context": "VIX-dependent cutoff"}
COLORS = {"q20": "#586574", "selected_fixed": "#2274a5",
          "selected_rolling": "#e07a26", "selected_context": "#548b42"}


def policy_text(key, protocol):
    p = protocol["policies"][key]
    if p["q_high"] is None:
        q = p["q_low"]
        return f"{key}: long at rank <= {q:.0%}; short at rank > {1-q:.0%}"
    low, high = p["q_low"], p["q_high"]
    return (f"{key}: below prior VIX 20, use {low:.0%}/{1-low:.0%} rank boundaries; "
            f"at prior VIX >= 20, use {high:.0%}/{1-high:.0%} boundaries")


def table(frame, columns=None):
    view = frame.loc[:, columns].copy() if columns else frame.copy()
    return view.to_string(index=False, float_format=lambda v: f"{v:,.3f}")


def read_completed(report, artifacts):
    completion = json.loads((report / "completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("Only a completed experiment can receive a findings report")
    protocol = json.loads((report / "protocol.json").read_text())
    periods = pd.read_csv(report / "funded/periods.csv")
    annual = pd.read_csv(report / "funded/annual.csv")
    intervals = pd.read_csv(report / "funded/sharpe_intervals.csv")
    candidates = pd.read_csv(report / "candidate_periods.csv")
    rolling = pd.read_csv(report / "rolling_schedule.csv")
    selections = {case: json.loads((artifacts / "selections" / f"{case}.json").read_text())
                  for case in ["selected_fixed", "selected_context"]}
    navs = {case: pd.read_parquet(artifacts / "replays" / case / "daily_nav.parquet", use_threads=False) for case in ORDER}
    for case in ORDER:
        if not navs[case].index.equals(navs["q20"].index):
            raise ValueError("Deployed books have inconsistent calendars")
    # Fixed/context deploy after the selection date, carrying baseline inventory.
    carry = []
    before = navs["q20"].loc[:"2021-12-31"]
    for case in ["selected_fixed", "selected_context"]:
        other = navs[case].loc[before.index]
        cols = [c for c in ["nav", "cash_benchmark", "option_value", "stock_value"] if c in before]
        error = float(np.max(np.abs(other[cols].to_numpy()-before[cols].to_numpy())))
        if error > 1e-7:
            raise ValueError(f"{case}: pre-2022 baseline carry does not reproduce")
        carry.append(dict(case=case, prior_close=str(before.index[-1].date()),
                          nav=float(other.nav.iloc[-1]),
                          option_value=float(other.option_value.iloc[-1]),
                          stock_value=float(other.stock_value.iloc[-1]),
                          maximum_pre2022_difference=error))
    return protocol, periods, annual, intervals, candidates, rolling, selections, navs, carry


def interval_interpretation(case, intervals, identical_returns=False):
    own = intervals.loc[(intervals.case == case) & (intervals.period == "evaluation")]
    if set(own.block_sessions) != {63, 126}:
        raise ValueError(f"Missing paired uncertainty estimates for {case}")
    if (own[["point", "lower95", "upper95"]] == 0).all().all():
        result = ("Daily excess returns are identical to the baseline; the difference and both intervals are exactly zero."
                  if identical_returns else "The measured Sharpe difference and both intervals are exactly zero.")
    elif (own.lower95 > 0).all():
        result = "Both conditional intervals favor the selected rule."
    elif (own.upper95 < 0).all():
        result = "Both conditional intervals favor the original 20% rule."
    elif ((own.lower95 <= 0) & (own.upper95 >= 0)).all():
        result = "Both intervals include zero; the estimated difference remains uncertain."
    else:
        result = "The interval conclusion changes with block length."
    return f"{NAMES[case]}: {result}"


def pair_findings(report, periods):
    """Optional actual funded attribution; never substitute fill cash flows."""
    pair_path = report / "funded/pair_periods.csv"
    concentration_path = report / "funded/concentration.csv"
    status_path = report / "funded/pair_attribution_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    if not pair_path.exists() or not concentration_path.exists():
        lines = ["Actual pair attribution", "Pair P&L attribution is unavailable; entry breadth alone does not establish diversified profits."]
        if status:
            lines.extend(f"  {r['case']}: {r['status']}" for r in status["cases"])
        return lines
    pairs = pd.read_csv(pair_path)
    concentration = pd.read_csv(concentration_path)
    successful = set(status["successful_cases"]) if status else set(concentration.case)
    pairs = pairs.loc[pairs.case.isin(successful) & pairs.period.eq("evaluation")].copy()
    concentration = concentration.loc[concentration.case.isin(successful) & concentration.period.eq("evaluation")].copy()
    labels = {}
    roster = ROOT / "data/submission/artifacts/broad_pairs/pairs.json"
    if roster.exists():
        labels.update({"__".join(s.split("p")[0][1:] for s in r["pair"].split("__")): r["label"]
                       for r in json.loads(roster.read_text())})
    existing = ROOT / "reports/p25_strategy_audit/pairs/pair_periods.csv"
    if existing.exists():
        prior = pd.read_csv(existing, usecols=["pair", "label"]).drop_duplicates("pair")
        labels.update(prior.set_index("pair").label.to_dict())
    pairs["label"] = pairs.pair.map(labels).fillna(pairs.pair)
    comparison = []
    ranking = []
    later = periods.loc[periods.period.eq("evaluation")].set_index("case")
    for case in ORDER:
        if case not in successful:
            continue
        own = concentration.loc[concentration.case.eq(case)]
        if len(own) != 1:
            raise ValueError(f"Missing unique later attribution for {case}")
        row = own.iloc[0]
        own_pairs = pairs.loc[pairs.case.eq(case)].sort_values("net_before_shared_financing", ascending=False)
        np.testing.assert_allclose(own_pairs.net_before_shared_financing.sum(), row.net_pair_pnl, atol=1e-5, rtol=0)
        np.testing.assert_allclose(row.net_pair_pnl + row.shared_financing_vs_cash,
            later.loc[case, "profit_above_cash"], atol=1e-5, rtol=0)
        comparison.append({"Policy": NAMES[case], "Positive pairs": int(row.positive_pairs),
            "Negative pairs": int(row.negative_pairs), "Net pair P&L ($)": row.net_pair_pnl,
            "Shared financing ($)": row.shared_financing_vs_cash,
            "Top 3 share of positive profit (%)": row.top3_share_positive_profit*100})
        for side, subset in [("Highest", own_pairs.head(5)), ("Lowest", own_pairs.tail(5).iloc[::-1])]:
            for r in subset.itertuples():
                ranking.append({"Policy": NAMES[case], "Group": side, "Pair": r.label,
                                "Stable pair ID": r.pair, "Net pair P&L ($)": r.net_before_shared_financing})
    lines = ["Actual pair attribution in 2022-2025", table(pd.DataFrame(comparison)), "",
          "Pair results include actual option and stock-hedge P&L, allocated trading/borrow costs, and dividends.",
          "Attribution includes positions carried into 2022, so these pair counts can exceed new-entry pair counts.",
        "Shared financing relative to cash is shown separately; pair profit plus shared financing reconciles",
        "to each funded book's profit above cash. A profitable pair count is not a count of independent bets.",
        "Top-three concentration is the share of all positive pair profits, not the share of net profit.", "",
        "Highest and lowest pair contributions", table(pd.DataFrame(ranking)), "",
        "Ticker names are display labels from the bundled roster/prior audit; stable pair IDs identify the data."]
    unavailable = [case for case in ORDER if case not in successful]
    if unavailable:
        lines += ["Attribution unavailable for: " + ", ".join(NAMES[c] for c in unavailable) + "."]
        if status:
            lines.extend(f"  {r['case']}: {r['status']}" for r in status["cases"] if r["case"] in unavailable)
    return lines


def selection_coverage(report):
    """How often monthly selection had any genuine choice among candidates."""
    scores = pd.read_csv(report / "selection_scores.csv")
    scores = scores.loc[scores.case.eq("selected_rolling")].copy()
    if scores.empty:
        raise ValueError("Rolling selection-score audit is missing")
    scores["asof"] = pd.to_datetime(scores["asof"])
    # Accept bool or CSV string representation; never treat the string False as True.
    scores["eligible_flag"] = scores.eligible.astype(str).str.lower().isin(["true", "1"])
    if scores.duplicated(["asof", "policy"]).any():
        raise ValueError("Duplicate candidate selection audit rows")
    monthly = scores.groupby("asof").eligible_flag.sum().rename("eligible_candidates").to_frame()
    baseline = scores.loc[scores.policy.eq("q20")].set_index("asof").eligible_flag
    if not baseline.index.equals(monthly.index):
        baseline = baseline.reindex(monthly.index)
    if baseline.isna().any():
        raise ValueError("Baseline eligibility audit is missing for some selection months")
    monthly["baseline_eligible"] = baseline.astype(bool)
    rows = []
    for label, own in [("All signal months", monthly),
                       ("2022-25 signal months", monthly.loc[monthly.index >= "2022-01-01"])]:
        if own.empty:
            continue
        rows.append({"Period": label, "Months": len(own),
            "0 eligible: fallback": int(own.eligible_candidates.eq(0).sum()),
            "1 eligible: no ranking choice": int(own.eligible_candidates.eq(1).sum()),
            "2+ eligible: ranking possible": int(own.eligible_candidates.ge(2).sum()),
            "20% rule eligible": int(own.baseline_eligible.sum())})
    coverage = pd.DataFrame(rows)
    monthly.reset_index().to_csv(report / "monthly_selection_coverage.csv", index=False)
    coverage.to_csv(report / "selection_coverage_summary.csv", index=False)
    return ["How much choice did monthly selection actually have?", table(coverage), "",
        "Zero eligible candidates means the 20% fallback was used. With exactly one eligible candidate,",
        "the rule had no competing eligible threshold to rank. Only months with two or more eligible",
        "candidates permit a comparison of historical Sharpe across thresholds. The entry-count minimum",
        "and available history can therefore explain an unchanged cutoff without establishing its optimality."]


def result_interpretation(later, intervals, selections, protocol):
    baseline = later.loc["q20"]
    fixed = later.loc["selected_fixed"]
    rolling = later.loc["selected_rolling"]
    context = later.loc["selected_context"]
    lines = ["What the completed experiment shows",
        f"The learned fixed cutoff produced later Sharpe {fixed.annualized_excess_sharpe:.3f}, versus "
        f"{baseline.annualized_excess_sharpe:.3f} for the original rule. Profit above cash was "
        f"${fixed.profit_above_cash:,.0f} versus ${baseline.profit_above_cash:,.0f}; entries were "
        f"{int(fixed.entries)} versus {int(baseline.entries)}; average reserve was "
        f"{fixed.average_reserve_pct:.3f}% versus {baseline.average_reserve_pct:.3f}% of NAV."]
    if (fixed.annualized_excess_sharpe > baseline.annualized_excess_sharpe and
        fixed.profit_above_cash < baseline.profit_above_cash and fixed.entries < baseline.entries and
        fixed.average_reserve_pct < baseline.average_reserve_pct):
        lines += ["Its higher Sharpe accompanies fewer trades, lower deployed reserve, and less total profit.",
            "This is a risk/return tradeoff; the higher Sharpe does not mean the strategy earned more money."]
    lines += [f"Maximum cash-relative drawdown was {abs(fixed.max_cash_relative_drawdown_pct):.3f}% "
              f"versus {abs(baseline.max_cash_relative_drawdown_pct):.3f}% for the original rule.",
        f"Monthly selection produced Sharpe {rolling.annualized_excess_sharpe:.3f} "
        f"(difference {rolling.annualized_excess_sharpe-baseline.annualized_excess_sharpe:+.3f}); "
        f"profit above cash was ${rolling.profit_above_cash:,.0f}.",
        f"The VIX-dependent rule produced Sharpe {context.annualized_excess_sharpe:.3f} "
        f"(difference {context.annualized_excess_sharpe-baseline.annualized_excess_sharpe:+.3f}); "
        f"profit above cash was ${context.profit_above_cash:,.0f}."]
    own = intervals.loc[intervals.period.eq("evaluation")]
    if ((own.lower95 <= 0) & (own.upper95 >= 0)).all():
        lines += ["All later Sharpe-difference intervals include zero at both tested block lengths.",
            "This experiment has not statistically established an improvement over the original rule."]
    q = protocol["policies"][selections["selected_fixed"]["policy"]]["q_low"]
    grid = [protocol["policies"][key]["q_low"] for key in protocol["fixed_family"]]
    if q == min(grid) or q == max(grid):
        edge = "lower" if q == min(grid) else "upper"
        lines += [f"The chosen {q:.0%} cutoff lies at the {edge} boundary of the tested grid.",
            "The experiment does not establish a globally optimal cutoff or rank untested thresholds."]
    lines += ["The later history was previously inspected. These findings are exploratory, including the",
              "candidate selected on earlier data; a frozen future test is still needed for confirmation."]
    return lines


def create_report(report=DEFAULT_REPORT, artifacts=DEFAULT_ARTIFACTS, budget=None):
    budget = budget or Budget(duty_cycle=.2, rss_limit_mib=384)
    started = time.monotonic()
    budget.check()
    report, artifacts = Path(report), Path(artifacts)
    (protocol, periods, annual, intervals, candidates, rolling, selections,
     navs, carry) = read_completed(report, artifacts)
    budget.rest(started)
    started = time.monotonic()
    later = periods.loc[periods.period.eq("evaluation")].set_index("case").loc[ORDER]
    earlier = periods.loc[periods.period.eq("validation")].set_index("case").loc[ORDER]
    displayed = pd.DataFrame({
        "Policy": [NAMES[c] for c in ORDER],
        "2018-21 Sharpe": earlier.annualized_excess_sharpe.to_numpy(),
        "2022-25 Sharpe": later.annualized_excess_sharpe.to_numpy(),
        "Later profit above cash ($)": later.profit_above_cash.to_numpy(),
        "Later entries": later.entries.to_numpy(),
        "Later entry pairs": later.active_entry_pairs.to_numpy(),
        "Later average reserve (%)": later.average_reserve_pct.to_numpy(),
        "Later maximum drawdown (%)": later.max_cash_relative_drawdown_pct.to_numpy(),
    })
    fixed_keys = protocol["fixed_family"]
    context_keys = protocol["context_family"]
    candidate_table = candidates.pivot(index="case", columns="period", values="annualized_excess_sharpe")
    fixed_table = candidate_table.loc[fixed_keys].reset_index()
    fixed_table.insert(1, "tail_percent", [100*protocol["policies"][k]["q_low"] for k in fixed_keys])
    context_table = candidate_table.loc[context_keys].reset_index()
    context_table.insert(1, "low_VIX_tail_percent", [100*protocol["policies"][k]["q_low"] for k in context_keys])
    context_table.insert(2, "high_VIX_tail_percent", [100*(protocol["policies"][k]["q_high"]
                          if protocol["policies"][k]["q_high"] is not None else protocol["policies"][k]["q_low"])
                          for k in context_keys])
    for view, case in [(fixed_table, "selected_fixed"), (context_table, "selected_context")]:
        audit = pd.DataFrame(selections[case]["candidates"]).set_index("policy")
        view["earlier_entries"] = view.case.map(audit.entries)
        view["selection_eligible"] = view.case.map(audit.eligible)
        view["eligibility_reason"] = view.case.map(audit.reason)
        view["chosen_before_2022"] = view.case.eq(selections[case]["policy"])
    fixed_table = fixed_table[["case", "tail_percent", "validation", "earlier_entries",
                               "selection_eligible", "eligibility_reason", "chosen_before_2022", "evaluation"]]
    context_table = context_table[["case", "low_VIX_tail_percent", "high_VIX_tail_percent",
        "validation", "earlier_entries", "selection_eligible", "eligibility_reason", "chosen_before_2022", "evaluation"]]
    distribution = []
    for label, own in [("All 90 signal months", rolling),
                       ("2022-25 signal months", rolling.loc[rolling.month >= "2022-01"])]:
        for key, group in own.groupby("policy"):
            distribution.append(dict(period=label, policy=key, tail_percent=float(group["tail"].iloc[0])*100,
                months=len(group), fraction_percent=100*len(group)/len(own),
                fallback_months=int(group.status.eq("fallback").sum()), proposals=int(group.proposals.sum())))
    distribution = pd.DataFrame(distribution)
    distribution.to_csv(report / "rolling_distribution.csv", index=False)
    ci = intervals.loc[intervals.period.eq("evaluation"), ["case", "block_sessions", "point", "lower95", "upper95"]]
    annual_table = annual.loc[annual.year >= 2022].pivot(index="year", columns="case",
        values="annualized_excess_sharpe").reindex(columns=ORDER).rename(columns=NAMES).reset_index()
    annual_pnl = annual.loc[annual.year >= 2022].pivot(index="year", columns="case",
        values="profit_above_cash").reindex(columns=ORDER).rename(columns=NAMES).reset_index()
    candidate_annual = pd.read_csv(report / "candidate_annual.csv")
    baseline_annual = candidate_annual.loc[candidate_annual.case.eq("q20") &
        candidate_annual.year.between(2018, 2021)].set_index("year")
    training_years = []
    for case in ["selected_fixed", "selected_context"]:
        key = selections[case]["policy"]
        own = candidate_annual.loc[candidate_annual.case.eq(key) &
                                  candidate_annual.year.between(2018, 2021)].sort_values("year")
        for row in own.itertuples():
            reference = baseline_annual.loc[row.year]
            training_years.append({"Selection": NAMES[case], "Candidate": key, "Year": row.year,
                "Candidate Sharpe": row.annualized_excess_sharpe,
                "Candidate profit above cash ($)": row.profit_above_cash, "Candidate entries": row.entries,
                "20% rule Sharpe": reference.annualized_excess_sharpe,
                "20% rule profit above cash ($)": reference.profit_above_cash,
                "20% rule entries": reference.entries,
                "Profit difference ($)": row.profit_above_cash-reference.profit_above_cash})
    training_years = pd.DataFrame(training_years)
    training_years.to_csv(report / "selected_candidate_earlier_annual.csv", index=False)
    lines = ["Direct threshold search: completed funded experiment", "",
        "Question: can a cutoff learned from prior funded returns improve opportunity selection?",
        "This experiment changes the volatility-rank entry threshold and holds position sizing at 100%.",
        "It preserves the historical profitability/count screens; it does not replace the entire proposal pipeline.", "",
        "Choices made using 2018-2021 only:"]
    for case in ["selected_fixed", "selected_context"]:
        selection = selections[case]
        record = next(r for r in selection["candidates"] if r["policy"] == selection["policy"])
        score = record["funded_excess_sharpe"]
        lines += [f"  {NAMES[case]}: {policy_text(selection['policy'], protocol)}.",
                  f"  Selection status: {selection['status']}; last available return: {selection['latest_return']}; "
                  f"candidate validation Sharpe: {score:.3f}." if score is not None else
                  f"  Selection status: {selection['status']}; baseline fallback because no candidate qualified."]
    lines += ["", "Funded results", table(displayed), "",
        *result_interpretation(later, intervals, selections, protocol), "",
        "The fixed and VIX-dependent deployment books follow the original rule through 2021, then change",
        "new proposals in 2022 while carrying their actual baseline holdings and NAV. Their 2018-2021",
        "deployment performance therefore equals the baseline; the separate candidate results below",
        "are the data used to choose them. The monthly rule begins adapting in 2018 and carries its own",
        "holdings into 2022. No selected book is made by stitching candidate return series.", "",
        "Verified pre-2022 carry into fixed/context deployments", table(pd.DataFrame(carry)), "",
        "Paired difference in later Sharpe versus the original 20% rule", table(ci), "",
        *[interval_interpretation(case, intervals, np.array_equal(
            navs[case].loc["2022":, "excess_return"].to_numpy(),
            navs["q20"].loc["2022":, "excess_return"].to_numpy())) for case in ORDER[1:]],
        "These are conditional circular-block percentile intervals using 2,000 paired resamples.",
        "They are not corrected for choosing policies, testing several policies, or earlier research.", "",
        "Every fixed candidate: earlier selection period and later exploratory period",
        table(fixed_table), "",
        "Locked selection requires at least 252 measured return sessions and 20 entered trades in 2018-2021.",
        "A candidate with a higher earlier Sharpe can be ineligible because it has too few entries.",
        "Every contextual candidate (a whole funded policy, not separately scored VIX subsets)",
        table(context_table), "",
        "Did the chosen candidates' earlier advantage persist across years?", table(training_years), "",
        "These are the counterfactual candidate books used in selection, compared with the 20% rule",
        "year by year. They expose concentration in 2020 or another year without changing the selected policy.",
        "An annual Sharpe difference is descriptive and does not add up to the full-period Sharpe difference.", "",
        "Monthly threshold distribution", table(distribution), "",
        *selection_coverage(report), "",
        "Monthly selection uses the preceding 18 calendar months, strictly before each signal month.",
        "Candidates need at least 252 measured return sessions and 20 entered trades in that window.",
        "If none qualify, the cutoff is 20%. Ties favor proximity to 20%, then the smaller tail cutoff.",
        "The contextual policy uses the already-lagged VIX feature: 20 belongs to the high-VIX regime.", "",
        "Annual stability: funded excess Sharpe", table(annual_table), "",
        "Annual stability: profit above cash ($)", table(annual_pnl), "",
        "2025 ends on August 29, with new signals ending June 30; it is not a full calendar year.",
        "Entry-pair counts measure breadth, not independent observations or equal pair contribution.", "",
        *pair_findings(report, periods), "",
        "Interpretation and limits",
        "The selection objective here is actual marked daily excess-return Sharpe from funded candidate",
        "ledgers, including cash days, transaction costs, risk limits, overlapping positions, and carry.",
        "It is not the earlier regression experiment's signal-date lifetime-payoff surrogate.",
        "A better later result would support this tested rule family; it would not establish a globally",
        "optimal threshold or show that arbitrary additional flexibility will help.",
        "The directional 91/365-calendar-day historical profit screens remain >0, with counts >=21/63.",
        "Their cached historical memory retains the archived $0.65 option-commission convention while",
        "the funded ledgers use $0.03. Inherited retrospective roster and formation-calendar limitations",
        "remain. The 2022-2025 history was already examined before this experiment, so it is exploratory.",
        "Serial dependence, shared stocks, and overlapping trades limit the effective sample size.",
        "Full candidate tables are disclosed rather than promoting the best later-period result.", "",
        "Files: candidate_periods.csv, candidate_annual.csv, selection_scores.csv, rolling_schedule.csv,",
        "rolling_distribution.csv, monthly_selection_coverage.csv, selection_coverage_summary.csv,",
        "selected_candidate_earlier_annual.csv,",
        "funded/periods.csv, funded/annual.csv, funded/sharpe_intervals.csv,",
        "funded/pair_admissions.csv, protocol.json, completion.json, and comparison.png.", ""]
    (report / "Findings.txt").write_text("\n".join(lines), encoding="utf-8")
    budget.rest(started)
    started = time.monotonic()
    budget.check()
    draw_plot(report, protocol, candidates, navs, rolling, intervals, selections, budget)
    budget.rest(started)
    return dict(findings=str(report / "Findings.txt"), plot=str(report / "comparison.png"),
                peak_rss_mib=budget.peak, duty_cycle=.2, rss_limit_mib=384)


def draw_plot(report, protocol, candidates, navs, rolling, intervals, selections, budget):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    budget.check()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    budget.check()
    ax = axes[0, 0]
    keys = protocol["fixed_family"]
    for period, name, color in [("validation", "2018-2021: selection", "#586574"),
                                ("evaluation", "2022-2025: exploratory", "#2274a5")]:
        values = candidates.loc[candidates.period.eq(period)].set_index("case").loc[keys]
        ax.plot([100*protocol["policies"][k]["q_low"] for k in keys],
                values.annualized_excess_sharpe, "o-", label=name, color=color)
    selected = protocol["policies"][selections["selected_fixed"]["policy"]]["q_low"]*100
    ax.axvline(selected, color="#e07a26", linestyle=":", label="Fixed cutoff chosen before 2022")
    ax.set(xlabel="Tail cutoff (%)", ylabel="Funded excess Sharpe", title="All fixed cutoff candidates")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for case in ORDER:
        nav = navs[case]
        absolute = nav.nav-nav.cash_benchmark
        before = absolute.loc[:"2021-12-31"].iloc[-1]
        ax.plot(absolute.loc["2022":].index, (absolute.loc["2022":]-before)/1000,
                label=NAMES[case], color=COLORS[case], linewidth=1.5)
    ax.axhline(0, color="#dddddd", linewidth=.8)
    ax.set(ylabel="Profit above cash since 2022 ($ thousands)", title="Continuous funded deployment books")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for position, case in enumerate(ORDER[1:]):
        for block, offset, marker in [(63, -.12, "o"), (126, .12, "s")]:
            row = intervals.loc[intervals.case.eq(case) & intervals.period.eq("evaluation") &
                                intervals.block_sessions.eq(block)].iloc[0]
            # A percentile interval need not enclose its original-sample estimate.
            ax.hlines(position+offset, row.lower95, row.upper95, color=COLORS[case], linewidth=2)
            ax.plot(row.point, position+offset, marker=marker, color=COLORS[case],
                    label=f"{block}-session blocks" if position == 0 else None)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(range(3), [NAMES[c] for c in ORDER[1:]])
    ax.set(xlabel="Later Sharpe difference versus original 20% rule",
           title="Conditional 95% intervals; selection unadjusted")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    dates = pd.to_datetime(rolling.month)
    ax.step(dates, rolling["tail"], where="post", color=COLORS["selected_rolling"], linewidth=1.5)
    fallback = rolling.status.eq("fallback")
    ax.scatter(dates[fallback], rolling.loc[fallback, "tail"], marker="x", color="#777777",
               s=14, label="Insufficient history: 20% fallback")
    ax.axhline(.20, color="#999999", linestyle=":", linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(ylabel="Selected tail cutoff", title="Monthly decisions from prior 18-month books")
    ax.legend(fontsize=8)
    fig.suptitle("Direct opportunity-threshold search | fixed sizing | previously inspected history", fontsize=15)
    budget.check()
    fig.savefig(report / "comparison.png", dpi=160)
    budget.check()
    plt.close(fig)
    budget.check()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    print(json.dumps(create_report(args.report, args.artifacts), indent=2))


if __name__ == "__main__":
    main()
