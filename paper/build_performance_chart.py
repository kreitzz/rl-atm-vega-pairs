"""Render the report's performance chart from saved main-branch accounts.

Run with the mfe_wrds Python environment; no model fitting or replay is performed.
"""
from pathlib import Path
import io
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures"
START = pd.Timestamp("2022-01-03")
STOP = pd.Timestamp("2025-08-29")
SELECTED_MEMBER = (
    "artifacts/contracts/p25_validation_split/replays/"
    "screened_selected/daily_nav.parquet"
)
BASELINE = (
    ROOT / "data/submission/artifacts/contracts/p25_capital_sensitivity/"
    "replays/10m_fixed100/daily_nav.parquet"
)


def curve(nav, case, expected):
    """Use the same period-dollar definition as period_metrics in main."""
    nav = nav.copy()
    nav.index = pd.to_datetime(nav.index)
    assert nav.index.is_monotonic_increasing and not nav.index.has_duplicates
    previous = nav.loc[nav.index < START].iloc[-1]
    previous_date = nav.index[nav.index < START][-1]
    part = nav.loc[START:STOP]
    assert previous_date == pd.Timestamp("2021-12-31")
    assert part.index[0] == START and part.index[-1] == STOP
    profit = (part.nav - part.cash_benchmark) - (
        previous.nav - previous.cash_benchmark
    )
    assert abs(float(profit.iloc[-1]) - expected) < 0.01
    plotted = pd.concat([pd.Series([0.0], index=[previous_date]), profit])
    plotted.name = case
    return plotted


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    periods = pd.read_csv(ROOT / "reports/p25_validation_split/periods.csv")
    periods = periods.loc[periods.period.eq("retrospective_holdout")].set_index("case")
    with zipfile.ZipFile(ROOT / "data/p25_validation_split_experiment.zip") as archive:
        selected = pd.read_parquet(io.BytesIO(archive.read(SELECTED_MEMBER)))
    constant = pd.read_parquet(BASELINE)
    learned = curve(selected, "tree_cem", periods.loc["screened_selected", "profit_above_cash"])
    fixed = curve(constant, "constant_full", periods.loc["screened_fixed100", "profit_above_cash"])
    assert learned.index.equals(fixed.index)
    paths = pd.concat([learned, fixed], axis=1)
    assert paths.notna().all().all()
    paths.index.name = "date"
    paths.to_csv(OUTPUT / "performance_above_cash.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.19, top=0.92)
    ax.plot(learned.index, learned / 1000, color="#167D8D", linewidth=1.9, label="Tree + CEM")
    ax.plot(fixed.index, fixed / 1000, color="#17364D", linewidth=1.6,
            linestyle=(0, (5, 2.5)), label="Constant 100%")
    ax.axhline(0, color="#9BAAB5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E3E9ED", linewidth=0.7)
    ax.set_ylabel("Cumulative profit above cash ($ thousands)")
    ax.set_xlabel("Evaluation year")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp("2021-12-31"), STOP + pd.Timedelta(days=175))
    ax.set_xticks(pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]))
    ax.set_ylim(-15, 280)
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.spines[["top", "right"]].set_visible(False)
    for name in ["left", "bottom"]:
        ax.spines[name].set_color("#B8C4CD")
        ax.spines[name].set_linewidth(0.7)
    ax.tick_params(length=3, color="#B8C4CD")
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.5)
    for values, color in [(learned, "#167D8D"), (fixed, "#17364D")]:
        ax.scatter([STOP], [values.iloc[-1] / 1000], s=15, color=color, zorder=3)
        ax.annotate(f"${values.iloc[-1]:,.0f}", (STOP, values.iloc[-1] / 1000),
                    xytext=(7, 0), textcoords="offset points", va="center",
                    color=color, fontsize=9, fontweight="bold")
    fig.savefig(OUTPUT / "performance_above_cash.pdf")
    fig.savefig(OUTPUT / "performance_above_cash.png", dpi=180)
    plt.close(fig)
    (OUTPUT / "performance_above_cash_sources.json").write_text(json.dumps({
        "selected": "data/p25_validation_split_experiment.zip!" + SELECTED_MEMBER,
        "constant": str(BASELINE.relative_to(ROOT)).replace("\\", "/"),
        "reconciliation": "reports/p25_validation_split/periods.csv",
        "start": str(START.date()), "stop": str(STOP.date()),
        "rebased_at": "2021-12-31",
        "definition": "Change in NAV minus cash benchmark since the preceding close",
        "account_reset": False,
        "tree_cem_profit": float(learned.iloc[-1]),
        "constant_full_profit": float(fixed.iloc[-1]),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Chart verified: tree/CEM ${learned.iloc[-1]:,.2f}; constant full ${fixed.iloc[-1]:,.2f}.")


if __name__ == "__main__":
    main()
