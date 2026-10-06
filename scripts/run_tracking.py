"""Run the rule-based tracking baselines and write the tracking-vs-cost results.

Periods follow the tracking plan: 2010-2020 (fit), 2021-2022 (tune), 2023-2025 (test).
The baselines have nothing to fit; the band is the only choice, so the frontier is shown on
2010-2022 and the frozen comparison on 2023-2025.

Usage:  python -m scripts.tracking_data --pair AMD_NVDA      (once)
        python -m scripts.run_tracking
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.tracking_baselines import Band, Settings, metrics, pair_daily, simulate

TICKERS = ('AMD', 'NVDA')
PERIODS = {'2010-2020': ('2010', '2020'), '2021-2022': ('2021', '2022'),
           '2010-2022': ('2010', '2022'), '2023-2025': ('2023', '2025')}
CONFIGS = {
    'single_daily':          ('single', Band(), False),
    'single_engine_like':    ('single', Band(9, .15, .25), False),
    'single_wide':           ('single', Band(15, .25, .5), False),
    'single_monthly_wide':   ('single', Band(15, .25, .5), True),
    'interp_daily':          ('interp', Band(), False),
    'interp_band_3d_5':      ('interp', Band(3, .05, .10), False),
    'interp_band_7d_10':     ('interp', Band(7, .10, .20), False),
    'interp_band_10d_15':    ('interp', Band(10, .15, .30), False),
    'interp_band_15d_20':    ('interp', Band(15, .20, .50), False),
    'interp_monthly_daily':  ('interp', Band(), True),
    'interp_monthly_7d_10':  ('interp', Band(7, .10, .20), True),
    'interp_monthly_10d_15': ('interp', Band(10, .15, .30), True),
    'interp_monthly_15d_20': ('interp', Band(15, .20, .50), True),
}


def run_one(job):
    name, ticker, chain_path, index_path, settings = job
    chain, index = pd.read_pickle(chain_path), pd.read_pickle(index_path)
    method, band, monthly = CONFIGS[name]
    d = simulate(chain, index, ticker, method, band, settings, monthly)
    d['config'] = name
    return d


def spot_sensitivity(daily, returns):
    """Share of daily tracking error explained by the stock's own move (skew / moneyness)."""
    out = {}
    for tk, g in daily.groupby('ticker'):
        r = returns[tk].shift(-1).reindex(g.date).to_numpy() * 100      # close t -> t+1, %
        y = g.tracking_error.to_numpy()
        ok = np.isfinite(r) & np.isfinite(y)
        b = np.polyfit(r[ok], y[ok], 1)
        res = y[ok] - np.polyval(b, r[ok])
        out[tk] = {'corr_with_stock_return': float(np.corrcoef(r[ok], y[ok])[0, 1]),
                   'vol_points_per_1pct_move': float(b[0]),
                   'te_rmse': float(np.sqrt((y[ok]**2).mean())),
                   'te_rmse_after_removing_stock_move': float(np.sqrt((res**2).mean()))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('artifacts/tracking'))
    p.add_argument('--returns', type=Path, default=Path('data/wrds_q/returns.csv'))
    p.add_argument('--out', type=Path, default=Path('reports/tracking'))
    p.add_argument('--target-vega', type=float, default=Settings.target_vega)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    s = Settings(target_vega=a.target_vega)
    jobs = [(n, t, a.data/'chain.pkl', a.data/'index.pkl', s) for n in CONFIGS for t in TICKERS]
    with ProcessPoolExecutor(a.workers) as ex:
        daily = pd.concat(list(ex.map(run_one, jobs)), ignore_index=True)
    a.out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(a.data/'daily.csv', index=False)

    rows = []
    for name in CONFIGS:
        books = {t: daily[(daily.config == name) & (daily.ticker == t)].reset_index(drop=True)
                 for t in TICKERS}
        books['AMD-NVDA'] = pair_daily(books['AMD'], books['NVDA'])
        for book, d in books.items():
            for period, (lo, hi) in PERIODS.items():
                x = d[(d.date >= lo) & (d.date < str(int(hi) + 1))]
                rows.append({'config': name, 'book': book, 'period': period, **metrics(x)})
    table = pd.DataFrame(rows)
    table.to_csv(a.out/'metrics.csv', index=False)

    returns = pd.read_csv(a.returns, parse_dates=['date']).pivot(index='date', columns='ticker', values='retx')
    base = daily[daily.config == 'interp_daily']
    diag = {period: spot_sensitivity(base[(base.date >= lo) & (base.date < hi)], returns)
            for period, (lo, hi) in {'2010-2022': ('2010', '2023'), '2017-2022': ('2017', '2023'),
                                     '2023-2025': ('2023', '2026')}.items()}
    json.dump({'settings': vars(s), 'configs': {k: [m, vars(b), mo] for k, (m, b, mo) in CONFIGS.items()},
               'spot_sensitivity_interp_daily': diag},
              open(a.out/'summary.json', 'w'), indent=2, default=str)
    cols = ['te_daily_rmse', 'te_median_abs', 'te_21d_rmse', 'index_21d_sd', 'r2', 'beta', 'level_error_mae', 'cost_per_year', 'trade_day_fraction']
    for period in ('2010-2022', '2023-2025'):
        print(f'\n== {period} ==')
        print(table[table.period == period].set_index(['book', 'config'])[cols].round(3).to_string())
    print(json.dumps(diag, indent=2))


if __name__ == '__main__':
    main()
