"""Small replay dependencies; frozen financial formulas and read-only caches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sharpe(r):
    return float(np.sqrt(252)*r.mean()/r.std(ddof=1)) if r.std(ddof=1)>1e-12 else 0.


def mark_book(nav,rates,capital):
    # Identical to run_aligned_contract_learning.mark_book.
    elapsed=nav.index.to_series().diff().dt.days.fillna(0)/365
    cash_return=rates.reindex(nav.index).shift(1).fillna(0)*elapsed
    nav['cash_benchmark']=capital*(1+cash_return).cumprod()
    nav['excess_return']=nav.return_-cash_return
    return nav


def metrics(nav,capital):
    # Identical to run_p25_capital_sensitivity.metrics; avoid its training imports.
    years=(nav.index[-1]-nav.index[0]).days/365.25
    relative=nav.nav/nav.cash_benchmark
    elapsed=nav.index.to_series().diff().dt.days.fillna(0)/365
    borrowing_spread=-nav.cash.shift(1).clip(upper=0).fillna(0)*.01*elapsed
    return dict(funded_equity=capital,annualized_excess_sharpe=sharpe(nav.excess_return.iloc[1:]),
        funded_cagr_pct=100*((nav.nav.iloc[-1]/capital)**(1/years)-1),
        relative_to_cash_cagr_pct=100*(relative.iloc[-1]**(1/years)-1),
        net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),
        maximum_nav_drawdown_pct=100*float((nav.nav/nav.nav.cummax()-1).min()),
        maximum_relative_to_cash_drawdown_pct=100*float((relative/relative.cummax()-1).min()),
        peak_reserve_fraction=float((nav.capital_reserve/nav.nav).max()),
        peak_stress_fraction=float((nav.stress_loss/nav.nav).max()),
        peak_gross_hedge_multiple=float((nav.gross_hedge_notional/nav.nav).max()),
        post_entry_risk_breach_days=int(nav.post_trade_risk_breach.sum()),
        pair_risk_breach_days=int(nav.pair_risk_breaches.gt(0).sum()),
        negative_cash_days=int(nav.cash.lt(0).sum()),borrowing_spread_cost=float(borrowing_spread.sum()))


def annual_metrics(nav,case,capital):
    rows=[]
    previous_nav=pd.Series(nav.nav.shift(1).fillna(capital),index=nav.index)
    previous_cash=nav.cash_benchmark.shift(1).fillna(capital)
    for year,g in nav.groupby(nav.index.year):
        funded=float(g.nav.iloc[-1]/previous_nav.loc[g.index[0]]-1)
        cash=float(g.cash_benchmark.iloc[-1]/previous_cash.loc[g.index[0]]-1)
        returns=g.excess_return.iloc[1:] if g.index[0]==nav.index[0] else g.excess_return
        rows.append(dict(case=case,year=int(year),funded_return_pct=100*funded,
                         return_above_cash_pp=100*(funded-cash),annualized_excess_sharpe=sharpe(returns)))
    return rows


def signature(path):
    stat=Path(path).stat()
    return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def cache_metadata(folder):
    path=folder/'market_cache'/'metadata.json'
    if not path.exists():return None
    meta=json.loads(path.read_text())
    if any(signature(p)!=wanted for p,wanted in meta['sources'].items()):
        raise RuntimeError('Market source changed; rebuild the exact volatility cache')
    return meta


class ScalarVariance:
    """Scalar .at API backed by an unmodified, read-only mapped numpy array."""
    def __init__(self,values,dates,names):
        self.values=values;self.dates=dates;self.names=names;self.at=self

    def __getitem__(self,key):
        date,sid=key
        return self.values[self.dates[date],self.names[sid]]


def load_variance(market,folder):
    meta=cache_metadata(folder)
    if meta is None:raise RuntimeError('Prepare the exact disk-backed variance cache first')
    dates={pd.Timestamp(d):i for i,d in enumerate(meta['dates'])}
    names={int(sid):i for i,sid in enumerate(meta['secids'])}
    return {half:ScalarVariance(np.load(folder/'market_cache'/f'variance_{half}.npy',mmap_mode='r'),dates,names)
            for half in [20,60,252]}


def load_protocol(folder):
    from scripts.option_flex.contract_engine import Settings
    data=json.loads((folder/'runtime_protocol.json').read_text())
    for name,wanted in data['source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=wanted:
            raise RuntimeError(f'Frozen configuration source changed: {name}')
    return Settings(**data['settings']),data['corporate_events']
