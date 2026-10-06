"""Bounded descriptive checks of name-relative regimes and RV/P&L agreement."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.diagnose_current_trade_quality import REPORT, EMPLOYEE, CHECKPOINT
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar
from scripts.run_broad_contract_replay import ROOT


def within_pair(h, feature):
    parts = []
    for pair, g in h[['pair', feature, 'net_per_vega']].dropna().groupby('pair'):
        if len(g)<5 or g[feature].nunique()<2 or g.net_per_vega.nunique()<2: continue
        x = g[feature].rank(pct=True); y = g.net_per_vega.rank(pct=True)
        parts.append(pd.DataFrame(dict(x=x-x.mean(), y=y-y.mean(), w=1/len(g))))
    if not parts: return np.nan, 0
    p = pd.concat(parts)
    denominator = np.sqrt((p.w*p.x*p.x).sum()*(p.w*p.y*p.y).sum())
    return float((p.w*p.x*p.y).sum()/denominator) if denominator>0 else np.nan, len(parts)


def main():
    c = pd.read_csv(REPORT/'cohorts.csv', parse_dates=['cohort', 'entry_date', 'exit_date'])
    c = c[c.model.eq('rl_baseline')].copy()
    l = pd.read_csv(REPORT/'legs.csv', parse_dates=['cohort', 'entry_date', 'exit_date'])
    l = l[l.model.eq('rl_baseline')].copy()
    paths = [Path(__file__), REPORT/'cohorts.csv', REPORT/'legs.csv', REPORT/'audit.json', CHECKPOINT]
    cal = load_calendar(ROOT)
    for pair, g in c.groupby('pair'):
        source = EMPLOYEE/pair/'states.parquet'; paths.append(source); p = pd.read_parquet(source)
        daily = p.iv_amd.diff().rolling(126, min_periods=63).corr(p.iv_nvda.diff())
        history = daily.shift(1).rolling(252, min_periods=126)
        correlation_z = (daily-history.mean())/history.std().where(lambda x:x>1e-8)
        wedge = daily-p.correlation; past = wedge.shift(1).rolling(252, min_periods=126)
        wedge_z = (wedge-past.mean())/past.std().where(lambda x:x>1e-8)
        c.loc[g.index, 'iv_change_corr_history_z'] = correlation_z.reindex(pd.DatetimeIndex(g.cohort)).to_numpy()
        c.loc[g.index, 'iv_stock_corr_wedge_history_z'] = wedge_z.reindex(pd.DatetimeIndex(g.cohort)).to_numpy()
    features = ['iv_change_corr_history_z', 'iv_stock_corr_wedge_history_z', 'cheap_iv_gap',
        'signal_gamma_ratio_long_short', 'signal_dte_gap', 'memory_91d', 'memory_365d', 'short_earnings_before_expiry']
    ranks = []
    for period, g in [('all', c), ('early_complete', c[c.exit_date<'2022-01-01']), ('later', c[c.entry_date>='2022-01-01'])]:
        for feature in features:
            r, pairs = within_pair(g, feature)
            ranks.append(dict(period=period, feature=feature, within_pair_spearman=r, pairs=pairs,
                observed_trades=int(g[feature].notna().sum()), episodes=g[g[feature].notna()].episode.nunique()))
    pd.DataFrame(ranks).to_csv(REPORT/'relative_regime_rank.csv', index=False)
    # One row per actual leg. Forward RV is an outcome explanation, never an input.
    merged = l.merge(c[['pair', 'cohort', 'long_forward_rv', 'short_forward_rv']], on=['pair', 'cohort'], validate='many_to_one')
    merged['forward_rv'] = np.where(merged.role.eq('long'), merged.long_forward_rv, merged.short_forward_rv)
    sessions = cal.get_indexer(merged.exit_date)-cal.get_indexer(merged.entry_date)
    assert (sessions>0).all()
    variance = (merged.forward_rv/100)**2*sessions/252-(merged.entry_iv/100)**2*merged.entry_dte/365
    merged['forward_horizon_variance_edge'] = np.where(merged.role.eq('long'), variance, -variance)
    merged['proxy_agrees_with_gross_pnl'] = np.sign(merged.forward_horizon_variance_edge)==np.sign(merged.gross_hedged_pnl)
    merged[['pair', 'label', 'name', 'role', 'cohort', 'entry_date', 'exit_date', 'entry_iv', 'forward_rv',
        'forward_horizon_variance_edge', 'gross_hedged_pnl', 'net_before_shared_financing', 'proxy_agrees_with_gross_pnl']].to_csv(REPORT/'rv_proxy_agreement.csv', index=False)
    confidence = c.assign(size=c.action.abs()).groupby('size').agg(trades=('cohort', 'size'), episodes=('episode', 'nunique'),
        mean_net_per_vega=('net_per_vega', 'mean'), net_before_shared_financing=('net_before_shared_financing', 'sum'),
        win_rate=('net_per_vega', lambda x:(x>0).mean())).reset_index()
    confidence.to_csv(REPORT/'sizing_quality.csv', index=False)
    directions = c.groupby(['label', 'long_name', 'short_name']).agg(trades=('cohort', 'size'), episodes=('episode', 'nunique'),
        net_before_shared_financing=('net_before_shared_financing', 'sum'), mean_net_per_vega=('net_per_vega', 'mean'),
        win_rate=('net_per_vega', lambda x:(x>0).mean())).reset_index()
    directions.to_csv(REPORT/'directions.csv', index=False)
    audit = dict(reference_unchanged=True, new_refits=0, source_sha256={str(p):digest(p) for p in paths},
        proxy_agreement_by_role=merged.groupby('role').proxy_agrees_with_gross_pnl.agg(['size', 'sum', 'mean']).to_dict('index'),
        proxy_mismatches=int((~merged.proxy_agrees_with_gross_pnl).sum()), total_legs=len(merged),
        method='RV proxy uses actual holding trading sessions and quoted calendar tenor; sign compared with actual gross option + stock hedge + dividends, before costs/funding.',
        limitations='Gamma/moneyness changes, discrete hedges, price jumps, and dividend/rate approximations can affect proxy agreement. This check does not isolate their individual contributions. New relative-regime variants were proposed after inspecting this history; no fresh holdout.')
    write_json(REPORT/'followup_audit.json', audit)
    print(pd.DataFrame(ranks).pivot(index='feature', columns='period', values='within_pair_spearman').round(3).to_string(), flush=True)
    print(confidence.to_string(index=False), flush=True)
    print({k:v for k,v in audit.items() if k!='source_sha256'}, flush=True)


if __name__ == '__main__': main()
