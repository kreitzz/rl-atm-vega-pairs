"""Separate 75-delta put / 25-delta call feature scan, with put25 control."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.analyze_tail_universe_change import BASE, working_inputs
from scripts.run_tail_put25_legacy_rl import CONTROL
from scripts.curve_trading.config import DATA, config, RUNOFF
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.contract_learning_labels import beta_screen
from scripts.tail_feature_exploration import past_z, scheduled_nonoverlap, ranked_statistics
from scripts.run_broad_contract_replay import ROOT, Market
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.select_atm_contracts import load_calendar

OUT = Path('artifacts/contracts/opposite_wing_exploration')
REPORT = Path('reports/opposite_wing_exploration')
SILOS = [('P', 25), ('P', 75), ('C', 25)]
FEATURES = ['wing_iv_gap', 'wing_iv_rank', 'wing_premium_z_gap', 'trailing_vrp_gap']
CONTROLS = ['beta', 'correlation', 'dte_gap', 'delta_gap', 'execution_cost_vol']


def wing_features(a, b, cp, bucket):
    fa, fb = a['frames'][cp, bucket], b['frames'][cp, bucket]
    out = pd.DataFrame(index=fa.index)
    out['wing_iv_gap'] = fa.bs_iv - fb.bs_iv
    # Build rolling ranks from the complete observable quote history, before
    # any eligibility, resolution or matched-sample filters.
    out['wing_iv_rank'] = out.wing_iv_gap.rolling(252, min_periods=63).rank(pct=True)
    out['wing_premium_z_gap'] = a['premium_z'][cp, bucket] - b['premium_z'][cp, bucket]
    out['trailing_vrp_gap'] = (fa.bs_iv**2-a['rv']**2) - (fb.bs_iv**2-b['rv']**2)
    out['dte_gap'] = abs(fa.dte-fb.dte)
    out['delta_gap'] = abs(fa.bs_delta.abs()-fb.bs_delta.abs())
    out['delta_error_max'] = pd.concat([abs(fa.bs_delta.abs()-bucket/100), abs(fb.bs_delta.abs()-bucket/100)], axis=1).max(axis=1, skipna=False)
    out['execution_cost_vol'] = .2 + .03/(100*fa.bs_vega.where(fa.bs_vega>0)) + .03/(100*fb.bs_vega.where(fb.bs_vega>0))
    out['quoted_width_vol_max'] = pd.concat([(fa.ask-fa.bid)/fa.bs_vega.where(fa.bs_vega>0),
                                             (fb.ask-fb.bid)/fb.bs_vega.where(fb.bs_vega>0)], axis=1).max(axis=1, skipna=False)
    for suffix, frame in [('a', fa), ('b', fb)]:
        for column in ['expiry', 'strike']:
            out[f'{column}_{suffix}'] = frame[column]
    out['observed'] = fa.bs_iv.gt(0) & fb.bs_iv.gt(0)
    return out


class WingMarket:
    def __init__(self, market, cfg, dates, sources):
        self.market, self.cfg, self.dates, self.sources = market, cfg, dates, sources
        self.cache = {}

    def retain(self, secids):
        for sid in list(self.cache):
            if sid not in secids: del self.cache[sid]
        gc.collect()

    def asset(self, sid):
        if sid in self.cache: return self.cache[sid]
        cal = self.market.calendar; selections = []; quotes = []
        for cp, buckets in [('P', ['25','75']), ('C', ['atm','25'])]:
            parts = []
            for year in cal.year.unique():
                path = DATA / 'selection' / cp / str(year) / f'{sid}.parquet'
                if path.exists():
                    self.sources.setdefault(str(path), digest(path))
                    frame = pd.read_parquet(path)
                    parts.append(frame[frame.tenor.eq(30) & frame.bucket.isin(buckets)])
            if not parts: continue
            selection = pd.concat(parts, ignore_index=True); selections.append(selection)
            ids = selection[selection.date.isin(self.dates) & selection.bucket.ne('atm')].optionid.unique().tolist()
            for year in cal.year.unique():
                path = DATA / 'history' / cp / str(year) / f'{sid}.parquet'
                if ids and path.exists():
                    self.sources.setdefault(str(path), digest(path))
                    quotes.append(pd.read_parquet(path, filters=[('optionid','in',ids)]))
        sel = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(columns=[
            'date','expiry','strike','bucket','cp_flag','optionid','bs_iv','bs_delta','bs_vega','dte','bid','ask'])
        for c in ['date','expiry']: sel[c] = pd.to_datetime(sel[c])
        q = pd.concat(quotes, ignore_index=True) if quotes else pd.DataFrame()
        stock = self.market.stocks[self.market.stocks.secid.eq(sid)].set_index('date').reindex(cal)
        stock['secid'] = sid; stock['rate'] = self.market.rates; stock.index.name = 'date'
        divs = self.market.divs[self.market.divs.secid.eq(sid)]
        raw = dict(q=q, s=stock.reset_index(), sel=sel, divs=divs)
        q, stock, sel, divs = working_inputs([raw], [sid])
        adjusted = stock.set_index('date').close * stock.set_index('date').cfadj
        returns = adjusted.pct_change(fill_method=None)
        rv = np.sqrt(252 * returns.pow(2).rolling(20, min_periods=20).mean())
        frames = {}
        for cp, bucket in [*SILOS, ('C','atm')]:
            frame = sel[sel.cp_flag.eq(cp) & sel.bucket.eq(str(bucket))].set_index('date').reindex(cal)
            assert not frame.index.has_duplicates
            frame['bs_iv'] = frame.bs_iv.where(frame.bs_iv.gt(0))
            frames[cp,bucket] = frame
        atm = frames['C','atm']
        premium_z = {key:past_z(frame.bs_iv / atm.bs_iv - 1).where(frame.expiry.eq(atm.expiry))
                     for key, frame in frames.items() if key in SILOS}
        unit = UnitPaths(q, stock, sel, cal, self.dates, divs, self.cfg) if len(q) else None
        result = dict(frames=frames, premium_z=premium_z, rv=rv, returns=returns, unit=unit)
        self.cache[sid] = result
        return result


def prepare(defs, market, cfg, sources):
    cal = market.calendar
    dates = cal[(cal >= '2018-01-01') & (cal <= cal[-1]-pd.Timedelta(days=RUNOFF))]
    loader = WingMarket(market, cfg, dates, sources)
    for n, d in enumerate(defs, 1):
        folder = OUT / 'pairs' / d['pair']; folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'coverage.json').exists(): continue
        loader.retain(d['secids']); a, b = [loader.asset(sid) for sid in d['secids']]
        source = BASE / d['pair'] / 'states.parquet'
        if source.exists():
            screen = pd.read_parquet(source, columns=['beta','correlation','eligible']).reindex(dates)
        else:
            screen = beta_screen(pd.concat([a['returns'], b['returns']], axis=1), cal, beta_tolerance=.15).reindex(dates)
        old = CONTROL / d['pair'] / 'legacy_calendar_memory/actions.csv'
        legacy_dates = read_actions(old).index if old.exists() else pd.DatetimeIndex([])
        coverage = []
        for cp, bucket in SILOS:
            frame = wing_features(a, b, cp, bucket).reindex(dates)
            frame['date'] = dates; frame['pair'] = d['pair']; frame['label'] = d['label']
            frame['beta'] = screen.beta; frame['correlation'] = screen.correlation
            frame['eligible'] = screen.eligible.fillna(False).astype(bool)
            frame['legacy'] = dates.isin(legacy_dates)
            frame['long_net'] = np.nan; frame['short_net'] = np.nan; frame['resolved'] = False
            frame['end'] = pd.NaT; frame['status'] = 'missing_history'
            if a['unit'] is not None and b['unit'] is not None:
                daily, ends, vegas, status = pair_labels(a['unit'], b['unit'], (30,cp,bucket,bucket), cfg)
                for name, index in [('long_net',-1),('short_net',0)]:
                    frame[name] = np.divide(daily[:,index].sum(axis=1)*cfg.entry_vega_per_leg, vegas[:,index],
                        out=np.full(len(dates),np.nan), where=vegas[:,index]>0)
                frame['resolved'] = np.isfinite(frame[['long_net','short_net']]).all(axis=1) & (vegas[:,[0,-1]]>0).all(axis=1)
                frame['end'] = cal[ends]; frame['status'] = status
                del daily
            frame['directional_edge'] = (frame.long_net-frame.short_net)/2
            planned = frame[['expiry_a','expiry_b']].max(axis=1, skipna=False)
            for sample in ['broad','legacy']:
                known = frame.observed & frame.eligible & (frame.legacy if sample == 'legacy' else True)
                frame[f'nonoverlap_{sample}'] = scheduled_nonoverlap(dates, planned, known)
            frame.to_parquet(folder / f'{cp}{bucket}.parquet', index=False)
            coverage.append(dict(silo=f'{cp}{bucket}', observed=int(frame.observed.sum()),
                beta_eligible=int((frame.observed & frame.eligible).sum()),
                resolved_beta_eligible=int((frame.resolved & frame.eligible).sum()),
                resolved_legacy=int((frame.resolved & frame.eligible & frame.legacy).sum()),
                unresolved=int(frame.status.eq('unresolved').sum())))
        write_json(folder / 'coverage.json', dict(pair=d['pair'], label=d['label'], buckets=coverage))
        write_json(OUT / 'market_sources.json', sources)
        write_json(OUT / 'progress.json', dict(stage='prepare', pair=n, roster=len(defs), label=d['label'],
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if n % 10 == 0: print(f'Opposite wing {n}/{len(defs)} {d["label"]}', flush=True)
        del a, b, frame; gc.collect()
    loader.cache.clear(); gc.collect()


def parity_matches(put, call):
    keys = ['pair','date']; cols = [*keys,'strike_a','strike_b','expiry_a','expiry_b','resolved','eligible','legacy']
    merged = put[cols].merge(call[cols], on=keys, suffixes=('_p','_c'), validate='one_to_one')
    exact = pd.Series(True, index=merged.index)
    for suffix in ['a','b']:
        exact &= (abs(merged[f'strike_{suffix}_p']-merged[f'strike_{suffix}_c']) < .00001)
        exact &= merged[f'expiry_{suffix}_p'].eq(merged[f'expiry_{suffix}_c'])
    merged['same_strikes_expiries'] = exact
    return merged


def diagnostics(defs):
    coverage = []
    for d in defs:
        raw = json.loads((OUT / 'pairs' / d['pair'] / 'coverage.json').read_text())
        coverage.extend(dict(pair=d['pair'],label=d['label'],**r) for r in raw['buckets'])
    pd.DataFrame(coverage).to_csv(REPORT / 'coverage.csv', index=False)
    data = {f'{cp}{bucket}':pd.concat([pd.read_parquet(OUT / 'pairs' / d['pair'] / f'{cp}{bucket}.parquet') for d in defs], ignore_index=True)
            for cp, bucket in SILOS}
    matches = parity_matches(data['P75'],data['C25'])
    matches.to_parquet(REPORT / 'parity_matches.parquet', index=False)
    eligible = matches.eligible_p & matches.eligible_c & matches.resolved_p & matches.resolved_c
    write_json(REPORT / 'parity_summary.json', dict(resolved_shared_rows=int(eligible.sum()),
        exact_strike_expiry_rows=int((eligible & matches.same_strikes_expiries).sum()),
        exact_fraction=float(matches.loc[eligible,'same_strikes_expiries'].mean()) if eligible.any() else None))
    shared = None
    for frame in data.values():
        keys = frame.loc[frame.resolved & frame.eligible,['pair','date']]
        shared = keys if shared is None else shared.merge(keys,on=['pair','date'],validate='one_to_one')
    exactkeys = matches.loc[eligible & matches.same_strikes_expiries,['pair','date']]
    summaries = []; pairrows = []; rules = []; yearly = []
    for silo, frame in data.items():
        frame = frame.merge(shared.assign(matched=True),on=['pair','date'],how='left',validate='one_to_one')
        frame = frame.merge(exactkeys.assign(parity_exact=True),on=['pair','date'],how='left',validate='one_to_one')
        periods = [('pre2022_completed',frame.end.lt('2022-01-01')),('2022plus_entries',frame.date.ge('2022-01-01'))]
        for period, mask in periods:
            resolved = frame[mask & frame.resolved & frame.eligible]
            samples = dict(broad=resolved, legacy=resolved[resolved.legacy],
                matched_legacy=resolved[resolved.legacy & resolved.matched.eq(True)],
                nonoverlap_legacy=resolved[resolved.legacy & resolved.nonoverlap_legacy],
                parity_exact=resolved[resolved.parity_exact.eq(True)])
            for sample, subset in samples.items():
                for feature in FEATURES:
                    stats, bypair = ranked_statistics(subset,feature,controls=CONTROLS)
                    summaries.append(dict(silo=silo,period=period,sample=sample,feature=feature,observations=len(subset),**stats))
                    pairrows.extend(dict(silo=silo,period=period,sample=sample,feature=feature,**r) for r in bypair)
                rank = subset.wing_iv_rank
                selected = subset[rank.le(.2) | rank.gt(.8)].copy()
                selected['rule_net'] = np.where(selected.wing_iv_rank.le(.2),selected.long_net,selected.short_net)
                means = selected.groupby('pair').rule_net.mean()
                rules.append(dict(silo=silo,period=period,sample=sample,opportunities=len(selected),pairs=int(selected.pair.nunique()),
                    positive_pairs=int(means.gt(0).sum()),equal_pair_mean_net_per_vega=float(means.mean()),
                    pooled_mean_net_per_vega=float(selected.rule_net.mean())))
        for year, subset in frame[frame.resolved & frame.eligible & frame.legacy].groupby(frame.date.dt.year):
            stats, _ = ranked_statistics(subset,'wing_iv_gap',controls=CONTROLS,min_observations=8)
            yearly.append(dict(silo=silo,year=int(year),observations=len(subset),**stats))
        print(f'Diagnostics {silo} complete',flush=True)
    pd.DataFrame(summaries).to_csv(REPORT / 'correlations.csv',index=False)
    pd.DataFrame(pairrows).to_csv(REPORT / 'by_pair.csv',index=False)
    pd.DataFrame(rules).to_csv(REPORT / 'spread_rule_outcomes.csv',index=False)
    pd.DataFrame(yearly).to_csv(REPORT / 'by_year.csv',index=False)
    return pd.DataFrame(summaries)


def main():
    started=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((BASE/'inputs.json').read_text()); defs=manifest['definitions']
    sources=dict(manifest['sources']); sources.update(json.loads((BASE/'market_sources.json').read_text()))
    for study in ['tail_put25_legacy_rl_all_months','tail_calendar_ablation']:
        sources.update(json.loads(Path(f'reports/{study}/output_hashes.json').read_text()))
    for path in [Path(__file__),Path('tests/test_opposite_wing.py')]: sources[str(path)]=digest(path)
    for path,sha in sources.items(): assert digest(path)==sha,f'Prior input changed: {path}'
    write_json(OUT/'inputs.json',dict(sources=sources,silos=SILOS,features=FEATURES,
        protocol='Feature exploration only, no fitted trading policy or portfolio Sharpe. Three separate 30D silos: put25 control, put75, call25. All 345 cached roster pairs; no inherited ATM-panel prerequisite. Same .03 commission/.1-vol-point execution penalty, daily delta hedge, initial vega match, own-expiry standalone outcomes. Prior monthly beta 0.85-1.15/correlation >=0.5. Features formed on observable full history before eligibility and payoff resolution. Report broad, original-calendar, common-three-bucket, nonoverlapping original-calendar, and exact-call25/put75-strike-expiry samples. No put25 profitability gate transferred to the opposite wing. No American early exercise/assignment. Corporate-action-unresolved standalone histories remain explicit exclusions. Inspected history, overlapping dependent cohorts, conditional quote/outcome coverage, no multiplicity-adjusted hypothesis test or untouched holdout.'))
    cal=load_calendar(ROOT); cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    cfg=replace(config(),option_commission=.03); market=Market(cal)
    prepare(defs,market,cfg,sources); result=diagnostics(defs)
    for path,sha in sources.items(): assert digest(path)==sha,f'Prior input changed: {path}'
    record=dict(status='complete',pairs_requested=len(defs),models_trained=0,source_files_verified=len(sources),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(REPORT/'audit.json',record)
    selected=result[result.feature.eq('wing_iv_gap') & result['sample'].isin(['legacy','matched_legacy','nonoverlap_legacy'])]
    (REPORT/'findings.txt').write_text(selected.to_string(index=False)+'\n\n'+json.dumps(record,indent=2)+'\n\n'+json.loads((OUT/'inputs.json').read_text())['protocol'])
    hashes={str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'}
    hashes[str(Path(__file__))]=digest(Path(__file__)); write_json(REPORT/'output_hashes.json',hashes)
    print(selected.to_string(index=False),flush=True)


if __name__=='__main__':main()
