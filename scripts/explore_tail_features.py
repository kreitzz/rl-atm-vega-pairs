"""Bounded, pre-policy tail-option feature exploration; no RL/NN fits."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.tail_feature_exploration import (BUCKETS, COMBINATIONS, FEATURES, CONTROLS,
    return_shape, surface_states, pair_features, scheduled_nonoverlap, ranked_statistics)
from scripts.run_atm65_volume_oi import EMPLOYEE, CHECKPOINT
from scripts.run_atm65_ablation import write_json
from scripts.curve_trading.config import DATA, config, COMBOS, RUNOFF
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.curve_trading.policy import Replay
from scripts.run_curve_trading import screen_dates
from scripts.run_broad_contract_replay import ROOT, Market
from scripts.run_contract_walkforward import digest
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar

OUT = Path('artifacts/contracts/tail_feature_exploration')
REPORT = Path('reports/tail_feature_exploration')
SELECTION_COLUMNS = ['date', 'secid', 'optionid', 'expiry', 'cp_flag', 'strike', 'tenor', 'bucket',
    'bs_iv', 'bs_delta', 'bs_vega', 'bs_gamma', 'dte', 'bid', 'ask', 'open_interest']


class TailMarket:
    def __init__(self, market, cfg, dates, sources):
        self.market, self.cfg, self.dates, self.sources = market, cfg, dates, sources
        self.cache = {}

    def retain(self, secids):
        for sid in list(self.cache):
            if sid not in secids: del self.cache[sid]
        gc.collect()

    def asset(self, sid):
        if sid in self.cache: return self.cache[sid]
        cal = self.market.calendar; selected = []; quotes = []
        for cp in ['C', 'P']:
            parts = []
            for year in sorted(set(cal.year)):
                path = DATA/'selection'/cp/str(year)/f'{sid}.parquet'
                if path.exists():
                    if str(path) not in self.sources: self.sources[str(path)] = digest(path)
                    s = pd.read_parquet(path, columns=SELECTION_COLUMNS)
                    parts.append(s[s.bucket.isin(BUCKETS)&s.tenor.isin([14, 30, 45])])
            if not parts: continue
            s = pd.concat(parts, ignore_index=True); selected.append(s)
            ids = s[s.tenor.eq(30)&s.date.isin(self.dates)].optionid.unique().tolist()
            if not ids: continue
            for year in sorted(set(cal.year)):
                path = DATA/'history'/cp/str(year)/f'{sid}.parquet'
                if path.exists():
                    if str(path) not in self.sources: self.sources[str(path)] = digest(path)
                    quotes.append(pd.read_parquet(path, filters=[('optionid', 'in', ids)]))
        selection = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=SELECTION_COLUMNS)
        for col in ['date', 'expiry']: selection[col] = pd.to_datetime(selection[col])
        for col in ['bs_iv', 'bs_delta', 'bs_vega', 'bs_gamma', 'dte', 'bid', 'ask']:
            selection[col] = pd.to_numeric(selection[col], errors='coerce')
        q = pd.concat(quotes, ignore_index=True) if quotes else pd.DataFrame(columns=['date', 'optionid', 'cp_flag'])
        stock = self.market.stocks[self.market.stocks.secid.eq(sid)].set_index('date').reindex(cal)
        stock['secid'] = sid; stock['rate'] = self.market.rates; stock.index.name = 'date'; stock = stock.reset_index()
        divs = self.market.divs[self.market.divs.secid.eq(sid)]
        native = selection[selection.tenor.eq(30)&selection.date.isin(self.dates)]
        unit = UnitPaths(q, stock, native, cal, self.dates, divs, self.cfg)
        surfaces = surface_states(selection, cal); shape = return_shape(stock, cal)
        self.cache[sid] = dict(unit=unit, surface=surfaces, shape=shape, q=q, s=stock, selection=native, divs=divs)
        return self.cache[sid]


def replay_control(a, b, combo, date, long_net, short_net, cfg, cal, secids):
    """Actual ledger comparison on one isolated tail cohort in each direction."""
    q = pd.concat([a['q'], b['q']], ignore_index=True)
    s = pd.concat([a['s'], b['s']], ignore_index=True)
    sel = pd.concat([a['selection'], b['selection']], ignore_index=True)
    divs = pd.concat([a['divs'], b['divs']], ignore_index=True)
    replay = Replay(q, s, sel, secids, divs, replace(cfg, risk_controls=False))
    index = COMBOS.index(combo); errors = []
    for code, expected in [(1+index*8+7, long_net), (1+index*8, short_net)]:
        pos = cal.get_loc(date); ids = [x['unit'].ids(*combo[:2], combo[2+i])[pos] for i, x in enumerate([a, b])]
        end = max(a['unit'].ends[ids[0]], b['unit'].ends[ids[1]])
        nav, trades, decisions = replay.run(pd.Series([code], index=pd.DatetimeIndex([date])), cal[pos:end+1])
        mark_book(nav, cfg_rates(s, cal[pos:end+1]), cfg.capital)
        entered = decisions[decisions.status.eq('entered')]
        assert len(entered)==1
        initial_vega = float((entered.vega_a.iloc[0]+entered.vega_b.iloc[0])/2)
        actual = (nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])/initial_vega
        errors.append(abs(actual-expected)); np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
    return max(errors)


def cfg_rates(stock, calendar):
    return stock.drop_duplicates('date').set_index('date').rate.reindex(calendar)


def prepare(defs, market, cfg, limit=None):
    cal = market.calendar; dates = cal  # Same date axis for bounded reusable name paths.
    source_path = OUT/'market_sources.json'
    sources = json.loads(source_path.read_text()) if source_path.exists() else {}
    loader = TailMarket(market, cfg, dates, sources); newly = 0
    for i, d in enumerate(defs, 1):
        folder = OUT/'pairs'/d['pair']; folder.mkdir(parents=True, exist_ok=True)
        if (folder/'coverage.json').exists(): continue
        source = EMPLOYEE/d['pair']/'states.parquet'
        if not source.exists():
            write_json(folder/'coverage.json', dict(pair=d['pair'], label=d['label'], state_rows=0, combinations=[])); continue
        started = time.monotonic(); panel = pd.read_parquet(source)
        loader.retain(d['secids']); a, b = [loader.asset(sid) for sid in d['secids']]
        eligible = (panel.index >= '2018-01-01')&(panel.index <= cal[-1]-pd.Timedelta(days=RUNOFF))&screen_dates(d, panel.index)&panel.eligible.to_numpy(bool)
        positions = cal.get_indexer(panel.index); coverage = []; controls = []
        for cp, bucket in COMBINATIONS:
            combo = (30, cp, bucket if bucket=='atm' else int(bucket), bucket if bucket=='atm' else int(bucket))
            daily, ends, vegas, status = pair_labels(a['unit'], b['unit'], combo, cfg)
            complete = np.isfinite(daily[:, [0, -1]]).all(axis=(1,2))&(vegas[:, [0, -1]]>0).all(axis=1)
            long_net = np.divide(daily[:, -1].sum(axis=1)*cfg.entry_vega_per_leg, vegas[:, -1], out=np.full(len(cal), np.nan), where=vegas[:, -1]>0)
            short_net = np.divide(daily[:, 0].sum(axis=1)*cfg.entry_vega_per_leg, vegas[:, 0], out=np.full(len(cal), np.nan), where=vegas[:, 0]>0)
            first, second = a['surface'][cp, bucket], b['surface'][cp, bucket]
            row = pair_features(first, second, a['shape'], b['shape'], panel, cp)
            row['date'] = panel.index; row['pair'] = d['pair']; row['label'] = d['label']; row['cp'] = cp; row['bucket'] = bucket
            row['end'] = cal[ends[positions]]
            row['long_net'] = long_net[positions]; row['short_net'] = short_net[positions]
            row['directional_edge'] = (row.long_net-row.short_net)/2
            row['resolved'] = complete[positions]; row['eligible'] = eligible
            # Build entry ranks before eligibility or future-payoff filtering.
            for feature in FEATURES:
                pct = row[feature].reindex(cal).rolling(252, min_periods=63).rank(pct=True)
                row[feature+'_pct'] = pct.reindex(panel.index)
            if bucket!='atm':
                errors = pd.concat([(first.abs_delta-int(bucket)/100).abs(), (second.abs_delta-int(bucket)/100).abs()], axis=1)
                row['delta_error_max'] = errors.max(axis=1, skipna=False).reindex(panel.index)
            else: row['delta_error_max'] = 0.
            expiry = pd.concat([first.expiry, second.expiry], axis=1).max(axis=1, skipna=False).reindex(panel.index)
            planned = pd.Series(cal[np.clip(cal.searchsorted(pd.DatetimeIndex(expiry), side='right')-1, 0, len(cal)-1)], index=panel.index).where(expiry.notna())
            known = first.optionid.reindex(panel.index).notna()&second.optionid.reindex(panel.index).notna()&eligible
            row['nonoverlap'] = scheduled_nonoverlap(panel.index, planned, known)
            row['scheduled_end'] = planned
            target = row.loc[eligible].copy(); target.to_parquet(folder/f'{cp}{bucket}.parquet', index=False)
            coverage.append(dict(cp=cp, bucket=bucket, eligible=int(eligible.sum()), signal_known=int(known.sum()),
                resolved=int((eligible&row.resolved).sum()), unresolved=int((eligible&(status[positions]=='unresolved')).sum()),
                no_entry=int((eligible&(status[positions]=='no_entry')).sum()),
                nonoverlap_resolved=int((eligible&row.resolved&row.nonoverlap).sum())))
            # Recover the native cached current-cost economic outcomes exactly.
            old_path = Path('artifacts/contracts/atm_variance_mining/pairs')/d['pair']/'outcomes.parquet'
            if cp=='C' and bucket=='atm' and old_path.exists():
                old = pd.read_parquet(old_path, columns=['date', 'long_net', 'short_net', 'end']).set_index('date')
                got = row[row.resolved].set_index('date'); both = old.index.intersection(got.index)
                if len(both):
                    np.testing.assert_allclose(got.loc[both, ['long_net','short_net']], old.loc[both, ['long_net','short_net']], atol=1e-6, rtol=0)
                    assert (got.loc[both, 'end']==old.loc[both, 'end']).all()
                    controls.append(dict(kind='native_ATM', cohorts=len(both), max_error=float(abs(got.loc[both, ['long_net','short_net']]-old.loc[both, ['long_net','short_net']]).to_numpy().max())))
            # Eight actual-ledger checks on the first available pair only.
            if not (OUT/'tail_replay_controls.json').exists() and bucket!='atm':
                available = row[row.eligible&row.resolved]
                if len(available):
                    chosen = available.iloc[0]
                    error = replay_control(a, b, combo, chosen.date, chosen.long_net, chosen.short_net, cfg, cal, d['secids'])
                    controls.append(dict(kind=f'{cp}{bucket}_actual_ledger', cohorts=2, max_error=error))
            del daily; gc.collect()
        if not (OUT/'tail_replay_controls.json').exists() and any('actual_ledger' in x['kind'] for x in controls):
            write_json(OUT/'tail_replay_controls.json', dict(pair=d['pair'], controls=controls))
        write_json(folder/'coverage.json', dict(pair=d['pair'], label=d['label'], state_rows=len(panel), combinations=coverage, controls=controls))
        write_json(source_path, sources); newly += 1
        write_json(OUT/'progress.json', dict(pair_number=i, roster=len(defs), last_pair=d['label'], last_seconds=time.monotonic()-started,
            completed_with_states=newly, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print(f'{i}/{len(defs)} {d["label"]}: six fixed tail/control cohorts, {time.monotonic()-started:.1f}s; peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MiB', flush=True)
        del panel, a, b, row, target; gc.collect()
        if limit and newly>=limit: return False
    write_json(OUT/'prepared.json', dict(roster=len(defs), combinations=COMBINATIONS, fitting=False))
    return True


def diagnostics(defs):
    summaries = []; correlations = []; years = []; quintiles = []; side_bins = []; coverage = []; annual_outcomes = []
    controls = []
    for d in defs:
        c = json.loads((OUT/'pairs'/d['pair']/'coverage.json').read_text())
        coverage.extend(dict(pair=d['pair'], label=d['label'], **r) for r in c['combinations'])
        controls.extend(dict(pair=d['pair'], **r) for r in c.get('controls', []))
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv', index=False); pd.DataFrame(controls).to_csv(REPORT/'ledger_controls.csv', index=False)
    for cp, bucket in COMBINATIONS:
        frames = []
        for d in defs:
            path = OUT/'pairs'/d['pair']/f'{cp}{bucket}.parquet'
            if not path.exists(): continue
            f = pd.read_parquet(path); frames.append(f)
        data = pd.concat(frames, ignore_index=True)
        atm = pd.concat([pd.read_parquet(OUT/'pairs'/d['pair']/f'{cp}atm.parquet', columns=['pair','date','resolved'])
            for d in defs if (OUT/'pairs'/d['pair']/f'{cp}atm.parquet').exists()], ignore_index=True)
        matched = atm[atm.resolved][['pair','date']].assign(atm_resolved=True)
        data = data.merge(matched, on=['pair','date'], how='left', validate='one_to_one')
        early = data.end.lt('2022-01-01')&data.resolved
        later = data.date.ge('2022-01-01')&data.resolved
        for period, mask in [('early_complete', early), ('later', later)]:
            period_data = data[mask].copy()
            samples = {'all':period_data, 'nonoverlap':period_data[period_data.nonoverlap],
                'matched_atm':period_data[period_data.atm_resolved.eq(True)],
                'tight_delta':period_data[period_data.delta_error_max.le(.03)]}
            for feature in FEATURES:
                for sample, h in samples.items():
                    stats, pair_rows = ranked_statistics(h, feature, min_observations=8 if sample=='nonoverlap' else 20)
                    summaries.append(dict(cp=cp, bucket=bucket, period=period, sample=sample, feature=feature,
                        observations=int(h[[feature,'directional_edge']].notna().all(axis=1).sum()), **stats))
                    if sample=='all': correlations.extend(dict(cp=cp, bucket=bucket, period=period, feature=feature, **r) for r in pair_rows)
                g = period_data.dropna(subset=[feature, 'directional_edge']).copy()
                g['percentile'] = g[feature+'_pct']
                g = g.dropna(subset=['percentile'])
                g['quintile'] = np.minimum(np.floor(g.percentile*5).astype(int),4)+1
                for q, h in g.groupby('quintile'):
                    quintiles.append(dict(cp=cp, bucket=bucket, period=period, feature=feature, quintile=q, observations=len(h), pairs=h.pair.nunique(),
                        mean_directional_edge=h.directional_edge.mean(), long_net_mean=h.long_net.mean(), short_net_mean=h.short_net.mean(),
                        long_win_rate=h.long_net.gt(0).mean(), short_win_rate=h.short_net.gt(0).mean()))
            for year, h in period_data.groupby(period_data.date.dt.year):
                annual_outcomes.append(dict(cp=cp, bucket=bucket, period=period, year=year, observations=len(h),
                    mean_long_net=h.long_net.mean(), mean_short_net=h.short_net.mean(), long_win_rate=h.long_net.gt(0).mean(), short_win_rate=h.short_net.gt(0).mean()))
        for year, h in data[data.resolved].groupby(data[data.resolved].date.dt.year):
            for feature in FEATURES:
                stats, _ = ranked_statistics(h, feature)
                years.append(dict(cp=cp, bucket=bucket, year=year, feature=feature, **stats))
        # Wing and ATM feature correlations on identical observed, resolved dates.
        if bucket!='atm':
            atm_full = pd.concat([pd.read_parquet(OUT/'pairs'/d['pair']/f'{cp}atm.parquet') for d in defs
                if (OUT/'pairs'/d['pair']/f'{cp}atm.parquet').exists()], ignore_index=True)
            common = data[data.resolved&data.atm_resolved.eq(True)]
            control = atm_full[atm_full.resolved].merge(common[['pair','date']+FEATURES], on=['pair','date'], suffixes=('_atm',''), validate='one_to_one')
            for feature in FEATURES:
                for period, h in [('early_complete', control[control.end.lt('2022-01-01')]), ('later', control[control.date.ge('2022-01-01')])]:
                    stats, _ = ranked_statistics(h, feature)
                    summaries.append(dict(cp=cp, bucket=bucket, period=period, sample='atm_payoff_same_wing_feature', feature=feature,
                        observations=int(h[[feature,'directional_edge']].notna().all(axis=1).sum()), **stats))
        print(f'Diagnostics {cp}{bucket}: {int(data.resolved.sum()):,} resolved screened cohorts', flush=True)
        del data, frames, atm; gc.collect()
    for name, rows in [('correlations', summaries), ('by_pair', correlations), ('by_year', years), ('quintiles', quintiles), ('outcomes_by_year', annual_outcomes)]:
        pd.DataFrame(rows).to_csv(REPORT/f'{name}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int); parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); cfg = replace(config(), option_commission=.03)
    source = EMPLOYEE/'inputs.json'; defs = json.loads(source.read_text())['definitions']
    cal = load_calendar(ROOT); cal = cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    files = [Path(__file__), Path('scripts/tail_feature_exploration.py'), Path('scripts/curve_trading/unit_paths.py'),
        Path('scripts/run_contract_tail_moments.py'), Path('scripts/option_flex/contract_engine.py'), Path('scripts/curve_trading/policy.py'),
        Path('scripts/run_curve_trading.py'), Path('scripts/option_flex/black_scholes.py'), CHECKPOINT, source,
        ROOT/'zero_curve.parquet', ROOT/'distributions.parquet',
        *[ROOT/'underlying'/f'year={year}.parquet' for year in cal.year.unique()],
        *[EMPLOYEE/d['pair']/'states.parquet' for d in defs if (EMPLOYEE/d['pair']/'states.parquet').exists()]]
    signature = dict(settings=asdict(cfg), combinations=[list(c) for c in COMBINATIONS], features=FEATURES, controls=CONTROLS,
        sources={str(p):digest(p) for p in files}, roster=defs, nn_refits=0, rl_refits=0,
        protocol='Independent tail silos, not a larger policy action menu. Pre-policy, beta-eligible historical screens; initial vega-neutral same-type/same-bucket 30D ATM/25-delta/10-delta options; daily delta hedge and own expiry; no portfolio overlay.')
    manifest = OUT/'inputs.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==signature, 'Inputs changed'
    else: write_json(manifest, signature)
    market = Market(cal)
    if not (OUT/'prepared.json').exists() and not args.report_only:
        if not prepare(defs, market, cfg, args.limit): return
    assert (OUT/'prepared.json').exists(), 'Preparation incomplete'
    diagnostics(defs)
    for path, sha in signature['sources'].items(): assert digest(path)==sha, f'Source changed: {path}'
    for path, sha in json.loads((OUT/'market_sources.json').read_text()).items(): assert digest(path)==sha, f'Market cache changed: {path}'
    reference = json.loads(CHECKPOINT.read_text()); assert digest(reference['archive'])==reference['archive_sha256']
    for path, sha in reference['files'].items(): assert digest(path)==sha, f'Checkpoint changed: {path}'
    rows = [json.loads((OUT/'pairs'/d['pair']/'coverage.json').read_text()) for d in defs]
    audit = dict(status='complete', roster=len(defs), pairs_with_state_inputs=sum(r['state_rows']>0 for r in rows),
        source_files_verified=len(signature['sources']), market_files_verified=len(json.loads((OUT/'market_sources.json').read_text())),
        frozen_files_verified=len(reference['files']), reference_archive_verified=True, no_policy_refits=True,
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        outcome_label='Terminal standalone excess P&L / actual mean initial leg vega; not daily portfolio returns or Sharpe.',
        sample_limitations='Inspected history; missing labels condition evaluation only; overlapping cohorts/shared names; quarter resampling is descriptive and unadjusted for multiple feature tests.',
        risk_limits='Same entry quote/initial-vega controls, costs and financing; standalone labels do not impose portfolio reserve/stress/capacity overlays.',
        curvature='Smile butterfly and wing steepness are proxies, not model-free risk-neutral skewness or kurtosis.')
    write_json(REPORT/'audit.json', audit); write_json(OUT/'status.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__=='__main__': main()
