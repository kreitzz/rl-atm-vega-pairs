"""Actual executed-cohort/name diagnostics; no policy fitting or exclusions."""
from pathlib import Path
import gc
import json
import resource
import time
import numpy as np
import pandas as pd
from scripts.trade_quality_ledger import reconstruct, connected_episodes, FIELDS
from scripts.run_atm65_mined_signals import OUT as MINED, REPORT as MINED_REPORT, MODELS, EMPLOYEE, baseline_results, employee_roster, CHECKPOINT
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.curve_trading.config import DATA
from scripts.select_atm_contracts import load_calendar
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import write_json
from scripts.diagnose_current_contract_losses import read_csv
from scripts.option_flex.black_scholes import option_values

OUT = Path('artifacts/contracts/current_trade_quality')
REPORT = Path('reports/current_trade_quality')
ALL_MODELS = ['rl_baseline']+MODELS
PREDICTORS = ['cheap_iv_gap', 'relative_trailing_vrp', 'short_own_vrp_z', 'relative_own_vrp_z',
    'beta_distance_from_one', 'correlation', 'iv_level_corr_126', 'iv_change_corr_126',
    'absolute_market_iv_beta_gap', 'signal_gamma_ratio_long_short', 'signal_dte_gap',
    'short_earnings_before_expiry', 'short_earnings_next_session', 'short_rv_above_iv',
    'memory_91d', 'memory_365d', 'memory_91_over_365', 'vix_previous_close',
    'downside_share_gap', 'momentum_gap', 'signal_quote_width_max', 'signal_open_interest_min']
PREDICTORS += ['signal_funding_debit_per_vega']


def history_panels(defs, calendar):
    panels = {}; marks = []
    for d in defs:
        path = EMPLOYEE/d['pair']/'states.parquet'
        if not path.exists(): continue
        p = pd.read_parquet(path)
        added_path = MINED/d['pair']/'states.parquet'
        if added_path.exists():
            added = pd.read_parquet(added_path, columns=['downside_fraction_gap', 'downside_fraction_available'])
            p['diagnostic_downside_fraction_gap'] = added.downside_fraction_gap.where(added.downside_fraction_available.eq(1))
        else: p['diagnostic_downside_fraction_gap'] = np.nan
        panels[d['pair']] = p
        for sid, suffix in zip(d['secids'], ['amd', 'nvda']):
            marks.append(pd.DataFrame(dict(date=p.index, secid=sid, iv=p['iv_'+suffix].to_numpy()/100)))
    iv = pd.concat(marks, ignore_index=True).drop_duplicates(['date', 'secid']).pivot(index='date', columns='secid', values='iv').reindex(calendar).where(lambda x: x>0)
    changes = iv.diff(); betas = pd.DataFrame(index=calendar, columns=iv.columns, dtype=float)
    for sid in iv:
        factor = changes.drop(columns=sid).median(axis=1).shift(1)
        own = changes[sid].shift(1)
        var = factor.rolling(252, min_periods=126).var().where(lambda x:x>1e-10)
        betas[sid] = own.rolling(252, min_periods=126).cov(factor)/var
    extras = {}
    for pair, p in panels.items():
        d = next(x for x in defs if x['pair']==pair); a, b = d['secids']
        e = pd.DataFrame(index=p.index)
        e['iv_level_corr_126'] = p.iv_amd.rolling(126, min_periods=63).corr(p.iv_nvda)
        e['iv_change_corr_126'] = p.iv_amd.diff().rolling(126, min_periods=63).corr(p.iv_nvda.diff())
        e['absolute_market_iv_beta_gap'] = (betas[a]-betas[b]).abs().reindex(p.index)
        for suffix in ['amd', 'nvda']:
            premium = (p['iv_'+suffix]/100)**2-(p['trailing_rv_'+suffix]/100)**2
            h = premium.shift(1).rolling(252, min_periods=126)
            e['own_vrp_z_'+suffix] = (premium-h.mean())/h.std().where(lambda x:x>1e-8)
        extras[pair] = e.replace([np.inf, -np.inf], np.nan)
    return panels, extras


def forward_path(stock, entry, exit_date):
    p = (stock.close*stock.cfadj).where(stock.close.gt(0)&stock.cfadj.gt(0)).loc[entry:exit_date]
    returns = np.log(p/p.shift(1)).iloc[1:]
    if returns.empty or not np.isfinite(returns).all(): return dict(forward_rv=np.nan, max_move_percent=np.nan, max_move_date=None)
    k = returns.abs().idxmax()
    return dict(forward_rv=float(100*np.sqrt(252*(returns**2).mean())), max_move_percent=float(100*returns.loc[k]), max_move_date=k)


def signal_inputs(d, decision, legs, panel, extra, lookup, stocks):
    date = decision.signal_date; state = panel.loc[date]; ex = extra.loc[date]
    sign = np.sign(decision.action); side = 'long' if sign>0 else 'short'
    long = legs[legs.role.eq('long')].iloc[0]; short = legs[legs.role.eq('short')].iloc[0]
    suffix = {d['secids'][0]:'amd', d['secids'][1]:'nvda'}
    ls, ss = suffix[long.secid], suffix[short.secid]
    lq = lookup[(date, int(long.optionid))]; sq = lookup[(date, int(short.optionid))]
    lspot = float(stocks[long.secid].loc[date, 'close']); sspot = float(stocks[short.secid].loc[date, 'close'])
    liv, siv = lq['bs_iv']*100, sq['bs_iv']*100
    ldte, sdte = (lq['expiry']-date).days, (sq['expiry']-date).days
    # Compact quote caches omit gamma; derive it from observed signal-close
    # IV/rate/dividend inputs using the same Black-Scholes model, no new fit.
    def gamma_per_vega(quote, spot, dte):
        _, _, gamma, _ = option_values(spot-quote['known_dividend_pv'], quote['strike'], dte/365,
            quote['bs_rate'], quote['bs_iv'], quote['cp_flag'])
        return float(gamma*spot**2/quote['vega']) if quote['vega']>0 else np.nan
    lg = gamma_per_vega(lq, lspot, ldte); sg = gamma_per_vega(sq, sspot, sdte)
    earnings = float(state[ss+'_earnings_days_capped'])
    result = dict(cheap_iv_gap=float(state['iv_'+ss]-state['iv_'+ls]),
        signal_contract_iv_gap=float(siv-liv), signal_long_iv=float(liv), signal_short_iv=float(siv),
        relative_trailing_vrp=float(((state['iv_'+ss]**2-state['trailing_rv_'+ss]**2)
            -(state['iv_'+ls]**2-state['trailing_rv_'+ls]**2))/10000),
        short_own_vrp_z=float(ex['own_vrp_z_'+ss]), relative_own_vrp_z=float(ex['own_vrp_z_'+ss]-ex['own_vrp_z_'+ls]),
        beta_distance_from_one=float(abs(state.beta-1)), beta=float(state.beta), correlation=float(state.correlation),
        signal_gamma_ratio_long_short=float(lg/sg) if sg>0 else np.nan, signal_dte_gap=float(abs(ldte-sdte)),
        short_earnings_days=earnings, short_earnings_before_expiry=float(0<=earnings<=sdte),
        short_earnings_next_session=float(0<=earnings<=1), short_rv_above_iv=float(state['trailing_rv_'+ss]>state['iv_'+ss]),
        memory_91d=float(state[f'profit_{side}_91d']), memory_365d=float(state[f'profit_{side}_365d']),
        vix_previous_close=float(state.vix_previous_close),
        momentum_gap=float(state['momentum_12_1_'+('a' if ls=='amd' else 'b')]-state['momentum_12_1_'+('a' if ss=='amd' else 'b')])
            if state.momentum_12_1_a_available==1 and state.momentum_12_1_b_available==1 else np.nan,
        signal_quote_width_max=float(max((lq['ask']-lq['bid'])/(2*lq['vega']), (sq['ask']-sq['bid'])/(2*sq['vega']))),
        signal_open_interest_min=float(min(lq['open_interest'], sq['open_interest'])),
        signal_funding_debit_per_vega=float(((lq['bid']+lq['ask'])/2-lq['delta']*lspot)/lq['vega']
            -((sq['bid']+sq['ask'])/2-sq['delta']*sspot)/sq['vega']),
        signal_iv_state_difference_max=float(max(abs(liv-state['iv_'+ls]), abs(siv-state['iv_'+ss]))),
        long_name=d['label'].split('/')[d['secids'].index(int(long.secid))],
        short_name=d['label'].split('/')[d['secids'].index(int(short.secid))])
    result.update({k:float(ex[k]) for k in ['iv_level_corr_126', 'iv_change_corr_126', 'absolute_market_iv_beta_gap']})
    result['memory_91_over_365'] = result['memory_91d']/result['memory_365d'] if result['memory_365d']>0 else np.nan
    result['downside_share_gap'] = float(sign*state.diagnostic_downside_fraction_gap)
    result['entry_iv_gap'] = float(short.entry_iv-long.entry_iv)
    result['iv_gap_lost_before_entry'] = result['signal_contract_iv_gap']-result['entry_iv_gap']
    return result


def rank_evidence(frame):
    rows = []
    for model, m in frame.groupby('model'):
        for period, g in [('all', m), ('early_complete', m[m.exit_date<'2022-01-01']), ('later', m[m.entry_date>='2022-01-01'])]:
            for feature in PREDICTORS:
                h = g[[feature, 'net_per_vega', 'pair', 'episode']].dropna()
                pooled = h[feature].corr(h.net_per_vega, method='spearman') if h[feature].nunique()>1 and len(h)>4 else np.nan
                parts = []
                for pair, p in h.groupby('pair'):
                    if len(p)<5 or p[feature].nunique()<3 or p.net_per_vega.nunique()<3: continue
                    x = p[feature].rank(pct=True); y = p.net_per_vega.rank(pct=True)
                    parts.append(pd.DataFrame(dict(x=x-x.mean(), y=y-y.mean(), weight=1/len(p), pair=pair)))
                within = np.nan
                if parts:
                    z = pd.concat(parts); w = z.weight.to_numpy(); x = z.x.to_numpy(); y = z.y.to_numpy()
                    denominator = np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
                    if denominator>0: within = np.sum(w*x*y)/denominator
                rows.append(dict(model=model, period=period, feature=feature, trades=len(h), episodes=h.episode.nunique(),
                    pairs=h.pair.nunique(), pooled_spearman=pooled, within_pair_spearman=within, pairs_within=len(parts)))
    pd.DataFrame(rows).to_csv(REPORT/'signal_rank.csv', index=False)


def conditional_groups(frame):
    rows = []
    for model, m in frame.groupby('model'):
        early = m[m.exit_date<'2022-01-01']
        for feature in ['cheap_iv_gap', 'relative_trailing_vrp', 'short_own_vrp_z', 'relative_own_vrp_z',
            'correlation', 'iv_level_corr_126', 'iv_change_corr_126', 'absolute_market_iv_beta_gap',
            'memory_91d', 'memory_91_over_365', 'downside_share_gap']:
            train = early[feature].dropna()
            if len(train)<15 or train.nunique()<3: continue
            cuts = train.quantile([1/3, 2/3]).to_numpy()
            if cuts[0]==cuts[1]: continue
            for period, g in [('early_complete', early), ('later', m[m.entry_date>='2022-01-01'])]:
                for bucket in [0, 1, 2]:
                    observed = g[feature].notna(); chosen = observed & (np.searchsorted(cuts, g[feature])==bucket)
                    h = g[chosen]
                    rows.append(dict(model=model, period=period, feature=feature, bucket=bucket, lower_cut=cuts[0], upper_cut=cuts[1],
                        trades=len(h), pairs=h.pair.nunique(), episodes=h.episode.nunique(), mean_net_per_vega=h.net_per_vega.mean(),
                        win_rate=h.net_before_shared_financing.gt(0).mean(), net_before_shared_financing=h.net_before_shared_financing.sum()))
        rules = {
            'short_recent_RV_exceeds_IV': m.short_rv_above_iv.eq(1),
            'short_earnings_inside_contract': m.short_earnings_before_expiry.eq(1),
            'short_earnings_next_session': m.short_earnings_next_session.eq(1),
            'expiry_gap_above_7_days': m.signal_dte_gap.gt(7),
            'gamma_ratio_outside_0.8_1.25': m.signal_gamma_ratio_long_short.lt(.8)|m.signal_gamma_ratio_long_short.gt(1.25),
            'short_VRP_below_own_history': m.short_own_vrp_z.lt(0),
            'downside_agrees_with_long': m.downside_share_gap.gt(0),
        }
        required = {'short_VRP_below_own_history':'short_own_vrp_z', 'downside_agrees_with_long':'downside_share_gap'}
        for name, rule in rules.items():
            for period, g in [('early_complete', early), ('later', m[m.entry_date>='2022-01-01'])]:
                if name in required: g = g[g[required[name]].notna()]
                for value in [False, True]:
                    h = g[rule.reindex(g.index).eq(value)]
                    rows.append(dict(model=model, period=period, feature=name, bucket=value, trades=len(h), pairs=h.pair.nunique(),
                        episodes=h.episode.nunique(), mean_net_per_vega=h.net_per_vega.mean(),
                        win_rate=h.net_before_shared_financing.gt(0).mean(), net_before_shared_financing=h.net_before_shared_financing.sum()))
    pd.DataFrame(rows).to_csv(REPORT/'conditional_groups.csv', index=False)


def summarize(cohorts, legs, pairs, annual):
    cohorts['episode'] = None
    for model, g in cohorts.groupby('model'): cohorts.loc[g.index, 'episode'] = connected_episodes(g)
    cohorts.to_csv(REPORT/'cohorts.csv', index=False); legs.to_csv(REPORT/'legs.csv', index=False)
    pairs.to_csv(REPORT/'pairs.csv', index=False); annual.to_csv(REPORT/'annual_pairs.csv', index=False)
    episodes = cohorts.groupby(['model', 'pair', 'label', 'episode']).agg(
        first_entry=('entry_date', 'min'), last_expiry=('exit_date', 'max'), trades=('cohort', 'size'),
        net_before_shared_financing=('net_before_shared_financing', 'sum')).reset_index()
    episodes.to_csv(REPORT/'episodes.csv', index=False)
    names = legs.groupby(['model', 'secid', 'name', 'role']).agg(trades=('cohort', 'size'), pairs=('pair', 'nunique'),
        gross_hedged_pnl=('gross_hedged_pnl', 'sum'), net_before_shared_financing=('net_before_shared_financing', 'sum'),
        mean_net_per_vega=('net_per_vega', 'mean'), win_rate=('profitable', 'mean')).reset_index()
    names.to_csv(REPORT/'names.csv', index=False)
    baseline = cohorts[cohorts.model.eq('rl_baseline')]
    ablations = []
    for model in MODELS:
        candidate = cohorts[cohorts.model.eq(model)]
        oldkeys = set(zip(baseline.pair, baseline.cohort, np.sign(baseline.action)))
        newkeys = set(zip(candidate.pair, candidate.cohort, np.sign(candidate.action)))
        for kind, source, keys, keep in [('added', candidate, oldkeys, False), ('dropped', baseline, newkeys, False),
                                       ('retained_candidate', candidate, oldkeys, True), ('retained_baseline', baseline, newkeys, True)]:
            chosen = [(p, c, np.sign(a)) in keys for p, c, a in zip(source.pair, source.cohort, source.action)]
            h = source[np.asarray(chosen)==keep]
            ablations.append(dict(model=model, kind=kind, trades=len(h), pairs=h.pair.nunique(),
                net_before_shared_financing=h.net_before_shared_financing.sum(), mean_net_per_vega=h.net_per_vega.mean()))
    pd.DataFrame(ablations).to_csv(REPORT/'changed_trades.csv', index=False)
    concentration = []
    for model, g in cohorts.groupby('model'):
        positive = g[g.net_before_shared_financing>0].sort_values('net_before_shared_financing', ascending=False)
        ep = episodes[(episodes.model==model)&(episodes.net_before_shared_financing>0)].sort_values('net_before_shared_financing', ascending=False)
        p = pairs[(pairs.model==model)&(pairs.net_excess_pnl>0)].sort_values('net_excess_pnl', ascending=False)
        concentration.append(dict(model=model, trades=len(g), episodes=g.episode.nunique(), win_rate=g.net_before_shared_financing.gt(0).mean(),
            mean_net_per_vega=g.net_per_vega.mean(), profitable_pairs=int(pairs[(pairs.model==model)&pairs.entries.gt(0)].net_excess_pnl.gt(0).sum()),
            losing_pairs=int(pairs[(pairs.model==model)&pairs.entries.gt(0)].net_excess_pnl.lt(0).sum()),
            largest_pair=p.label.iloc[0], top_3_pair_share_positive_pnl=p.net_excess_pnl.head(3).sum()/p.net_excess_pnl.sum(),
            top_3_trade_share_positive_pnl=positive.net_before_shared_financing.head(3).sum()/positive.net_before_shared_financing.sum(),
            top_3_episode_share_positive_pnl=ep.net_before_shared_financing.head(3).sum()/ep.net_before_shared_financing.sum()))
    pd.DataFrame(concentration).to_csv(REPORT/'concentration.csv', index=False)
    rank_evidence(cohorts); conditional_groups(cohorts)
    features = []
    for model, g in cohorts.groupby('model'):
        for outcome, h in [('winner', g[g.net_before_shared_financing>0]), ('loser', g[g.net_before_shared_financing<=0])]:
            for feature in PREDICTORS:
                features.append(dict(model=model, outcome=outcome, feature=feature, trades=h[feature].notna().sum(),
                    median=h[feature].median(), mean=h[feature].mean()))
    pd.DataFrame(features).to_csv(REPORT/'winner_loser_features.csv', index=False)
    base_pairs = pairs[pairs.model.eq('rl_baseline')&pairs.entries.gt(0)].sort_values('net_excess_pnl')
    lines = ['Current best model: executed-trade and name diagnostics',
        'Main reference: frozen 0.368 full-period RL model, 2018-Aug2025. Latest three state ablations are secondary comparisons, not independent repetitions.',
        'No policy retraining, new trades, changed fills, excluded losing names or new selection rules.',
        'Actual option/stock hedge ledger reconstructed by cohort and underlying, with original Greeks and causal fallback. Initial quantities, splits, dividends and own expiries are preserved.',
        'Netted stock execution/borrow costs allocated pro rata by absolute hedge changes/short hedges. Cohort and leg P&L exclude shared sleeve financing; pair and calendar-year net excess include it.',
        'Cohort labels use complete lifecycles and entry years. Calendar-year pair P&L includes positions crossing year boundaries. Do not mix these definitions.',
        'Forward realized volatility, largest holding-period moves, next-session IV changes and vega drift are retrospective explanations, not available trading states.',
        'Signal features use original signal-close information, prior-only market IV betas/historical premium statistics and observable daily closes. Entries are next session.',
        'Episodes connect overlapping holding periods within each pair. They still share names and market events across pairs and are not independent samples.',
        'Early bins use completed pre-2022 executed trades; boundaries remain fixed later. Fixed rule groups are descriptive. All history has already been inspected; no winner-based roster is promoted.',
        '', pd.DataFrame(concentration).to_string(index=False), '', 'Reference pair ranking:',
        base_pairs[['label', 'entries', 'gross_hedged_pnl', 'net_before_shared_financing', 'financing_vs_cash', 'net_excess_pnl']].to_string(index=False),
        '', 'Files: cohorts.csv, legs.csv, names.csv, pairs.csv, annual_pairs.csv, episodes.csv, concentration.csv, changed_trades.csv, signal_rank.csv, conditional_groups.csv, winner_loser_features.csv, audit.json']
    (REPORT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(concentration).to_string(index=False), flush=True)
    print(base_pairs[['label', 'entries', 'net_excess_pnl']].to_string(index=False), flush=True)


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((EMPLOYEE/'inputs.json').read_text()); defs = inputs['definitions']; cfg = inputs['actual']; employees = employee_roster()
    coverage = pd.read_csv(MINED_REPORT/'coverage.csv'); selected = set(coverage.loc[coverage.retrain, 'pair'])
    cal = load_calendar(ROOT); cal = cal[(cal>='2010-01-01')&(cal<='2025-08-29')]; live = cal[cal>='2018-01-01']
    market = Market(cal); panels, extras = history_panels(defs, cal)
    def folder(d, model):
        return MINED/d['pair']/model/'full' if model in MODELS and d['pair'] in selected else baseline_results(d['pair'], 'rl', 'full', d['pair'] in employees)
    sources = {str(Path(__file__)):digest(__file__), str(Path('scripts/trade_quality_ledger.py')):digest('scripts/trade_quality_ledger.py'),
        str(CHECKPOINT):digest(CHECKPOINT), 'config/current_contract_reference.json':digest('config/current_contract_reference.json'),
        'data/curve_trading/C_history_manifest.json':digest('data/curve_trading/C_history_manifest.json')}
    for d in defs:
        path = EMPLOYEE/d['pair']/'states.parquet'
        if path.exists(): sources[str(path)] = digest(path)
        added_path = MINED/d['pair']/'states.parquet'
        if added_path.exists(): sources[str(added_path)] = digest(added_path)
    cohort_rows = []; leg_rows = []; pair_rows = []; annual_rows = []; errors = []
    aggregate = {m:pd.Series(0., index=live) for m in ALL_MODELS}
    for i, d in enumerate(defs, 1):
        statuses = {m:json.loads((folder(d, m)/'status.json').read_text()) for m in ALL_MODELS}
        assert all(s['status']=='complete' for s in statuses.values())
        if not any(s['entries'] for s in statuses.values()):
            for model in ALL_MODELS: pair_rows.append(dict(model=model, pair=d['pair'], label=d['label'], industry=d['industry'], entries=0,
                gross_hedged_pnl=0., net_before_shared_financing=0., financing_vs_cash=0., net_excess_pnl=0.))
            continue
        books = {}
        for model, status in statuses.items():
            f = folder(d, model)
            if not status['entries']: continue
            paths = [f/'trades.csv', f/'decisions.csv', Path(status.get('nav_source', f/'daily_nav.parquet')), f/'status.json']
            for path in paths: sources[str(path)] = digest(path)
            t = read_csv(paths[0], ['date', 'cohort']); dec = read_csv(paths[1], ['signal_date', 'entry_date'])
            nav = pd.read_parquet(paths[2]); assert nav.index.equals(live)
            books[model] = (nav, t, dec[dec.status.eq('entered')].set_index('signal_date', drop=False))
        quote_parts = []
        for sid in d['secids']:
            ids = sorted({int(x) for _, t, _ in books.values() for x in t.loc[t.secid.eq(sid), 'optionid'].unique()})
            for year in live.year.unique():
                path = DATA/'history/C'/str(year)/f'{sid}.parquet'
                if path.exists(): quote_parts.append(pd.read_parquet(path, filters=[('optionid', 'in', ids)]))
        quotes = pd.concat(quote_parts, ignore_index=True)
        lookup = {(r.date, int(r.optionid)):r._asdict() for r in quotes.itertuples(index=False)}
        stock = market.stocks[market.stocks.secid.isin(d['secids'])]
        stocks = {sid:stock[stock.secid.eq(sid)].set_index('date').reindex(cal) for sid in d['secids']}
        divs = market.divs[market.divs.secid.isin(d['secids'])]
        for model, status in statuses.items():
            if not status['entries']:
                pair_rows.append(dict(model=model, pair=d['pair'], label=d['label'], industry=d['industry'], entries=0,
                    gross_hedged_pnl=0., net_before_shared_financing=0., financing_vs_cash=0., net_excess_pnl=0.)); continue
            nav, t, entered = books[model]
            legs, daily, finance, vega, gross_vega, error = reconstruct(nav, t, quotes, stock, divs, cfg); errors.append(error)
            leg_sum = legs.gross_hedged_pnl.sum(); net_sum = legs.net_before_shared_financing.sum()
            excess = nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
            np.testing.assert_allclose(net_sum+finance.sum(), excess, atol=1e-6, rtol=0)
            aggregate[model] += daily.sum(axis=1)+finance
            pair_rows.append(dict(model=model, pair=d['pair'], label=d['label'], industry=d['industry'], entries=status['entries'],
                gross_hedged_pnl=leg_sum, net_before_shared_financing=net_sum, financing_vs_cash=finance.sum(), net_excess_pnl=excess))
            for year, h in daily.groupby(daily.index.year):
                annual_rows.append(dict(model=model, pair=d['pair'], label=d['label'], year=year,
                    entries=int(entered.entry_date.dt.year.eq(year).sum()), net_before_shared_financing=h.sum().sum(),
                    financing_vs_cash=finance.loc[h.index].sum(), net_excess_pnl=h.sum().sum()+finance.loc[h.index].sum()))
            for j, cohort in enumerate(daily.columns):
                own = legs[legs.cohort.eq(cohort)].copy(); assert len(own)==2
                e = entered.loc[cohort]; assert isinstance(e, pd.Series)
                mean_vega = (e.vega_a+e.vega_b)/2; l = own[own.role.eq('long')].iloc[0]; s = own[own.role.eq('short')].iloc[0]
                signal = signal_inputs(d, e, own, panels[d['pair']], extras[d['pair']], lookup, stocks)
                end = own.exit_date.max(); first_end = own.exit_date.min()
                period = daily.loc[e.entry_date:end, cohort]; net = float(period.sum())
                active = gross_vega[:, j] > 1e-6
                ratios = np.divide(abs(vega[:, j]), gross_vega[:, j], out=np.zeros(len(live)), where=active)
                row = dict(model=model, pair=d['pair'], label=d['label'], industry=d['industry'], cohort=cohort,
                    entry_date=e.entry_date, exit_date=end, action=e.action, mean_initial_vega=mean_vega,
                    net_before_shared_financing=net, net_per_vega=net/mean_vega, gross_hedged_pnl=float(own.gross_hedged_pnl.sum()),
                    worst_day=period.idxmin(), worst_day_pnl=period.min(), after_first_expiry_pnl=float(period[period.index>first_end].sum()),
                    first_expiry=first_end, days_after_first_expiry=int(((period.index>first_end)&(period.index<=end)).sum()),
                    max_net_vega_fraction=float(ratios.max()), mean_net_vega_fraction=float(ratios[active].mean()) if active.any() else 0.,
                    **signal)
                for role, leg in [('long', l), ('short', s)]:
                    path = forward_path(stocks[leg.secid], leg.entry_date, leg.exit_date)
                    for key, value in path.items(): row[role+'_'+key] = value
                    row[role+'_entry_iv'] = leg.entry_iv
                    row[role+'_realized_minus_iv'] = path['forward_rv']-leg.entry_iv
                for k, leg in own.iterrows():
                    name = d['label'].split('/')[d['secids'].index(int(leg.secid))]
                    lr = dict(model=model, pair=d['pair'], label=d['label'], name=name, **leg.to_dict())
                    lr['net_per_vega'] = leg.net_before_shared_financing/leg.initial_vega
                    lr['profitable'] = leg.net_before_shared_financing>0
                    leg_rows.append(lr)
                cohort_rows.append(row)
            print(f'{d["label"]}: attributed '+', '.join(m+"="+str(len(b[2])) for m, b in books.items()), flush=True)
        del books, quotes, lookup, stock, stocks, legs, daily; gc.collect()
        write_json(OUT/'progress.json', dict(last_pair=d['label'], original_roster_position=i, original_pairs=len(defs),
            cohorts=len(cohort_rows), elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    portfolio_errors = {}
    for model in ALL_MODELS:
        book = pd.read_csv(MINED_REPORT/f'all_{model}_full_portfolio.csv', index_col=0, parse_dates=[0])
        expected = book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
        portfolio_errors[model] = float(abs(aggregate[model]-expected).max())
        np.testing.assert_allclose(aggregate[model], expected, atol=5e-6, rtol=0)
    c, l, p, a = map(pd.DataFrame, [cohort_rows, leg_rows, pair_rows, annual_rows])
    summarize(c, l, p, a)
    reference = json.loads(CHECKPOINT.read_text())
    assert digest(reference['archive'])==reference['archive_sha256']
    for path, sha in reference['files'].items(): assert digest(path)==sha
    for path, sha in sources.items(): assert digest(path)==sha
    audit = dict(status='complete', original_pairs=len(defs), model_cohorts=c.groupby('model').size().to_dict(),
        models=ALL_MODELS, reference_unchanged=True, original_checkpoint_files_verified=len(reference['files']),
        source_sha256=sources, maximum_cohort_ledger_error=max(errors), maximum_portfolio_daily_error=portfolio_errors,
        financing_allocation='Kept separate; actual netted stock execution/borrow allocated pro rata with exact daily reconciliation.',
        fitting=False, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        caveat='Executed trades are selected by the saved policies. Correlations and group outcomes are exploratory, overlapping and conditional on execution, not a new unbiased signal test.')
    write_json(REPORT/'audit.json', audit); write_json(OUT/'task_status.json', dict(status='complete', elapsed_seconds=audit['elapsed_seconds'], peak_rss_bytes=audit['peak_rss_bytes']))
    print({k:v for k,v in audit.items() if k!='source_sha256'}, flush=True)


if __name__ == '__main__': main()
