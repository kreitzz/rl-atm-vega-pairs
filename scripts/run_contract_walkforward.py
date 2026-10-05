"""Monthly actual-contract learning, profitability memory and beta ablations.

One pair at a time. Fixed settings; no tuning on reported test years.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from scripts.contract_engine import Settings, UnresolvedPosition, simulate
from scripts.contract_learning_labels import ShadowLabels, LADDER, profitability_memory, beta_screen
from scripts.run_nn_rl_benchmark import NEW_FEATURES, add_features
from scripts.run_selected_pair_backtest import audit, scores
from scripts.select_atm_contracts import load_calendar

ROOT = Path('data/option_contracts')
CACHE = Path('artifacts/contracts/four_pairs/cache')
DEFINITIONS = [p for p in json.loads(Path('config/universe_gap_pairs.json').read_text()) if p['original_pair']]
MODELS = ['rl_annual_base', 'rl_monthly_base', 'rl_monthly_memory',
          'rl_monthly_beta', 'rl_monthly_combined', 'nn_monthly_combined']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_pair(definition):
    pair = definition['pair']; names = list(definition['names'].values())
    secids = [n['secid'] for n in names]
    q = pd.concat([pd.read_parquet(p) for p in sorted((CACHE/pair).glob('????.parquet'))], ignore_index=True)
    selection = pd.concat([pd.read_parquet(p) for p in sorted((CACHE/pair).glob('*_selection.parquet'))], ignore_index=True)
    stocks = pd.concat([pd.read_parquet(p, filters=[('secid', 'in', secids)])
                        for p in sorted((ROOT/'underlying').glob('year=*.parquet'))], ignore_index=True)
    stocks['date'] = pd.to_datetime(stocks.date)
    zero = pd.read_parquet(ROOT/'zero_curve.parquet'); zero['date'] = pd.to_datetime(zero.date)
    zero['distance'] = abs(zero.days-30)
    zero = zero.sort_values(['date','distance','days']).drop_duplicates('date').sort_values('date')
    zero['rate'] /= 100
    stocks = pd.merge_asof(stocks.sort_values('date'), zero[['date','rate']], on='date', direction='backward', tolerance=pd.Timedelta(days=7))
    divs = pd.read_parquet(ROOT/'distributions.parquet')
    for col in ['ex_date','declare_date']: divs[col] = pd.to_datetime(divs[col])
    divs = divs[divs.secid.isin(secids) & divs.cancel_flag.eq('0') & divs.distr_type.eq('1') &
                divs.currency.eq('USD') & divs.adj_factor.eq(1) & divs.approx_flag.eq('0') & divs.liquid_flag.eq('0')]
    assets = [pd.read_parquet(Path('artifacts/broad_pairs/assets')/f"s{n['secid']}p{n['permno']}.parquet",
                              columns=['ret','trailing_rv','earnings']) for n in names]
    iv = selection.pivot(index='date', columns='secid', values='bs_iv')*100
    panel = pd.DataFrame(index=iv.index)
    for s, asset, suffix in zip(secids, assets, ['amd','nvda']):
        panel[f'iv_{suffix}'] = iv[s]
        panel[f'trailing_rv_{suffix}'] = asset.trailing_rv
        panel[f'{suffix}_earnings_days_capped'] = asset.earnings
    panel['spread'] = panel.iv_amd-panel.iv_nvda
    panel = add_features(panel)
    returns = pd.concat([a.ret.rename(s) for a,s in zip(assets,secids)], axis=1)
    return q, stocks, selection, divs, panel, returns, secids


def daily_moments(states, labels, positions, ends, nstates):
    """Exact daily shadow-book moments, including overlapping cohorts."""
    first = int(positions.min()+1); last = int(ends.max())
    book = np.zeros((last-first+1, nstates*len(LADDER)))
    offsets = np.arange(labels.shape[2])
    for state in range(nstates):
        rows = np.flatnonzero(states == state)
        days = positions[rows, None]+1+offsets[None, :]-first
        valid = (days >= 0) & (days < len(book))
        for action in range(len(LADDER)):
            np.add.at(book[:, state*len(LADDER)+action], days[valid], labels[rows, action, :][valid])
    return book.mean(axis=0), np.cov(book, rowvar=False, ddof=1)


def search_policy(mean, covariance, nstates, seed=2026):
    """Direct Sharpe policy search; no Q-learning or actor-critic claim."""
    rng = np.random.default_rng(seed); na = len(LADDER)
    probabilities = np.full((nstates, na), 1/na)
    best = np.full(nstates, na//2, dtype=int); best_score = 0.
    for _ in range(50):
        u = rng.random((192, nstates))
        choices = (u[:, :, None] > probabilities.cumsum(axis=1)[None, :, :]).sum(axis=2).clip(max=na-1)
        choices[0] = best
        # Include every uniform action, including flat, in every generation.
        choices[1:na+1] = np.arange(na)[:, None]
        columns = choices+np.arange(nstates)[None, :]*na
        m = mean[columns].sum(axis=1)
        var = covariance[columns[:, :, None], columns[:, None, :]].sum(axis=(1,2))
        sharpe = np.sqrt(252)*np.divide(m, np.sqrt(np.maximum(var, 0)), out=np.zeros_like(m), where=var > 1e-18)
        winner = sharpe.argmax()
        if sharpe[winner] > best_score: best, best_score = choices[winner].copy(), float(sharpe[winner])
        elite = choices[np.argsort(sharpe)[-19:]]
        frequency = np.stack([(elite == a).mean(axis=0) for a in range(na)], axis=1)
        probabilities = .97*(.3*probabilities+.7*frequency)+.03/na
    return best, best_score


def fit_rl(x, labels, positions, ends):
    target = (labels[:, -1].sum(axis=1)-labels[:, 0].sum(axis=1))/2
    tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=63, random_state=2026).fit(x, target)
    leaves = np.unique(tree.apply(x)); state = np.searchsorted(leaves, tree.apply(x))
    mean, cov = daily_moments(state, labels, positions, ends, len(leaves))
    policy, objective = search_policy(mean, cov, len(leaves))
    return (tree, leaves, policy), dict(training_shadow_daily_sharpe=objective)


def fit_nn(x, labels):
    scaler = StandardScaler().fit(x)
    # Predict the two directional NET outcomes, so a positive gross prediction
    # is not enough to overcome spread/commission/hedge costs.
    target = labels[:, [0,-1]].sum(axis=2)
    center = target.mean(axis=0); scale = np.maximum(target.std(axis=0), 1e-8)
    nn = MLPRegressor(hidden_layer_sizes=(16,), activation='tanh', solver='lbfgs',
                      alpha=1., max_iter=1000, max_fun=30000, random_state=2026)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always', ConvergenceWarning)
        nn.fit(scaler.transform(x), (target-center)/scale)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in messages)
    return (scaler, nn, center, scale), dict(converged=converged, iterations=nn.n_iter_)


def predict(model, x, neural=False):
    if neural:
        scaler, nn, center, scale = model
        prediction = nn.predict(scaler.transform(x))*scale+center
        choice = prediction.argmax(axis=1)
        return np.where(prediction.max(axis=1) > 0, np.array([-1.,1.])[choice], 0.)
    tree, leaves, policy = model
    return LADDER[policy[np.searchsorted(leaves, tree.apply(x))]]


def train_walkforward(panel, daily, positions, ends, eligible, start, stop, calendar):
    actions = pd.DataFrame(0., index=panel.index[(panel.index >= start) & (panel.index <= stop)], columns=MODELS)
    logs = []; models = {}
    memory = [c for c in panel if c.startswith(('profit_', 'count_'))]
    for month in pd.period_range(start, stop, freq='M'):
        cutoff = month.start_time
        test = (panel.index >= cutoff) & (panel.index < (month+1).start_time) & panel.index.isin(actions.index)
        if not test.any(): continue
        # No incomplete label, future completion, or stale data can train a fit.
        train = (panel.index >= cutoff-pd.DateOffset(years=5)) & (ends < cutoff) & np.isfinite(daily).all(axis=(1,2))
        assert not np.any(train & test)
        for name, features, neural in [('annual',NEW_FEATURES,False), ('base',NEW_FEATURES,False),
                                       ('memory',NEW_FEATURES+memory,False), ('nn',NEW_FEATURES+memory,True)]:
            refit = name != 'annual' or month.month == 1 or name not in models
            if refit and train.sum() >= 504:
                args = (panel.loc[train, features], daily[train])
                model, details = fit_nn(*args) if neural else fit_rl(*args, positions[train], calendar.get_indexer(ends[train]))
                models[name] = model
                logs.append(dict(fit=str(cutoff.date()), policy=name, training_rows=int(train.sum()),
                                 last_outcome=str(ends[train].max().date()), **details))
            if name not in models: continue
            a = predict(models[name], panel.loc[test, features], neural)
            dates = panel.index[test]
            if name == 'annual': actions.loc[dates, 'rl_annual_base'] = a
            if name == 'base':
                actions.loc[dates, 'rl_monthly_base'] = a
                actions.loc[dates, 'rl_monthly_beta'] = a*eligible.loc[dates].to_numpy()
            if name == 'memory':
                actions.loc[dates, 'rl_monthly_memory'] = a
                actions.loc[dates, 'rl_monthly_combined'] = a*eligible.loc[dates].to_numpy()
            if name == 'nn': actions.loc[dates, 'nn_monthly_combined'] = a*eligible.loc[dates].to_numpy()
        if month.month == 1: print(f'  trained through {month}', flush=True)
    return actions, pd.DataFrame(logs)


def evaluate(pair, q, stocks, selection, divs, secids, calendar, actions, cfg, out):
    records = []; navs = {}
    for model in MODELS:
        dest = out/pair/model; dest.mkdir(parents=True, exist_ok=True)
        try:
            nav, trades, decisions = simulate(q[q.date.isin(calendar)], stocks[stocks.date.isin(calendar)],
                                              actions[model], calendar, secids, cfg, divs, selection)
            rates = stocks.groupby('date').rate.mean().reindex(nav.index).shift(1).fillna(0)
            cash_return = rates*nav.index.to_series().diff().dt.days.fillna(0)/365
            nav['cash_benchmark'] = cfg.capital*(1+cash_return).cumprod()
            nav['excess_return'] = nav.return_-cash_return
            checks = audit(nav, trades, stocks, cfg, secids)
            # Engine assertions check every end-of-day cap; retain independent maxima.
            for field, limit in [('gross_hedge_notional',cfg.max_hedge_leverage),
                                  ('capital_reserve',cfg.max_reserve_fraction), ('stress_loss',cfg.max_stress_loss_fraction),
                                  ('gross_vega',cfg.max_gross_vega/cfg.capital)]:
                assert (nav[field]/nav.nav <= limit+1e-10).all()
            nav.to_csv(dest/'daily_nav.csv'); trades.to_csv(dest/'trades.csv',index=False); decisions.to_csv(dest/'decisions.csv',index=False)
            navs[model] = nav
            records.append(dict(pair=pair, model=model, status='complete', entries=int(decisions.status.eq('entered').sum()),
                                **scores(nav,cfg.capital), **checks))
        except UnresolvedPosition as exc:
            records.append(dict(pair=pair, model=model, status='incomplete', error=str(exc)))
            if hasattr(exc, 'partial_nav'): exc.partial_nav.to_csv(dest/'partial_nav.csv')
        print(json.dumps(records[-1]), flush=True)
    return records, navs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--option-penalty-vol-points',type=float,
                        help='Fill at mid and charge dollar vega times this penalty per leg per fill. Omit for legacy quoted-spread execution.')
    parser.add_argument('--start',default='2013-01-01')
    parser.add_argument('--pairs',nargs='+',choices=[p['pair'] for p in DEFINITIONS],default=[p['pair'] for p in DEFINITIONS])
    parser.add_argument('--reuse-labels',action='store_true')
    parser.add_argument('--reuse-actions',action='store_true')
    args = parser.parse_args()
    suffix='' if args.option_penalty_vol_points is None else '_mid_'+format(args.option_penalty_vol_points,'g').replace('.','p')
    args.output=args.output or Path('artifacts/contracts/walkforward'+suffix)
    args.report=args.report or Path('reports/contract_walkforward'+suffix)
    args.output.mkdir(parents=True,exist_ok=True); args.report.mkdir(parents=True,exist_ok=True)
    if len(args.pairs)!=len(set(args.pairs)):raise ValueError('Duplicate pairs are not allowed')
    cfg = Settings(option_spread_fraction=.5 if args.option_penalty_vol_points is None else 0.,
                   option_penalty_vol_points=args.option_penalty_vol_points,
                   risk_controls=True, repair_crossed_held_quotes=True, support_integer_splits=True)
    calendar = load_calendar(ROOT); calendar = calendar[(calendar >= '2010-01-01') & (calendar <= '2025-08-29')]
    eval_cal = calendar[calendar >= args.start]; stop = calendar[-1]-pd.Timedelta(days=35)
    records = []; portfolios = {}; label_audits = []; screening = []
    inputs={str(p):digest(p) for p in [*sorted((ROOT/'underlying').glob('year=*.parquet')),
                                      ROOT/'zero_curve.parquet',ROOT/'distributions.parquet']}
    for definition in DEFINITIONS:
        pair = definition['pair']
        if pair not in args.pairs: continue
        print(f'Loading {pair}',flush=True)
        q, stocks, selection, divs, panel, returns, secids = load_pair(definition)
        folder = args.output/pair; folder.mkdir(parents=True,exist_ok=True)
        panel = panel.loc[panel.index.isin(calendar) & (panel.index <= stop)].copy()
        positions = calendar.get_indexer(panel.index)
        label_path = folder/'labels.npz'
        source_paths=[*sorted((CACHE/pair).glob('*.parquet')),
                      *[Path('artifacts/broad_pairs/assets')/f"s{n['secid']}p{n['permno']}.parquet" for n in definition['names'].values()]]
        sources={**inputs,**{str(p):digest(p) for p in source_paths}}
        label_metadata=dict(settings=asdict(cfg),sources=sources,
                            code={str(p):digest(p) for p in [Path('scripts/contract_learning_labels.py'),Path('scripts/contract_engine.py')]})
        metadata_path=folder/'labels_metadata.json'
        if args.reuse_labels and label_path.exists():
            if not metadata_path.exists() or json.loads(metadata_path.read_text())!=label_metadata:
                raise ValueError(f'{pair}: label inputs changed; rerun without --reuse-labels')
            with np.load(label_path,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['dates'],panel.index.values)
                daily=saved['daily']; end_positions=saved['end_positions']; vegas=saved['vegas']; statuses=saved['statuses']
        else:
            maker = ShadowLabels(q,stocks,selection,calendar,secids,cfg,divs)
            values = [maker.label(int(i)) for i in positions]
            daily=np.stack([v['daily'] for v in values]);end_positions=np.array([v['end_index'] for v in values])
            vegas=np.stack([v['entry_vega'] for v in values]);statuses=np.array([v['status'] for v in values])
            np.savez_compressed(label_path,dates=panel.index.values,daily=daily,end_positions=end_positions,vegas=vegas,statuses=statuses)
            metadata_path.write_text(json.dumps(label_metadata,indent=2)+'\n')
            del maker, values
        ends=calendar[end_positions]
        for status, count in pd.Series(statuses).value_counts().items(): label_audits.append(dict(pair=pair,status=status,rows=int(count)))
        print(f'  labels: {pd.Series(statuses).value_counts().to_dict()}',flush=True)
        rewards = np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
                            out=np.full((len(panel),2),np.nan),where=vegas[:,[0,-1]] > 0)
        panel = panel.join(profitability_memory(panel.index,ends,rewards,vegas[:,[0,-1]] > 0))
        screen = beta_screen(returns, calendar); screen.index.name='date'
        screen.to_csv(folder/'beta_screen.csv'); screening.append(screen.assign(pair=pair))
        action_metadata=dict(start=args.start,stop=str(stop),labels_sha256=digest(label_path),
                             code_sha256=digest(Path(__file__)),
                             states_sha256=hashlib.sha256(pd.util.hash_pandas_object(panel,index=True).values.tobytes()).hexdigest())
        action_meta_path=folder/'actions_metadata.json'
        if args.reuse_actions and (folder/'actions.csv').exists():
            if not action_meta_path.exists() or json.loads(action_meta_path.read_text())!=action_metadata:
                raise ValueError(f'{pair}: training inputs changed; rerun without --reuse-actions')
            actions=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0])
        else:
            actions, fits = train_walkforward(panel,daily,positions,ends,screen.eligible,pd.Timestamp(args.start),stop,calendar)
            actions.to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
            action_meta_path.write_text(json.dumps(action_metadata,indent=2)+'\n')
        panel.to_parquet(folder/'states.parquet')
        rows, navs = evaluate(pair,q,stocks,selection,divs,secids,eval_cal,actions,cfg,args.output)
        records.extend(rows)
        for model, nav in navs.items(): portfolios.setdefault(model,[]).append(nav[['nav','cash_benchmark']])
        pd.DataFrame(records).to_csv(args.report/'status.csv',index=False)
        del q, stocks, selection, panel, daily
    pd.DataFrame(label_audits).to_csv(args.report/'labels.csv',index=False)
    pd.concat(screening).reset_index().to_csv(args.report/'screening.csv',index=False)
    summaries=[];annual=[]
    for model in MODELS:
        navs=portfolios.get(model,[])
        if len(navs) != len(args.pairs):
            summaries.append(dict(model=model,status='incomplete',completed_pairs=len(navs)));continue
        assert all(n.index.equals(navs[0].index) for n in navs)
        book=sum(navs);book['return_']=book.nav.pct_change().fillna(0)
        book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
        capital=len(navs)*cfg.capital
        summaries.append(dict(model=model,status='complete',completed_pairs=len(navs),**scores(book,capital)))
        book.to_csv(args.output/f'{model}_portfolio.csv')
        for year,g in book.groupby(book.index.year):
            previous=capital if g.index[0]==book.index[0] else book.loc[:g.index[0]].nav.iloc[-2]
            annual.append(dict(model=model,year=int(year),total_return=g.nav.iloc[-1]/previous-1,
                               annualized_excess_sharpe=np.sqrt(252)*g.excess_return.mean()/g.excess_return.std() if g.excess_return.std()>0 else np.nan))
    pd.DataFrame(summaries).to_csv(args.report/'portfolio.csv',index=False)
    pd.DataFrame(annual).to_csv(args.report/'annual.csv',index=False)
    manifest=dict(settings=asdict(cfg),pairs=args.pairs,start=args.start,end=str(calendar[-1].date()),
                  train_years=5,min_training_rows=504,tree_leaves=8,tree_min_leaf=63,ladder=LADDER.tolist(),
                  beta_window=126,beta_tolerance=.25,correlation_floor=.5,memory_days=[91,365],
                  training_objective='daily standalone shadow-book excess PnL Sharpe; final evaluation includes NAV scaling and risk limits',
                  memory='completed independent directional shadow trades, normalized by actual mean initial leg vega',
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/contract_learning_labels.py'),Path('scripts/contract_engine.py')]})
    (args.report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__ == '__main__': main()
