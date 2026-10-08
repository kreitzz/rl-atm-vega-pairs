"""Read-only pair attribution of the frozen validation-selected P25 book.

Uses archived fills and the existing audited shared-stock attribution. No fit,
search, hypothetical order, or portfolio replay. Run with .venv-pca312 Python.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[key] = '1'
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from scripts.diagnose_p25_pca import Budget, digest
from scripts.shared_trade_ledger import FIELDS, reconstruct_shared, connected_episodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/p25_strategy_audit/pairs'
ART = ROOT / 'artifacts/contracts/p25_strategy_audit/pairs'
MARKET = ROOT / 'data/submission/artifacts/contracts/p25_signal_transfer/market'
ARCHIVE = ROOT / 'data/p25_validation_split_experiment.zip'
PREFIX = 'artifacts/contracts/p25_validation_split/replays/screened_selected/'
PERIODS = [('full', '2018-01-02', '2025-08-29'), ('validation', '2018-01-02', '2021-12-31'),
           ('evaluation', '2022-01-03', '2025-08-29')]


def sharpe(values):
    return float(np.sqrt(252)*values.mean()/values.std(ddof=1)) if values.std(ddof=1) > 1e-12 else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(ART / 'matplotlib')
    budget = Budget()
    hashes = {}
    manifest_path = ROOT / 'reports/p25_validation_split/bundle_manifest.json'
    archive_manifest = json.loads(manifest_path.read_text())['files']
    submission_path = ROOT / 'data/submission/manifest.json'
    submission = {r['stored']: r['sha256'] for r in json.loads(submission_path.read_text())['files']}
    for path in [ARCHIVE, manifest_path, submission_path]:
        hashes[str(path.relative_to(ROOT))] = digest(path)

    def read(path, **kwargs):
        key = path.relative_to(ROOT).as_posix()
        sha = digest(path)
        assert sha == submission[key], key
        hashes[key] = sha
        return pd.read_parquet(path, use_threads=False, **kwargs)

    with zipfile.ZipFile(ARCHIVE) as archive:
        def bundled(name, csv=False):
            raw = archive.read(PREFIX+name)
            sha = hashlib.sha256(raw).hexdigest()
            assert sha == archive_manifest[PREFIX+name]
            hashes[ARCHIVE.name+'::'+PREFIX+name] = sha
            return pd.read_csv(io.BytesIO(raw)) if csv else pd.read_parquet(io.BytesIO(raw), use_threads=False)
        nav = bundled('daily_nav.parquet')
        trades = bundled('trades.csv', True)
        decisions = bundled('decisions.csv', True)
        scored = bundled('scored_allocations.parquet')
    trades['date'] = pd.to_datetime(trades.date)
    for field in ['entry_date', 'signal_date']:
        decisions[field] = pd.to_datetime(decisions[field])
    assert len(decisions) == len(scored) == 608
    assert decisions.status.eq('entered').sum() == trades.cohort.nunique() == 207
    assert set(trades.reason) == {'entry', 'expiry_settlement'}
    assert nav.index.is_unique and nav.index.is_monotonic_increasing
    ids = trades.optionid.unique().tolist()
    parts = []
    for year in range(2018, 2026):
        stage = time.monotonic()
        parts.append(read(MARKET/f'{year}_quotes.parquet', filters=[('optionid','in',ids)]))
        budget.rest(stage)
    stage = time.monotonic()
    quotes = pd.concat(parts, ignore_index=True)
    del parts
    stocks = read(MARKET/'stocks.parquet', columns=['date','secid','close','cfadj'], filters=[('secid','in',trades.secid.unique().tolist())])
    dividends = read(MARKET/'dividends.parquet', columns=['secid','ex_date','amount'])
    legs, daily, financing, vegas, grossvegas, error, components = reconstruct_shared(
        nav, trades, quotes, stocks, dividends, {'stock_cost_bps':1., 'stock_borrow_rate':.003})
    del quotes, stocks, dividends
    gc.collect()
    budget.rest(stage)
    stage = time.monotonic()
    # Preserve cohort accounting before any grouping; all calendar rows included.
    daily.to_parquet(ART/'cohort_daily.parquet')
    legs['pair'] = legs.cohort.str.split('|').str[1]
    meta = legs.groupby('cohort').agg(pair=('pair','first'), entry_date=('entry_date','min'),
        exit_date=('exit_date','max'), mean_initial_vega=('initial_vega','mean'),
        gross_initial_vega=('initial_vega','sum'), contracts=('contracts','sum'))
    meta['net_lifecycle_pnl'] = daily.sum()
    meta['episode'] = connected_episodes(meta)
    d = decisions.copy()
    d['cohort'] = d.signal_date.dt.strftime('%Y-%m-%d')+'|'+d.pair+'|P25'
    d['requested_size_fraction'] = d.action.abs()
    meta['requested_size_fraction'] = d.set_index('cohort').requested_size_fraction.reindex(meta.index)
    assets_path = ROOT/'data/submission/artifacts/broad_pairs/assets.json'
    hashes[str(assets_path.relative_to(ROOT))] = digest(assets_path)
    assets = {str(r['secid']):r['ticker'] for r in json.loads(assets_path.read_text())}
    pair_labels_path = ROOT/'data/submission/artifacts/broad_pairs/pairs.json'
    hashes[str(pair_labels_path.relative_to(ROOT))] = digest(pair_labels_path)
    fallback_labels = {'__'.join(s.split('p')[0][1:] for s in r['pair'].split('__')):r['label']
                       for r in json.loads(pair_labels_path.read_text())}
    pairs = sorted(set(d.pair))
    labels = {p:'/'.join(assets.get(s,s) for s in p.split('__')) for p in pairs}
    labels = {p:fallback_labels.get(p,p) if '<NA>' in label else label for p,label in labels.items()}
    meta['label'] = meta.pair.map(labels)
    meta.to_csv(OUT/'cohorts.csv')
    legs.to_csv(OUT/'legs.csv', index=False)
    pair_daily = daily.T.groupby(meta.pair).sum().T.reindex(columns=pairs, fill_value=0.)
    pair_gross = pd.DataFrame(grossvegas,index=daily.index,columns=daily.columns).T.groupby(meta.pair).sum().T.reindex(columns=pairs,fill_value=0.)
    pair_netvega = pd.DataFrame(vegas,index=daily.index,columns=daily.columns).T.groupby(meta.pair).sum().T.reindex(columns=pairs,fill_value=0.)
    pair_daily.to_parquet(ART/'pair_daily.parquet')
    pair_gross.to_parquet(ART/'pair_gross_vega.parquet')
    financing.to_frame('shared_financing_vs_cash').to_parquet(ART/'shared_financing.parquet')
    pc = {field:table.T.groupby(meta.pair).sum().T.reindex(columns=pairs,fill_value=0.) for field,table in components.items()}
    del components, vegas, grossvegas
    gc.collect()
    rows, recon, concentration, leaveout, ddrows = [], [], [], [], []
    # Dollar attribution is exact. For return attribution, use the actual prior
    # portfolio NAV denominator and leave shared financing/denominator effects separate.
    prior_nav = nav.nav.shift(1).fillna(nav.nav.iloc[0])
    pair_returns = pair_daily.div(prior_nav,axis=0)
    shared_returns = nav.excess_return-pair_returns.sum(axis=1)
    ratio = nav.nav/nav.cash_benchmark
    ratio_parts = pair_daily.div(nav.cash_benchmark,axis=0)
    ratio_shared = ratio.diff().fillna(0)-ratio_parts.sum(axis=1)
    for period, first, last in PERIODS+[(str(y),f'{y}-01-01',f'{y}-12-31') for y in range(2018,2026)]:
        idx = nav.loc[first:last].index
        first_pos = nav.index.get_loc(idx[0])
        prev = nav.index[max(0,first_pos-1)]
        ret_idx = idx[1:] if first_pos == 0 else idx
        returns = nav.loc[ret_idx,'excess_return']
        total = pair_daily.loc[idx].sum()
        funding = financing.loc[idx].sum()
        expected = (nav.nav-nav.cash_benchmark).loc[idx[-1]]-(nav.nav-nav.cash_benchmark).loc[prev]
        assert abs(total.sum()+funding-expected)<1e-5
        r = pd.concat([pd.Series([ratio.loc[prev]],index=[prev]),ratio.loc[idx]])
        dd = r/r.cummax()-1
        trough_pos = int(np.argmin(dd.to_numpy()))
        peak_pos = int(np.argmax(r.iloc[:trough_pos+1].to_numpy()))
        peak, trough = r.index[peak_pos],r.index[trough_pos]
        draw_idx = nav.index[(nav.index>peak)&(nav.index<=trough)]
        ddparts = ratio_parts.loc[draw_idx].sum()/ratio.loc[peak]*100
        shared_dd = ratio_shared.loc[draw_idx].sum()/ratio.loc[peak]*100
        assert abs(ddparts.sum()+shared_dd-100*dd.min())<1e-9
        sharpe_all = sharpe(returns)
        variance = returns.var(ddof=1)
        shared_var = float(shared_returns.loc[ret_idx].cov(returns)/variance) if variance else np.nan
        recon.append(dict(period=period,start=str(idx[0].date()),end=str(idx[-1].date()),
            proposals=int(d.signal_date.between(first,last).sum()), entries=int((d.status.eq('entered')&d.entry_date.between(first,last)).sum()),
            net_pair_pnl=total.sum(),shared_financing_vs_cash=funding,portfolio_profit_above_cash=expected,
            reconciliation_error=total.sum()+funding-expected,annualized_excess_sharpe=sharpe_all,
            max_cash_relative_drawdown_pct=100*dd.min(),drawdown_peak=str(peak.date()),drawdown_trough=str(trough.date()),
            shared_drawdown_contribution_pct=shared_dd,shared_return_variance_contribution=shared_var))
        positive = total.clip(lower=0).sort_values(ascending=False)
        losses = -total.clip(upper=0)
        concentration.append(dict(period=period,positive_pairs=int(total.gt(0).sum()),negative_pairs=int(total.lt(0).sum()),
            zero_pairs=int(total.eq(0).sum()),positive_pair_profit=positive.sum(),negative_pair_loss=losses.sum(),
            top_positive_pair=positive.index[0],top_positive_label=labels[positive.index[0]],
            top1_share_positive_profit=positive.iloc[:1].sum()/positive.sum(),
            top3_share_positive_profit=positive.iloc[:3].sum()/positive.sum(),
            top5_share_positive_profit=positive.iloc[:5].sum()/positive.sum(),
            top1_share_net_pair_pnl=positive.iloc[:1].sum()/total.sum(),
            top3_share_net_pair_pnl=positive.iloc[:3].sum()/total.sum(),
            positive_profit_hhi=float(((positive/positive.sum())**2).sum())))
        for pair in pairs:
            prop = d.loc[d.pair.eq(pair)&d.signal_date.between(first,last)]
            entries = d.loc[d.pair.eq(pair)&d.status.eq('entered')&d.entry_date.between(first,last)]
            own = meta.loc[meta.pair.eq(pair)]
            new = own.loc[own.entry_date.between(first,last)]
            lifecycle = new.net_lifecycle_pnl
            carried = own.index[own.entry_date.lt(idx[0])]
            vals = {f:float(pc[f].loc[idx,pair].sum()) for f in FIELDS}
            var_share = float(pair_returns.loc[ret_idx,pair].cov(returns)/variance) if variance else np.nan
            seq = np.r_[0.,pair_daily.loc[idx,pair].cumsum().to_numpy()]
            rows.append(dict(period=period,pair=pair,label=labels[pair],proposals=len(prop),entries=len(entries),
                contributing_cohorts=int(daily.loc[idx,own.index].abs().sum().gt(0).sum()),
                new_entry_overlap_episodes=new.episode.nunique(),
                mean_requested_size=prop.requested_size_fraction.mean(),mean_accepted_size=entries.requested_size_fraction.mean(),
                mean_initial_leg_vega=new.mean_initial_vega.mean(),sum_initial_gross_vega=new.gross_initial_vega.sum(),
                mean_daily_gross_vega=pair_gross.loc[idx,pair].mean(),max_daily_gross_vega=pair_gross.loc[idx,pair].max(),
                max_absolute_net_vega=pair_netvega.loc[idx,pair].abs().max(),active_sessions=int(pair_gross.loc[idx,pair].gt(0).sum()),
                calendar_net_before_shared_financing=total[pair],carried_inventory_pnl=daily.loc[idx,carried].to_numpy().sum(),
                new_entry_lifecycle_pnl=lifecycle.sum(),new_entry_lifecycle_win_fraction=lifecycle.gt(0).mean() if len(lifecycle) else np.nan,
                worst_new_cohort=lifecycle.min(),best_new_cohort=lifecycle.max(),
                max_pair_accounting_dollar_drawdown=float((seq-np.maximum.accumulate(seq)).min()),
                return_variance_contribution=var_share,portfolio_drawdown_contribution_pct=ddparts[pair],**vals))
            leaveout.append(dict(period=period,pair=pair,label=labels[pair],
                original_sharpe=sharpe_all,accounting_subtraction_sharpe=sharpe(returns-pair_returns.loc[ret_idx,pair]),
                accounting_subtraction_dollar_profit=expected-total[pair],removed_pair_pnl=total[pair],
                recomputed_admission=False,recomputed_hedge_netting=False,recomputed_financing=False))
            ddrows.append(dict(period=period,pair=pair,label=labels[pair],peak=str(peak.date()),trough=str(trough.date()),
                cash_relative_drawdown_contribution_pct=ddparts[pair],interval_pair_dollar_pnl=pair_daily.loc[draw_idx,pair].sum()))
    results = pd.DataFrame(rows)
    reconciliations = pd.DataFrame(recon)
    for row in recon:
        own = results.loc[results.period.eq(row['period'])]
        assert abs(own.return_variance_contribution.sum()+row['shared_return_variance_contribution']-1)<1e-9
    results.to_csv(OUT/'pair_periods.csv',index=False)
    reconciliations.to_csv(OUT/'reconciliation.csv',index=False)
    pd.DataFrame(concentration).to_csv(OUT/'concentration.csv',index=False)
    pd.DataFrame(leaveout).to_csv(OUT/'accounting_leave_one_pair_out.csv',index=False)
    pd.DataFrame(ddrows).to_csv(OUT/'drawdown_contributions.csv',index=False)
    d.assign(label=d.pair.map(labels)).to_csv(OUT/'proposals_and_admission.csv',index=False)
    d.assign(label=d.pair.map(labels)).groupby(['pair','label','status']).size().rename('count').reset_index().to_csv(OUT/'admission_status.csv',index=False)
    budget.rest(stage)
    stage = time.monotonic()
    plot(results,labels)
    budget.rest(stage)
    evaluation = results.loc[results.period.eq('evaluation')].sort_values('calendar_net_before_shared_financing',ascending=False)
    validation = results.loc[results.period.eq('validation')].set_index('pair')
    top = evaluation.iloc[0]
    c = next(x for x in concentration if x['period']=='evaluation')
    rr = next(x for x in recon if x['period']=='evaluation')
    both = evaluation.loc[evaluation.pair.map(validation.calendar_net_before_shared_financing).gt(0)&evaluation.calendar_net_before_shared_financing.gt(0)]
    active_both = evaluation.loc[evaluation.pair.map(validation.contributing_cohorts).gt(0)&evaluation.contributing_cohorts.gt(0)]
    text = [
        'PAIR AUDIT: CURRENT 27-INPUT VALIDATION-SELECTED P25 POLICY',
        'Frozen archived 608 proposals, 207 accepted option-pair cohorts; 2018-01-02 through 2025-08-29. No model fitting or backtest rerun.',
        '',
        f"Later evaluation: {rr['entries']} new entries, ${rr['portfolio_profit_above_cash']:,.2f} profit above cash, excess Sharpe {rr['annualized_excess_sharpe']:.6f}.",
        f"Pair-trading attribution totals ${rr['net_pair_pnl']:,.2f}; shared cash financing versus the cash benchmark adds ${rr['shared_financing_vs_cash']:,.2f}.",
        f"{c['positive_pairs']} profitable pairs, {c['negative_pairs']} losing pairs, {c['zero_pairs']} inactive/zero pairs in the later period, across {len(pairs)} pairs with at least one qualified proposal over the full history.",
        f"Largest contributor is {top.label}: ${top.calendar_net_before_shared_financing:,.2f}, {top.entries} new entries, {top.new_entry_overlap_episodes} overlapping-entry episodes.",
        f"Top one/three/five positive pairs supply {c['top1_share_positive_profit']:.1%}/{c['top3_share_positive_profit']:.1%}/{c['top5_share_positive_profit']:.1%} of total positive pair profit. Top three equal {c['top3_share_net_pair_pnl']:.1%} of net pair profit after losing pairs.",
        f"{len(both)} of {len(active_both)} pairs active in both periods are positive in both validation and later evaluation. This does not establish a stable pair edge: sizes, dates, overlapping positions, opportunity selection and market regimes differ.",
        f"Worst later cash-relative portfolio drawdown: {rr['max_cash_relative_drawdown_pct']:.6f}% from {rr['drawdown_peak']} to {rr['drawdown_trough']}; drawdown_contributions.csv sums exactly to that decline including the separate shared term in reconciliation.csv.",
        'Annual pair net P&L: 2022 $175,530.17; 2023 $45,777.19; 2024 -$30,868.97; 2025 through August $20,823.80. Thus2022 generated83.1% of later net pair profit and66of110 later entries; performance is concentrated in time as well as pairs.',
        '', 'LATER PAIR RESULTS (calendar P&L after attributed costs, before shared financing):',
    ]
    text += [f"{r.label:16} proposals={r.proposals:3} entries={r.entries:3} episodes={r.new_entry_overlap_episodes:3} net=${r.calendar_net_before_shared_financing:10,.2f} mean_size={r.mean_accepted_size:.1%}" for r in evaluation.itertuples()]
    text += ['', 'ACCOUNTING AND INTERPRETATION',
        'P&L is attributed on the calendar day it occurs. Open positions at 2022/year boundaries are carried in; entry-year lifecycle outcomes are a separate column and must not replace calendar P&L.',
        'Each option leg uses its actual saved fills and observed daily marks/Greeks. All option values, stock values, gross hedge notional, gross/net vega, dividends, option costs, netted stock execution costs, borrow costs and daily funded P&L reconcile.',
        'Stock execution costs are allocated among cohorts pro rata to absolute hedge changes for each underlying; actual shared short-stock borrow costs are allocated by short-hedge amounts. These allocations are additive conventions, not uniquely identifiable standalone pair costs. Fees and penalties are positive cost columns; option, stock and dividend columns are signed P&L.',
        'Shared cash interest minus benchmark interest remains separate; it is not arbitrarily assigned to pairs. It reflects actual option/stock cash flows and interest on accumulated strategy profits, including pre-2022 profits carried into evaluation; it is not independent pair alpha. For additive return-variance contributions, pair dollars use actual prior total NAV; the shared residual also includes differing funded-versus-benchmark return denominators. Variance contribution is covariance(pair contribution, portfolio return)/portfolio variance and may be negative.',
        'Drawdown contributions allocate the change in NAV/cash benchmark over the actual portfolio peak-to-trough interval; they are percentage points of that drawdown. Pair-specific dollar drawdowns are accounting paths without allocated financing/capital.',
        'Leave-one-pair-out is accounting subtraction only: original portfolio excess returns minus that pair\'s daily P&L divided by actual prior NAV. It does not re-run admission, contract sizing, cash, costs, hedges or financing. It is a concentration diagnostic, not the Sharpe of a tradable strategy without the pair.',
        'An entry can overlap another entry in the same pair. Episode counts merge overlapping holding intervals; they remain dependent across pairs and market regimes. Small trade/episode counts limit confidence; no independent-trade significance claims are made.',
        'Gross initial vega sums both legs in dollars per volatility point; mean initial leg vega measures actual integer-filled entry exposure. Mean daily exposure includes flat sessions. Requested size is the tree\'s fraction of the entry budget, not a portfolio weight.',
        'Pair labels use bundled identifier-to-ticker metadata and are display labels; the missing ticker for SECID109497 is filled from the bundled pair roster as RTN (Raytheon). SECIDs remain the stable keys. The census contains46pairs with a qualified proposal,44with accepted entries, not every upstream candidate pair. The chart shows32pairs active in evaluation; all46are included in the tables.',
        'All history has already been inspected. Results are descriptive/exploratory and do not constitute a fresh holdout or establish causality. A true pair-removal test requires a newly declared funded replay.',
        '',f'Daily reconstruction maximum absolute error: ${error:.12f}.',
        'Files: pair_periods.csv; reconciliation.csv; concentration.csv; accounting_leave_one_pair_out.csv; drawdown_contributions.csv; proposals_and_admission.csv; admission_status.csv; cohorts.csv; legs.csv; pair_performance.png; protocol.json.'
    ]
    (OUT/'Findings.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    for path in [Path(__file__),ROOT/'scripts/shared_trade_ledger.py',ROOT/'scripts/diagnose_p25_pca.py']:
        hashes[str(path.relative_to(ROOT))] = digest(path)
    budget.check()
    protocol = dict(status='complete',baseline='screened_selected 27-input monthly18month 50/75/100% $10m',
        no_fits=True,no_new_replay=True,proposal_count=len(d),entry_count=len(meta),pair_count=len(pairs),
        source_hashes=hashes,max_daily_reconstruction_error=error,threads=1,duty_cycle=.2,rss_limit_mib=384,
        peak_rss_mib=budget.peak,all_history_previously_inspected=True,
        outputs={str(p.relative_to(OUT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='protocol.json'})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    print(reconciliations.head(3).to_string(index=False),flush=True)
    print(evaluation[['label','proposals','entries','calendar_net_before_shared_financing']].to_string(index=False),flush=True)
    print(f'Complete; peak RSS {budget.peak:.1f} MiB; max daily reconciliation error {error}',flush=True)


def plot(results,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    later = results.loc[results.period.eq('evaluation')].sort_values('calendar_net_before_shared_financing')
    active = later.loc[later.contributing_cohorts.gt(0)]
    fig,axes=plt.subplots(1,2,figsize=(14,max(5,len(active)*.36)),layout='constrained',sharex=True)
    for ax,period,title in zip(axes,['validation','evaluation'],['2018–2021 validation','2022–August 2025 evaluation']):
        own=results.loc[results.period.eq(period)].set_index('pair').reindex(active.pair)
        val=own.calendar_net_before_shared_financing/1000
        ax.barh(range(len(own)),val,color=np.where(val>=0,'#268978','#c55757'))
        ax.set_yticks(range(len(own)),[f'{labels[p]}  (n={int(n)})' for p,n in zip(own.index,own.entries)])
        ax.axvline(0,color='#666',lw=.8)
        ax.set_xlabel('Calendar net P&L, $ thousands; shared financing excluded')
        ax.set_title(title)
    fig.suptitle('Pairs active in 2022–2025: contributions to the frozen current portfolio\nn = new accepted entries; carried inventory contributes across boundaries',fontsize=13)
    fig.savefig(OUT/'pair_performance.png',dpi=160)
    plt.close(fig)


def refresh_presentation():
    """Refresh labels/layout/explanatory text from saved accounting tables."""
    budget = Budget()
    started = time.monotonic()
    os.environ['MPLCONFIGDIR'] = str(ART/'matplotlib')
    for path in OUT.glob('*.csv'):
        original = path.read_text(encoding='utf-8')
        corrected = original.replace('NOC/<NA>', 'NOC/RTN')
        if corrected != original:
            path.write_text(corrected,encoding='utf-8')
    results = pd.read_csv(OUT/'pair_periods.csv',dtype={'period':str})
    labels = results.drop_duplicates('pair').set_index('pair').label.to_dict()
    val = results.loc[results.period.eq('validation')].set_index('pair')
    later = results.loc[results.period.eq('evaluation')]
    active_both = later.loc[later.pair.map(val.contributing_cohorts).gt(0)&later.contributing_cohorts.gt(0)]
    positive_both = active_both.loc[active_both.pair.map(val.calendar_net_before_shared_financing).gt(0)&active_both.calendar_net_before_shared_financing.gt(0)]
    findings_path = OUT/'Findings.txt'
    findings = findings_path.read_text(encoding='utf-8').replace('NOC/<NA>','NOC/RTN')
    findings = findings.replace('3 pairs are positive in both validation and later evaluation.',
        f'{len(positive_both)} of {len(active_both)} pairs active in both periods are positive in both validation and later evaluation. The common-active denominator includes carried positions; CCL/NCLH contributes $14.60 in evaluation with no new entries.')
    if 'It reflects actual option/stock cash flows' not in findings:
        findings = findings.replace('Shared cash interest minus benchmark interest remains separate; it is not arbitrarily assigned to pairs.',
            'Shared cash interest minus benchmark interest remains separate; it is not arbitrarily assigned to pairs. It reflects actual option/stock cash flows and interest on accumulated strategy profits, including pre-2022 profits carried into evaluation; it is not independent pair alpha.')
    findings = findings.replace('Pair labels use bundled identifier-to-ticker metadata and are display labels; SECIDs remain the stable keys. The census contains only pairs with a qualified proposal, not every upstream candidate pair.',
        'Pair labels use bundled identifier-to-ticker metadata and are display labels; the missing ticker for SECID 109497 is filled from the bundled pair roster as RTN (Raytheon). SECIDs remain the stable keys. The census contains 46 pairs with qualified proposals, 44 with accepted entries, not every upstream candidate pair. The chart shows 32 pairs active in evaluation; all 46 are included in the tables.')
    if 'Annual pair net P&L:' not in findings:
        findings += '\nAnnual pair net P&L: 2022 $175,530.17; 2023 $45,777.19; 2024 -$30,868.97; 2025 through August $20,823.80. Thus 2022 generated 83.1% of later net pair profit and 66 of 110 later entries; performance is concentrated in time as well as pairs.\n'
    findings_path.write_text(findings,encoding='utf-8')
    plot(results,labels)
    budget.rest(started)
    protocol_path = OUT/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for path in [Path(__file__),ROOT/'data/submission/artifacts/broad_pairs/pairs.json']:
        protocol['source_hashes'][str(path.relative_to(ROOT))] = digest(path)
    protocol['outputs'] = {str(p.relative_to(OUT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='protocol.json'}
    protocol['presentation_refreshed'] = True
    protocol['common_active_pairs'] = len(active_both)
    protocol['profitable_in_both'] = len(positive_both)
    protocol['presentation_peak_rss_mib'] = budget.peak
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    print(f'Presentation complete: {len(positive_both)} / {len(active_both)} common-active pairs positive in both; peak RSS {budget.peak:.1f} MiB',flush=True)


if __name__=='__main__':
    refresh_presentation() if '--refresh-presentation' in sys.argv else main()
