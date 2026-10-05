"""Matched ETF constituent comparison, including idle capital and annual results."""
import json
import numpy as np
import pandas as pd
from scripts.run_tech_etf_pilot import OUTPUT, REPORT, definitions, eligibility
from scripts.diagnose_current_contract_losses import read_csv

VARIANTS=['tight','no_beta_cutoff']
MODELS=['rl_sized','nn_fixed']


def stats(book):
    r=book.nav.pct_change().fillna(0)
    e=r-book.cash_benchmark.pct_change().fillna(0)
    return r,e


def main():
    defs=definitions();frames=[];gates=[];coverage=[]
    fingerprints=[json.loads((OUTPUT/v/'inputs.json').read_text()) for v in VARIANTS]
    for key in ['settings','definitions','sources','filters','assets','code','evaluation_start']:
        assert fingerprints[0][key]==fingerprints[1][key],key
    for d in defs:
        a=pd.read_parquet(OUTPUT/'tight'/d['pair']/'states.parquet')
        b=pd.read_parquet(OUTPUT/'no_beta_cutoff'/d['pair']/'states.parquet')
        pd.testing.assert_frame_equal(a.drop(columns='eligible'),b.drop(columns='eligible'))
        with np.load(OUTPUT/'tight'/d['pair']/'labels.npz') as x, np.load(OUTPUT/'no_beta_cutoff'/d['pair']/'labels.npz') as y:
            for key in x.files:np.testing.assert_equal(x[key],y[key])
    for variant in VARIANTS:
        status=pd.read_csv(REPORT/variant/'status.csv');fits=[]
        for d in defs:
            folder=OUTPUT/variant/d['pair'];panel=pd.read_parquet(folder/'states.parquet')
            coverage.append(dict(variant=variant,pair=d['pair'],state_rows=len(panel),
                first_state=str(panel.index.min().date()),last_state=str(panel.index.max().date())))
            np.testing.assert_array_equal(panel.eligible,eligibility(panel,variant))
            actions=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0])
            assert (actions.index>=pd.Timestamp('2019-01-01')).all()
            f=read_csv(folder/'fits.csv')
            if len(f):
                for end,start in [('last_training_outcome','validation_start'),('last_recent_outcome','fit')]:
                    if end in f:
                        use=f[end].notna();assert (pd.to_datetime(f.loc[use,end])<pd.to_datetime(f.loc[use,start])).all()
                f['pair']=d['pair'];fits.append(f)
            for model in MODELS:
                row=status[status.pair.eq(d['pair'])&status.model.eq(model)].iloc[0].to_dict()
                if row['status']=='complete':
                    nav=pd.read_parquet(folder/model/'daily_nav.parquet')
                    row['net_excess_pnl']=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
                    row['execution_cost']=(nav.option_execution_cost+nav.hedge_cost).sum()
                frames.append(row)
            window=panel.loc['2019-01-01':]
            gates.append(dict(variant=variant,pair=d['pair'],state_days=len(window),eligible_days=int(window.eligible.sum()),
                              median_beta=window.beta.median(),median_correlation=window.correlation.median()))
        pd.concat(fits,ignore_index=True).to_csv(REPORT/variant/'fits.csv',index=False)
    pairs=pd.DataFrame(frames);pairs.to_csv(REPORT/'pair_results.csv',index=False)
    pd.DataFrame(gates).to_csv(REPORT/'eligibility.csv',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    failed=set(pairs.loc[pairs.status.ne('complete'),'pair'])
    matched=[d for d in defs if d['pair'] not in failed]
    (REPORT/'matched_pairs.json').write_text(json.dumps(matched,indent=2)+'\n')
    rows=[];annual=[]
    for variant in VARIANTS:
        for model in MODELS:
            books=[pd.read_parquet(OUTPUT/variant/d['pair']/model/'daily_nav.parquet',columns=['nav','cash_benchmark']) for d in matched]
            assert books and all(b.index.equals(books[0].index) for b in books)
            book=sum(books);book.to_csv(REPORT/f'{variant}_{model}_portfolio.csv')
            r,e=stats(book);use=pairs.variant.eq(variant)&pairs.model.eq(model)&pairs.pair.isin([d['pair'] for d in matched])
            rows.append(dict(variant=variant,model=model,pairs=len(matched),active_pairs=int(pairs.loc[use,'entries'].gt(0).sum()),
                entries=int(pairs.loc[use,'entries'].sum()),net_excess_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1],
                annualized_excess_sharpe=np.sqrt(252)*e.iloc[1:].mean()/e.iloc[1:].std() if e.std()>1e-12 else np.nan,
                total_return_percent=100*(book.nav.iloc[-1]/book.nav.iloc[0]-1)))
            for year,g in book.groupby(book.index.year):
                er=e.loc[g.index];rr=r.loc[g.index]
                annual.append(dict(variant=variant,model=model,year=year,return_percent=100*((1+rr).prod()-1),
                    annualized_excess_sharpe=np.sqrt(252)*er.mean()/er.std() if er.std()>1e-12 else np.nan))
    summary=pd.DataFrame(rows);summary.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    lines=['# Semiconductor constituents versus SMH','',
        'Eight fixed pairs: SMH against AMD, AMAT, INTC, LRCX, MU, NVDA, TER and QRVO. These are the cached stock histories available from the [September 2017 SMH holdings](https://www.sec.gov/Archives/edgar/data/1137360/000093041317004003/c89776_ncsr.htm). This is a frozen historical constituent roster, not reconstructed daily membership. Selection precedes inspecting this experiment’s P&L, but uses retrospective data availability and previously explored stocks.','',
        'SMH option and underlying history was newly downloaded from WRDS/OptionMetrics for 2012–2025, with bounded-memory streaming. The ETF starts in December 2011; the earlier Semiconductor HOLDRS security is not spliced into it. SMH states use direct CRSP ex-distribution returns (legacy dsf through 2024 and dsf_v2 in 2025), 30-calendar-day realized volatility and an ETF earnings sentinel of 31. Existing stock states retain their prior preprocessing, including standardized-IV availability gaps. Actual contract IV and curve features are used for both legs.','',
        'Evaluation January 2019–August 29, 2025. RL has the existing nine-level sizing ladder; NN uses fixed size. Both retain VIX, profitability memory, existing curve/contract states, monthly retraining, the same training windows and validation gates. Separate retraining compares beta 0.85–1.15 with no beta cutoff. Both require correlation ≥0.5 and retain finite beta as a feature; beta is SMH-on-stock and is lagged then frozen monthly.','',
        'Execution and risk remain unchanged: near-ATM calls targeting 30 days, daily close delta hedging, hold to expiry, mid plus 0.1 vol point per leg on entry, $0.65 per contract, 1 bp stock execution, stock borrow and cash financing. Capital is $1m per pair, including flat sleeves. Reserve, hedge and stress constraints remain per pair; the combined portfolio has no additional shared-SMH risk limit.','',
        f'{len(matched)}/8 pairs completed under all four configurations and enter the matched aggregate. Incomplete runs are disclosed in pair_results.csv.','',
        '| Beta filter | Model | Active pairs | Entries | Annualized excess Sharpe | Net excess P&L |',
        '|---|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.variant} | {row.model} | {row.active_pairs} | {row.entries} | {row.annualized_excess_sharpe:.3f} | ${row.net_excess_pnl:,.0f} |')
    lines+=['','| Pair | Tight RL entries | Tight RL P&L above cash | No-cutoff RL entries | No-cutoff RL P&L above cash |',
            '|---|---:|---:|---:|---:|']
    for d in defs:
        own=pairs[pairs.pair.eq(d['pair'])&pairs.model.eq('rl_sized')].set_index('variant')
        cells=[]
        for variant in VARIANTS:
            row=own.loc[variant]
            cells.append(f'{int(row.entries)} | ${row.net_excess_pnl:,.0f}' if row.status=='complete' else '— | incomplete')
        lines.append(f'| {d["label"]} | '+ ' | '.join(cells)+' |')
    lines+=['','Sharpe is daily cash-excess return annualized by √252, including idle periods. Annual percentage returns include interest on unused cash. 2025 is partial. The monthly validation gate evaluates the proposal before the recent-data refit rather than the exact deployed weights. NN convergence warnings and unresolved training labels are recorded below. This is exploratory research, not an untouched holdout.','',
        'Cross-variant checks require identical standalone payoff labels, features, costs, risk limits and evaluation dates. Only beta eligibility changes. Calls versus puts have not been tested: the current cache and replay remain calls-only. American early assignment is not modeled.','',
        '[Pair results](tech_etf_pilot/pair_results.csv) · [Yearly returns and Sharpe](tech_etf_pilot/annual.csv) · [Beta eligibility](tech_etf_pilot/eligibility.csv) · [State coverage](tech_etf_pilot/coverage.csv) · [Comparison](tech_etf_pilot/comparison.csv)','']
    lines+=['SMH has 3,052 selected ATM contract dates across 2012–August 2025; early coverage is thinner. Raw calls total 1,019,744 rows (42.4 MiB compressed). SMH split-adjusted OptionMetrics close returns agree with CRSP ex-distribution returns within 0.00000059 across 3,434 overlapping observations. The 151-test suite passes. [SMH contract coverage](tech_etf_pilot/smh_contract_coverage.csv).','']
    lines+=['SMH/MU is excluded consistently from all four aggregates: both no-cutoff books stop on May 5, 2023 because the held SMH inventory does not pass the engine’s existing verified-integer-split checks. This is a simulation limitation, not a trading loss or a zero-return result. Other complete books and all individual statuses remain visible in the CSV.','',
        'The tight-filter aggregate trades only SMH/INTC, with five entries. Its positive Sharpe is not a broad semiconductor result. Without the cutoff, all seven matched RL pairs lose money above cash; AMAT, TER and QRVO account for roughly $101k of the $126k loss. No-cutoff NN has one profitable pair, QRVO, which does not offset its other losses.','']
    for variant in VARIANTS:
        path=REPORT/variant/'audit.json'
        if path.exists():
            a=json.loads(path.read_text());m=json.loads((REPORT/variant/'manifest.json').read_text())
            lines.append(f'{variant}: audit passed for {a["pairs_checked"]} pairs and {a["label_rows"]:,} label rows; {a["unresolved_labels"]} unresolved shadow labels excluded from training. NN convergence warnings: {a["nn_validation_warnings"]}/{a["nn_validation_fits"]} validation fits, {a["nn_recent_warnings"]}/{a["nn_recent_fits"]} recent fits. Runtime {m["elapsed_seconds"]:.0f}s, two workers, peak worker RSS {m["max_worker_peak_rss_bytes"]/1024**3:.2f} GiB. [Audit](tech_etf_pilot/{variant}/audit.json).\n')
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(summary.to_string(index=False));print(pairs[['variant','label','model','status','entries','net_excess_pnl']].to_string(index=False))


if __name__=='__main__':main()
