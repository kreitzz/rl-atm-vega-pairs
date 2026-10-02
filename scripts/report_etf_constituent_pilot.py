"""Matched ETF constituent comparison, including idle capital and annual results."""
import json
import numpy as np
import pandas as pd
from scripts.run_etf_constituent_pilot import OUTPUT, REPORT, definitions, eligibility
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
    lines=['# ETF constituent pilot: beta gate ablation','',
        'Seven frozen pairs: OIH/SLB, OIH/HAL, OIH/RIG, OIH/BKR, SIL/HL, SIL/AG and SIL/CDE. Membership is evidenced by the [OIH March 2018 SEC holdings](https://www.sec.gov/Archives/edgar/data/1137360/000093041318001946/c91215_nq.htm) and [SIL July 2018 SEC holdings](https://www.sec.gov/Archives/edgar/data/1432353/000114420418051506/tv502272_nq.htm). BKR is the current label for the linked Baker Hughes history. These are historical snapshot constituents, not a reconstructed daily membership universe. Selection is restricted to available cached histories and is exploratory; several pairs were previously tested.','',
        'Evaluation: January 2019–August 29, 2025, after both 2018 filings. Training can use earlier completed price and option outcomes; it does not assert earlier ETF membership. Roster is frozen after selection, with no subsequent membership removals.','',
        'Two separately retrained variants: beta 0.85–1.15 versus no beta cutoff. Both retain correlation ≥0.5, finite beta as an input, VIX, the existing features, monthly refits, profitability-memory and validation gates. Beta is ETF-on-stock (the existing leg orientation), lagged and frozen monthly. Models remain independent per pair: RL with the nine-level sizing ladder versus fixed-size NN.','',
        'Costs and risk settings are identical: actual near-ATM calls targeting 30 days, daily delta hedging, hold to expiry; mid plus 0.1 vol point per leg on entry, $0.65 per contract, 1 bp stock execution, borrow and financing. Initial capital is $1m per pair, including idle sleeves. Existing position, reserve, hedge and stress limits apply per pair. Shared ETF exposure is not netted or constrained across sleeves.','',
        f'{len(matched)}/7 pairs completed in all four model/variant combinations; only these enter every aggregate comparison. Incomplete results remain in the pair CSV.','',
        '| Beta filter | Model | Active pairs | Entries | Annualized excess Sharpe | Net excess P&L |',
        '|---|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.variant} | {row.model} | {row.active_pairs} | {row.entries} | {row.annualized_excess_sharpe:.3f} | ${row.net_excess_pnl:,.0f} |')
    lines+=['','Sharpe uses daily returns above the cash benchmark and √252, including flat periods. Annual percentage returns include interest on unused cash. The 2025 observation is partial. This is a two-sector pilot on previously inspected dates, not an untouched test. Validation evaluates the proposal before the recent-data refit, not the exact deployed weights. NN convergence warnings and unresolved shadow labels remain disclosed by each variant’s audit.','',
        '[Pair results](etf_constituent_pilot/pair_results.csv) · [Yearly returns and Sharpe](etf_constituent_pilot/annual.csv) · [Beta eligibility](etf_constituent_pilot/eligibility.csv) · [Comparison](etf_constituent_pilot/comparison.csv)','']
    lines+=['The tight RL basket gains $6.3k above cash, with $5.0k from four OIH/BKR entries. Removing the beta cutoff increases RL entries from 37 to 65 but loses $45.6k: OIH/RIG loses $30.5k and SIL/HL loses $28.9k, partly offset by other pairs. Tight-beta NN and RL have nearly identical aggregate Sharpe; neither result establishes a strong edge.','',
        'The tight beta gate excludes OIH/RIG and all SIL pairs despite median stock-return correlations of roughly 0.81–0.88, because ETF-on-stock betas are around 0.48–0.56. Removing it changes opportunity eligibility rather than vega neutrality. SIL/CDE has only 442 usable state rows across the full history and cannot satisfy the 504-row training minimum; SIL/HL and SIL/AG also have substantial training gaps. This is not a broad test of liquid index ETF constituents.','',
        'Cross-variant checks confirm identical state values apart from eligibility, identical standalone payoff labels, costs, features, risk settings and evaluation dates. The 150-test suite passes.','']
    for variant in VARIANTS:
        path=REPORT/variant/'audit.json'
        if path.exists():
            a=json.loads(path.read_text());m=json.loads((REPORT/variant/'manifest.json').read_text())
            lines.append(f'{variant}: audit passed for {a["pairs_checked"]} pairs and {a["label_rows"]:,} label rows; {a["unresolved_labels"]} unresolved shadow labels excluded from training. NN convergence warnings: {a["nn_validation_warnings"]}/{a["nn_validation_fits"]} validation fits, {a["nn_recent_warnings"]}/{a["nn_recent_fits"]} recent fits. Runtime {m["elapsed_seconds"]:.0f}s, two workers, peak worker RSS {m["max_worker_peak_rss_bytes"]/1024**3:.2f} GiB. [Audit](etf_constituent_pilot/{variant}/audit.json).\n')
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(summary.to_string(index=False));print(pairs[['variant','label','model','status','entries','net_excess_pnl']].to_string(index=False))


if __name__=='__main__':main()
