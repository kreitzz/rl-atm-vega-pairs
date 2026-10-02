"""Report every 90D sleeve, with common-roster and full-roster completeness."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_curve90 import OUTPUT,REPORT,DATA,definitions,config
from scripts.curve90.config import ACTIONS,COMBOS,RUNOFF
from scripts.curve90.policy import gate
from scripts.run_contract_walkforward import digest
from scripts.report_option_flex_ablation import sharpe

def main():
    signature=json.loads((OUTPUT/'inputs.json').read_text())
    for p,h in signature['code'].items():assert digest(p)==h,p
    checks=json.loads((OUTPUT/'reconciliation.json').read_text());assert len(checks)==6
    defs=definitions();rows=[];failed=set();mix=[]
    for d in defs:
        for variant in ['atm_control','curve']:
            folder=OUTPUT/d['pair']/variant;r=json.loads((folder/'status.json').read_text());rows.append(r)
            if r['status']!='complete':failed.add(d['pair']);continue
            nav=pd.read_parquet(r.get('nav_source',folder/'daily_nav.parquet'))
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
            if r.get('structural_cash'):continue
            panel=pd.read_parquet(folder.parent/'states.parquet');a=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0]).code.astype(int)
            with np.load(folder.parent/'gates.npz') as z:pd.testing.assert_series_equal(a,gate(a,a.index,panel,z['gates'],z['available']),check_names=False)
            assert a.empty or a.index.max()<=nav.index[-1]-pd.Timedelta(days=RUNOFF)
            try:fits=pd.read_csv(folder/'fits.csv')
            except pd.errors.EmptyDataError:fits=pd.DataFrame()
            for k,b in [('last_training_outcome','validation_start'),('last_recent_outcome','fit')]:
                if k in fits:
                    use=fits[k].notna();assert (pd.to_datetime(fits.loc[use,k])<pd.to_datetime(fits.loc[use,b])).all()
            try:dec=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
            except pd.errors.EmptyDataError:continue
            for t in dec[dec.status.eq('entered')].itertuples():
                c,size=ACTIONS[int(a.loc[t.signal_date])];tenor,cp,ba,bb=COMBOS[c];assert tenor==90
                assert abs(t.vega_a-t.vega_b)/max(t.vega_a,t.vega_b)<=config().max_entry_vega_imbalance+1e-12
                for sid,b,oid in zip(d['secids'],[ba,bb],[t.optionid_a,t.optionid_b]):
                    selected=pd.read_parquet(DATA/'selection'/cp/str(t.signal_date.year)/f'{sid}.parquet')
                    selected=selected[selected.date.eq(t.signal_date)&selected.bucket.eq(str(b))]
                    assert len(selected)==1 and selected.optionid.iloc[0]==oid and 75<=(selected.expiry.iloc[0]-t.entry_date).days<=105
                mix.append(dict(pair=d['pair'],variant=variant,cp=cp,bucket_a=ba,bucket_b=bb,size=size,date=t.entry_date))
    REPORT.mkdir(parents=True,exist_ok=True);status=pd.DataFrame(rows);status.to_csv(REPORT/'pair_status.csv',index=False)
    status[status.status.ne('complete')].to_csv(REPORT/'incomplete.csv',index=False);pd.DataFrame(mix).to_csv(REPORT/'contract_mix.csv',index=False)
    summary=[];annual=[];pairs=[]
    for variant in ['atm_control','curve']:
        own=status[status.variant.eq(variant)].set_index('pair')
        for scope,roster in [('common',[d for d in defs if d['pair'] not in failed]),('full',defs)]:
            if scope=='full' and own.status.ne('complete').any():summary.append(dict(variant=variant,scope=scope,status='incomplete'));continue
            book=None;entries=0;active=0
            for d in roster:
                row=own.loc[d['pair']];p=row.nav_source if isinstance(row.get('nav_source'),str) else OUTPUT/d['pair']/variant/'daily_nav.parquet'
                nav=pd.read_parquet(p)[['nav','cash_benchmark']]
                if book is None:book=nav.copy()
                else:assert book.index.equals(nav.index);book+=nav
                entries+=int(row.entries);active+=int(row.entries>0)
                if scope=='common':pairs.append(dict(pair=d['pair'],label=d['label'],variant=variant,entries=row.entries,net_excess_pnl=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]))
            r=book.nav.pct_change().fillna(0);cash=book.cash_benchmark.pct_change().fillna(0);e=r-cash
            book.assign(excess_return=e).to_csv(REPORT/f'{variant}_{scope}_portfolio.csv')
            summary.append(dict(variant=variant,scope=scope,status='complete',pairs=len(roster),entries=entries,active_pairs=active,annualized_excess_sharpe=sharpe(e.iloc[1:]),net_excess_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]))
            if scope=='common':
                for year,g in e.groupby(e.index.year):annual.append(dict(variant=variant,year=year,annualized_excess_sharpe=sharpe(g),return_percent=100*((1+r.loc[g.index]).prod()-1),excess_return_percent=100*((1+r.loc[g.index]).prod()-(1+cash.loc[g.index]).prod())))
    result=pd.DataFrame(summary);result.to_csv(REPORT/'comparison.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'pair_results.csv',index=False)
    lines=['# 90-day-only contract test','',result.to_string(index=False),'',
        'Target 90 days, entry window 75–105 days; one maturity target, independent delta buckets, calls/calls or puts/puts. Initial integer vega matching, daily delta hedge, hold each leg to its own expiry. Same costs/risk limits and original 30D reference state. The new ATM control also trades 90D calls. Seventy-two curve combinations are screened annually using past training data; monthly sizing policy and chosen-contract profitability gates remain unchanged.', '',
        'A 110-calendar-day runoff applies to both validation and final entry eligibility. Older 30D/14–45D results used 60 days, so those are contextual comparisons, not a matched maturity ablation. Evaluation history is already explored research data, not an untouched holdout. Early assignment and daily option-vega rebalancing are not modeled.', '',
        '[Annual results](curve90/annual.csv) · [Pair results](curve90/pair_results.csv) · [Contract choices](curve90/contract_mix.csv) · [Incomplete sleeves](curve90/incomplete.csv)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(result.to_string(index=False))
    (OUTPUT/'task_status.json').write_text(json.dumps(dict(status='complete',pairs_attempted=len(defs),common_complete=len(defs)-len(failed),failed_pairs=sorted(failed)),indent=2)+'\n')

if __name__=='__main__':main()
