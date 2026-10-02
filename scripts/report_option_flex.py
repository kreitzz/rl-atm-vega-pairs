"""Matched full-roster call/put comparison and lifecycle/timing audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_option_flex import BASE,LABELBASE,OUTPUT,REPORT,definitions
from scripts.option_flex.policy import ACTIONS,COMBOS,FLAT,gate
from scripts.run_contract_walkforward import digest


def main():
    inputs=json.loads((OUTPUT/'inputs.json').read_text())
    for path,h in inputs['code'].items():assert digest(path)==h,path
    for path,h in inputs['reference_files'].items():assert digest(path)==h,path
    assert digest('data/call_put/manifest.json')==inputs['put_manifest']
    assert digest('data/contract_history_shared/manifest.json')==inputs['call_cache']
    defs=definitions();rows=[];mix=[];fits=[];audits=[]
    for d in defs:
        folder=OUTPUT/d['pair'];row=json.loads((folder/'status.json').read_text());rows.append(row)
        panel=pd.read_parquet(folder/'states.parquet')
        baseline=pd.read_parquet(BASE/d['pair']/'states.parquet')
        pd.testing.assert_frame_equal(panel[baseline.columns],baseline)
        coverage={}
        if (folder/'labels.npz').exists():
            with np.load(folder/'labels.npz',allow_pickle=False) as z:
                daily=z['daily'];assert daily.shape[1:]==(33,inputs['settings']['max_dte']+2)
                with np.load(LABELBASE/d['pair']/'labels.npz',allow_pickle=False) as old:
                    old_finite=np.isfinite(old['daily']).all(axis=(1,2))
                joint_finite=np.isfinite(daily).all(axis=(1,2))
                coverage=dict(call_complete_rows=int(old_finite.sum()),joint_complete_rows=int(joint_finite.sum()),
                    additional_rows_excluded=int((old_finite&~joint_finite).sum()))
                np.testing.assert_allclose(daily[np.isfinite(daily).all(axis=(1,2)),FLAT],0.,atol=1e-8)
                for c in COMBOS:
                    ends=pd.DatetimeIndex(z[c+'_ends']);vega=z[c+'_vegas'][:,[0,-1]]
                    outcome=daily[:,[ACTIONS.index((c,-1.)),ACTIONS.index((c,1.))]].sum(axis=2)
                    reward=np.divide(outcome*inputs['settings']['entry_vega_per_leg'],vega,out=np.full_like(outcome,np.nan),where=vega>0)
                    for i in np.unique(np.linspace(0,len(panel)-1,min(6,len(panel)),dtype=int)):
                        date=panel.index[i]
                        for days in [91,365]:
                            for k,side in enumerate(['short','long']):
                                use=(ends<date)&(ends>=date-pd.Timedelta(days=days))&(vega[:,k]>0)&np.isfinite(reward[:,k])
                                assert panel.iloc[i][f'{c}_count_{side}_{days}d']==use.sum()
                                np.testing.assert_allclose(panel.iloc[i][f'{c}_profit_{side}_{days}d'],reward[use,k].mean() if use.any() else 0,atol=1e-12)
        actions=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0]).action_code
        assert np.isin(actions,np.arange(len(ACTIONS))).all()
        pd.testing.assert_series_equal(actions,gate(actions,panel))
        try:f=pd.read_csv(folder/'fits.csv')
        except pd.errors.EmptyDataError:f=pd.DataFrame()
        if len(f):
            for end,start in [('last_training_outcome','validation_start'),('last_recent_outcome','fit')]:
                if end in f:
                    use=f[end].notna();assert (pd.to_datetime(f.loc[use,end])<pd.to_datetime(f.loc[use,start])).all()
            f['pair']=d['pair'];fits.append(f)
        if row['status']=='complete':
            nav=pd.read_parquet(folder/'daily_nav.parquet');row['net_excess_pnl']=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
            dec=pd.read_csv(folder/'decisions.csv');entered=dec[dec.status.eq('entered')].copy()
            if len(entered):
                assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a','vega_b']].max(axis=1)).le(inputs['settings']['max_entry_vega_imbalance']+1e-12).all()
                entered['code']=actions.reindex(pd.to_datetime(entered.signal_date)).to_numpy()
                entered['combination']=[ACTIONS[int(k)][0] for k in entered.code]
                for c,n in entered.combination.value_counts().items():mix.append(dict(pair=d['pair'],label=d['label'],combination=c,entries=n))
                t=pd.read_csv(folder/'trades.csv');assert set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
                assert t.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
                settled=t[t.reason.eq('expiry_settlement')];assert settled.commission.eq(0).all() and settled.execution_penalty.eq(0).all()
                entry=t[t.reason.eq('entry')]
                for e in entered.itertuples():
                    group=entry[entry.cohort.eq(e.signal_date)].set_index('secid')
                    actual=''.join(group.loc[sid,'cp_flag'] for sid in d['secids'])
                    assert actual==e.combination
        audits.append(dict(pair=d['pair'],calls_only_replay=row['calls_control'],checks='passed',status=row['status'],**coverage))
    status=pd.DataFrame(rows);status.to_csv(REPORT/'pair_results.csv',index=False)
    pd.DataFrame(mix,columns=['pair','label','combination','entries']).to_csv(REPORT/'option_types.csv',index=False)
    pd.concat(fits,ignore_index=True).to_csv(REPORT/'fits.csv',index=False);pd.DataFrame(audits).to_csv(REPORT/'audit.csv',index=False)
    complete=set(status.loc[status.status.eq('complete'),'pair']);matched=[d for d in defs if d['pair'] in complete]
    summary=[];annual=[];components=[];reference_pairs=[]
    for model in ['calls_only','call_put_choice']:
        books=[];entries=0;active_pairs=0
        for d in matched:
            path=BASE/d['pair']/'rl_sized/daily_nav.parquet' if model=='calls_only' else OUTPUT/d['pair']/'daily_nav.parquet'
            nav=pd.read_parquet(path);books.append(nav[['nav','cash_benchmark']])
            if model=='calls_only':
                own=next(x for x in json.loads((BASE/d['pair']/'status.json').read_text()) if x['model']=='rl_sized')
                reference_pairs.append(dict(pair=d['pair'],reference_entries=own['entries'],reference_pnl=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]))
            else:own=next(x for x in rows if x['pair']==d['pair'])
            entries+=int(own.get('entries',0));active_pairs+=int(own.get('entries',0)>0)
            c=dict(model=model,pair=d['pair'],gross_option_pnl=own.get('option_pnl',0.)+nav.option_execution_cost.sum(),
                gross_hedge_pnl=own.get('hedge_pnl',0.)+nav.hedge_cost.sum(),option_cost=-nav.option_execution_cost.sum(),
                hedge_cost=-nav.hedge_cost.sum(),stock_borrow=-nav.stock_borrow_cost.sum(),dividends=nav.dividend_cash.sum(),
                financing_above_cash=nav.interest.sum()-(nav.cash_benchmark.iloc[-1]-nav.cash_benchmark.iloc[0]))
            np.testing.assert_allclose(sum(v for k,v in c.items() if k not in ['model','pair']),nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1],rtol=0,atol=1e-5)
            components.append(c)
        assert books and all(b.index.equals(books[0].index) for b in books)
        book=sum(books);r=book.nav.pct_change().fillna(0);e=r-book.cash_benchmark.pct_change().fillna(0)
        pnl=book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
        book.assign(return_=r,excess_return=e,excess_pnl=pnl).to_csv(REPORT/f'{model}_portfolio.csv')
        summary.append(dict(model=model,pairs=len(matched),active_pairs=active_pairs,entries=entries,annualized_excess_sharpe=np.sqrt(252)*e.iloc[1:].mean()/e.iloc[1:].std(),
            net_excess_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1],total_return_percent=100*(book.nav.iloc[-1]/book.nav.iloc[0]-1)))
        for year,g in e.groupby(e.index.year):
            annual.append(dict(model=model,year=year,return_percent=100*((1+r.loc[g.index]).prod()-1),net_excess_pnl=pnl.loc[g.index].sum(),
                annualized_excess_sharpe=np.sqrt(252)*g.mean()/g.std() if g.std()>1e-12 else np.nan))
    summary=pd.DataFrame(summary);summary.to_csv(REPORT/'comparison.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(components).to_csv(REPORT/'pair_pnl_components.csv',index=False)
    pd.DataFrame(components).drop(columns='pair').groupby('model').sum().to_csv(REPORT/'pnl_components.csv')
    changes=status[['pair','label','entries','net_excess_pnl']].merge(pd.DataFrame(reference_pairs),on='pair')
    changes['change']=changes.net_excess_pnl-changes.reference_pnl
    changes.sort_values('change').to_csv(REPORT/'pair_changes.csv',index=False)
    lines=['# Call/put choice on the 0.226 Sharpe reference','',
        'The full fixed 346-pair VIX reference roster is retained. Each pair retrains independently with 33 actions: CC, CP, PC or PP at each nonzero size in the original ladder, plus flat. Puts match the original selected call’s strike and expiry. Direction determines which underlying is long vega; call/put selection does not reverse that direction.','',
        'Market-state features, beta 0.85–1.15, correlation ≥0.5, monthly retraining, trailing training windows, validation, recent refits, hold-to-expiry treatment, costs and per-pair risk limits are retained. Each contract combination has its own completed-outcome profitability-memory gate. The original call-memory features remain the tree inputs; put-specific memory is used as a gate, not added state features. Tree partitions use the CC directional payoff target; Sharpe optimization selects among all 33 actions.','',
        'Training requires completed finite trajectories for every available action; an unresolved trajectory excludes that historical training row. Missing or ineligible put entries are zero-reward no-trades, and a put action cannot deploy without a valid signal-date selection. As in the reference, training uses standalone fixed-capital payoff trajectories; validation and evaluation apply the full portfolio risk overlay and NAV-dependent sizing. The wider action set can therefore change training samples, memory eligibility, validation and fitted policies. This is a retrained policy comparison, not a fixed-trade attribution.','',
        'Vega is matched initially using integer contracts within the existing 10% mismatch tolerance; daily delta hedging continues. Vega can drift after entry, and expiry mismatch can leave one leg outstanding. The fixed 0.1-vol-point entry penalty per leg, commissions, stock execution, financing and borrow settings are unchanged. The model uses European Black–Scholes call/put fits with declared-dividend adjustment and intrinsic cash settlement; American early exercise/assignment is not modeled. Put results require that qualification.','',
        f'{len(matched)}/{len(defs)} pairs completed. Every aggregate uses the same completed roster on both sides; incomplete books remain disclosed. The original full calls-only reference Sharpe was 0.2257. For pairs with any beta/correlation-eligible evaluation date, frozen calls-only actions were replayed through the new engine and required to reproduce each original NAV path within $0.0000001. The other 218 sleeves are provably ineligible regardless of option type and retain their verified cash-only NAV; they remain in the aggregate capital base.','',
        '| Policy | Annualized excess Sharpe | P&L above cash | Entries | Active pairs |', '|---|---:|---:|---:|---:|']
    methods=lines[2:-2]
    old,new=summary.iloc[0],summary.iloc[1]
    lines=lines[:2]+[f'Allowing independent call/put choice changed annualized excess Sharpe from {old.annualized_excess_sharpe:.3f} to {new.annualized_excess_sharpe:.3f} on the same {len(matched)} pairs.','']+lines[-2:]
    for r in summary.itertuples():
        label={'calls_only':'Calls only','call_put_choice':'Call/put choice'}[r.model]
        lines.append(f'| {label} | {r.annualized_excess_sharpe:.3f} | ${r.net_excess_pnl:,.0f} | {r.entries} | {r.active_pairs} |')
    type_counts=pd.DataFrame(mix,columns=['pair','label','combination','entries']).groupby('combination').entries.sum().reindex(COMBOS,fill_value=0)
    lines+=['','Actual entries by option type (first letter is the first underlying): '+', '.join(f'{c}: {int(n)}' for c,n in type_counts.items())+'.']
    declines=changes.sort_values('change').head(3)
    lines+=['','Largest pair-level P&L changes versus calls only: '+', '.join(f'{r.label} ${r.change:,.0f}' for r in declines.itertuples())+'. These compare retrained policies, so entries and trade dates can differ.','']+methods
    lines+=['','January 2018–August 2025. Sharpe is daily cash-excess return annualized by √252, including inactive sleeves. Each sleeve starts with $1m. Percentage returns include cash interest. Shared-underlying exposures are not subject to an additional global portfolio limit. These are previously inspected research dates, not an untouched holdout. As in the reference, validation tests the pre-refit proposal rather than the exact deployed weights.','',
        '[Pair results](option_flex/pair_results.csv) · [Pair changes](option_flex/pair_changes.csv) · [Annual results](option_flex/annual.csv) · [Call/put entry mix](option_flex/option_types.csv) · [P&L components](option_flex/pnl_components.csv) · [Audit](option_flex/audit.csv) · [Matched-mark quality](option_flex/matched_mark_quality.csv)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(summary.to_string(index=False));print(status.status.value_counts())


if __name__=='__main__':main()
