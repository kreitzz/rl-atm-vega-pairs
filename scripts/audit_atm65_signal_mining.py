"""Source, timing, neutrality, fee, exit and matched-roster checks."""
import json
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.run_atm65_signal_mining import (OUT,REPORT,MODELS,FAMILIES,EXITS,CODES,PERIODS,
    NN,CURVE,source_folder,exit_replay)
from scripts.run_atm65_ablation import read_actions,features
from scripts.atm65_mining_features import DYNAMICS,GAMMA,names
from scripts.run_contract_walkforward import digest
from scripts.retrain_atm_commission import check_fits
from scripts.curve_trading.policy import ATM,Replay,gate
from scripts.curve_trading.config import COMBOS,config
from scripts.select_atm_contracts import load_calendar
from scripts.run_broad_contract_replay import ROOT,Market
from scripts.replay_frozen_atm_commission import load_pair,read_trades,cash_book
from scripts.run_selected_pair_backtest import audit

def main():
    assert json.loads((OUT/'task_status.json').read_text())['status']=='complete'
    inputs=json.loads((OUT/'inputs.json').read_text())
    for path,expected in inputs['sources'].items():assert digest(path)==expected,path
    coverage=json.loads((REPORT/'coverage.json').read_text())
    comparison=pd.read_csv(REPORT/'comparison.csv');pairs=pd.read_csv(REPORT/'pairs.csv')
    assert comparison.pairs.eq(coverage['common_complete_pairs']).all()
    rosters=[set(g.pair) for _,g in pairs.groupby(['model','period'])]
    assert all(r==rosters[0] for r in rosters)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    market=Market(cal);cfg=replace(config(),option_commission=.03)
    windows=unresolved=exit_cohorts=0;ledger_error=baseline_error=0.;reconciled=0
    for d in inputs['definitions']:
        folder=OUT/d['pair'];source=NN/d['pair']/'fee_065'
        if (folder/'states.parquet').exists():
            base=pd.read_parquet(source/'states.parquet');panel=pd.read_parquet(folder/'states.parquet')
            pd.testing.assert_frame_equal(panel[base.columns],base)
            assert len(features(base))==47
            assert np.isfinite(panel[names(DYNAMICS+GAMMA)]).all().all()
            with np.load(source/'gates.npz') as z:
                gates=np.zeros((len(COMBOS),len(panel),2),bool);gates[ATM]=z['gates']
                available=np.zeros((len(COMBOS),len(panel)),bool);available[ATM]=z['available']
            for model in FAMILIES:
                a=read_actions(folder/model/'actions.csv');assert a.isin(CODES).all()
                pd.testing.assert_series_equal(a,gate(a,a.index,panel,gates,available),check_names=False)
            saved=read_actions(CURVE/d['pair']/'atm_control/actions.csv')
            for model in EXITS:pd.testing.assert_series_equal(read_actions(folder/model/'actions.csv'),saved,check_names=False)
            fits=pd.read_csv(folder/'fits.csv');check_fits(fits)
            ranks=pd.read_csv(folder/'early_rank.csv');valid=ranks.last_outcome.notna()
            assert (pd.to_datetime(ranks.loc[valid,'last_outcome'])<'2018-01-01').all()
            if saved.ne(0).any():
                q,s,selected,divs=load_pair(market,d['secids']);replay=Replay(q,s,selected,d['secids'],divs,cfg)
                nav,_,_=exit_replay(replay,saved,cal[cal>='2018-01-01'],'none')
                oldstatus=json.loads((source_folder(d['pair'],'rl_sized','full')/'status.json').read_text())
                old=cash_book(nav.index,market.rates,cfg.capital) if oldstatus.get('cash_only') else pd.read_parquet(oldstatus.get('nav_source',source_folder(d['pair'],'rl_sized','full')/'daily_nav.parquet'),columns=['nav'])
                np.testing.assert_allclose(nav.nav,old.nav,atol=1e-7,rtol=0)
                baseline_error=max(baseline_error,float(abs(nav.nav-old.nav).max()));reconciled+=1
        stocks=market.stocks[market.stocks.secid.isin(d['secids'])]
        for model in MODELS:
            for period,_ in PERIODS:
                dest=folder/model/period;status=json.loads((dest/'status.json').read_text())
                assert status['training_hurdle']==.65 and status['execution_commission']==.03
                if status['status']=='incomplete':
                    assert status['error'] and 'entries' not in status and not status.get('cash_only');unresolved+=1;continue
                assert status['status']=='complete';windows+=1
                if status.get('cash_only'):assert status['entries']==0;continue
                nav=pd.read_parquet(dest/'daily_nav.parquet');t=read_trades(dest/'trades.csv')
                checks=audit(nav,t,stocks,cfg,d['secids']);ledger_error=max(ledger_error,checks['ledger_error'])
                if len(t):
                    paid=t.reason.isin(['entry','signal_reversal_exit','signal_flat_exit'])
                    np.testing.assert_allclose(t.loc[paid,'commission'],abs(t.loc[paid,'quantity'])*.03,atol=1e-10)
                    assert t.loc[~paid,'commission'].eq(0).all()
                dec=pd.read_csv(dest/'decisions.csv');assert int(dec.status.eq('entered').sum())==status['entries']
                entered=dec[dec.status.eq('entered')]
                if len(entered):assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a','vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
                exited=t[t.reason.isin(['signal_reversal_exit','signal_flat_exit'])] if len(t) else t
                if model not in EXITS:assert exited.empty
                if len(exited):
                    a=read_actions(folder/model/'actions.csv')
                    from scripts.curve_trading.config import ACTIONS
                    for (date,cohort),group in exited.groupby(['date','cohort']):
                        signal=pd.Timestamp(group.exit_signal_date.iloc[0]);assert cal[cal.get_loc(date)-1]==signal
                        assert signal in a.index
                        recommendation=ACTIONS[int(a.loc[signal])][1] if a.loc[signal] else 0.
                        prior=t[t.cohort.eq(cohort)&t.date.lt(date)].groupby('secid').quantity.sum()
                        # Natural expiry may already have removed one leg on this session.
                        natural=t[t.cohort.eq(cohort)&t.date.eq(date)&t.reason.eq('expiry_settlement')].groupby('secid').quantity.sum()
                        live=prior.add(natural,fill_value=0);live=live[live.ne(0)]
                        assert set(group.secid)==set(live.index)
                        assert len(t[t.cohort.eq(cohort)&t.date.gt(date)&~t.reason.eq('expiry_settlement')])==0
                        direction=np.sign(prior.get(d['secids'][0],0) or -prior.get(d['secids'][1],0))
                        if group.reason.iloc[0]=='signal_reversal_exit':assert recommendation*direction<0
                        else:assert recommendation==0 and model=='exit_flat_or_reverse'
                    exit_cohorts+=len(exited[['date','cohort']].drop_duplicates())
    if coverage['common_complete_pairs']==345:
        old=pd.read_csv('reports/atm65_ablation/comparison.csv')
        for period,_ in PERIODS:
            a=comparison[comparison.model.eq('rl_sized')&comparison.period.eq(period)].iloc[0]
            b=old[old.model.eq('rl_sized')&old.period.eq(period)].iloc[0]
            np.testing.assert_allclose(a.annualized_excess_sharpe,b.annualized_excess_sharpe,atol=1e-12)
            assert a.entries==b.entries
    result=dict(status='complete',complete_windows=windows,unresolved_windows=unresolved,
      common_complete_pairs=coverage['common_complete_pairs'],largest_ledger_error=ledger_error,
      disabled_exit_baselines_reconciled=reconciled,largest_baseline_error=baseline_error,
      policy_exit_cohorts_verified=exit_cohorts,original_inputs_unchanged=True,early_ranks_end_before_2018=True)
    (REPORT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
