"""Inventory conservation, matched-roster, training timing and exit cost audit."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scripts.run_curve_management import OUT,REPORT,MODELS,PERIODS,NN
from scripts.curve_management.config import COMBOS,ACTIONS,UNIVERSES,RUNOFF,config
from scripts.curve_management.learning import gate
from scripts.run_atm65_ablation import read_actions,features
from scripts.run_contract_walkforward import digest
from scripts.retrain_atm_commission import check_fits
from scripts.run_broad_contract_replay import ROOT,Market
from scripts.select_atm_contracts import load_calendar
from scripts.replay_frozen_atm_commission import read_trades
from scripts.run_selected_pair_backtest import audit

def main():
    assert json.loads((OUT/'task_status.json').read_text())['status']=='complete'
    inputs=json.loads((OUT/'inputs.json').read_text())
    for path,expected in inputs['sources'].items():assert digest(Path(path))==expected,path
    coverage=json.loads((REPORT/'coverage.json').read_text());comparison=pd.read_csv(REPORT/'comparison.csv')
    assert comparison.pairs.eq(coverage['common_complete_pairs']).all()
    pairs=pd.read_csv(REPORT/'pairs.csv');rosters=[set(g.pair) for _,g in pairs.groupby(['model','period'])]
    assert all(r==rosters[0] for r in rosters)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')];market=Market(cal);cfg=config(.03)
    windows=unresolved=exits=0;ledger_error=0.;overlapping_books=0
    for d in inputs['definitions']:
        folder=OUT/d['pair'];panel=None;gates=available=None
        if (folder/'states.parquet').exists():
            panel=pd.read_parquet(folder/'states.parquet');base=pd.read_parquet(NN/d['pair']/'fee_065/states.parquet')
            pd.testing.assert_frame_equal(panel[base.columns],base);assert len(features(base))==47
            inventory=json.loads((folder/'inventory_inputs.json').read_text())
            for path,snapshot in inventory['source_files'].items():
                p=Path(path);assert p.stat().st_size==snapshot['bytes'] and p.stat().st_mtime_ns==snapshot['mtime_ns'],path
            with np.load(folder/'gates.npz') as z:gates=z['gates'];available=z['available']
        stocks=market.stocks[market.stocks.secid.isin(d['secids'])]
        for model in MODELS:
            dest=folder/model;universe,management=model.rsplit('_',1)
            actions=read_actions(dest/'actions.csv') if (dest/'actions.csv').exists() else None
            if actions is not None:
                assert actions.isin(range(len(ACTIONS))).all()
                assert actions.empty or actions.index.max()<=cal[-1]-pd.Timedelta(days=RUNOFF)
                assert all(ACTIONS[int(c)][0] in UNIVERSES[universe] for c in actions[actions.ne(0)])
                pd.testing.assert_series_equal(actions,gate(actions,actions.index,panel,gates,available),check_names=False)
                fits=pd.read_csv(dest/'fits.csv');check_fits(fits)
                for field,cutoff in [('proposal_exit_last_outcome','validation_start'),('recent_exit_last_outcome','fit')]:
                    if field in fits:
                        good=fits[field].notna();assert (pd.to_datetime(fits.loc[good,field])<pd.to_datetime(fits.loc[good,cutoff])).all()
                ranks=pd.read_csv(dest/'shortlists.csv');ranks=ranks[ranks.year.notna()]
                if len(ranks):
                    assert (pd.to_datetime(ranks.ranking_cutoff)<pd.to_datetime(ranks.year.astype(int).astype(str)+'-01-01')).all()
                    assert ranks.combo.isin(UNIVERSES[universe]).all()
                    assert ranks[ranks.selected].groupby('year').combo.nunique().le(4).all()
                if (dest/'reused_policy.json').exists():
                    peer=folder/json.loads((dest/'reused_policy.json').read_text())['source']
                    assert digest(dest/'actions.csv')==digest(peer/'actions.csv')
                    assert digest(dest/'fits.csv')==digest(peer/'fits.csv')
                    other=pd.read_csv(peer/'shortlists.csv');other=other[other.year.notna()]
                    left=ranks[ranks.selected].groupby('year').combo.apply(lambda x:tuple(sorted(set(x))))
                    right=other[other.selected].groupby('year').combo.apply(lambda x:tuple(sorted(set(x))))
                    assert left.equals(right)
                    for file in (dest/'exit_models').glob('*.joblib'):assert digest(file)==digest(peer/'exit_models'/file.name)
                fit_dates=set(fits.fit.dropna())
                for path in (dest/'exit_models').glob('*.joblib'):
                    assert path.stem in fit_dates
                    model_data=joblib.load(path)
                    if model_data is not None:
                        tree,leaves,close=model_data;assert len(leaves)<=8 and close.dtype==bool
            for period,_ in PERIODS:
                target=dest/period;status=json.loads((target/'status.json').read_text())
                assert status['training_hurdle']==.65 and status['execution_commission']==.03
                if status['status']=='incomplete':
                    assert status['error'] and 'entries' not in status and not status.get('cash_only');unresolved+=1;continue
                assert status['status']=='complete';windows+=1
                if status.get('cash_only'):assert status['entries']==0;continue
                nav=pd.read_parquet(target/'daily_nav.parquet');trades=read_trades(target/'trades.csv')
                checks=audit(nav,trades,stocks,cfg,d['secids']);ledger_error=max(ledger_error,checks['ledger_error'])
                overlapping_books+=int(nav.active_cohorts.gt(1).any())
                decisions=pd.read_csv(target/'decisions.csv');entered=decisions[decisions.status.eq('entered')]
                assert len(entered)==status['entries']
                if len(entered):assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a','vega_b']].max(axis=1)).le(.1+1e-12).all()
                if trades.empty:continue
                assert set(trades.reason).issubset({'entry','split_adjustment','expiry_settlement','learned_policy_exit'})
                paid=trades.reason.isin(['entry','learned_policy_exit'])
                np.testing.assert_allclose(trades.loc[paid,'commission'],abs(trades.loc[paid,'quantity'])*.03,atol=1e-10)
                assert trades.loc[~paid,'commission'].eq(0).all()
                by_cohort={str(pd.Timestamp(r.signal_date).date()):r for r in entered.itertuples()}
                for cohort,group in trades.groupby('cohort'):
                    r=by_cohort[str(cohort.date())]
                    assert set(group.optionid).issubset({int(r.optionid_a),int(r.optionid_b)})
                    assert len(group[group.reason.eq('entry')])==2
                    code=int(actions.loc[cohort]);combo,size=ACTIONS[code]
                    assert COMBOS[combo][1]=='C'
                    first=group[group.reason.eq('entry')].set_index('secid')
                    assert int(first.loc[d['secids'][0],'optionid'])==int(r.optionid_a)
                    assert int(first.loc[d['secids'][1],'optionid'])==int(r.optionid_b)
                    assert np.sign(first.loc[d['secids'][0],'quantity'])==np.sign(size)
                    assert np.sign(first.loc[d['secids'][1],'quantity'])==-np.sign(size)
                closed=trades[trades.reason.eq('learned_policy_exit')]
                if management=='hold':assert closed.empty
                if len(closed):
                    intents=json.loads((target/'exit_intents.json').read_text())
                    for (date,cohort),group in closed.groupby(['date','cohort']):
                        signal=pd.Timestamp(group.exit_signal_date.iloc[0]);assert cal[cal.get_loc(date)-1]==signal
                        assert str(cohort.date()) in intents[str(signal.date())]
                        fit=signal.to_period('Q').start_time
                        assert (dest/'exit_models'/f'{fit.date()}.joblib').exists()
                        prior=trades[trades.cohort.eq(cohort)&trades.date.lt(date)].groupby('secid').quantity.sum()
                        onday=trades[trades.cohort.eq(cohort)&trades.date.eq(date)&trades.reason.isin(['split_adjustment','expiry_settlement'])].groupby('secid').quantity.sum()
                        live=prior.add(onday,fill_value=0);live=live[live.ne(0)]
                        assert set(group.secid)==set(live.index)
                        assert trades[trades.cohort.eq(cohort)&trades.date.gt(date)].empty
                        exits+=1
    result=dict(status='complete',complete_windows=windows,unresolved_windows=unresolved,common_complete_pairs=coverage['common_complete_pairs'],
        largest_ledger_error=ledger_error,learned_exit_cohorts_verified=exits,overlapping_inventory_books=overlapping_books,
        original_states_unchanged=True,training_outcomes_purged=True,original_contract_inventory_conserved=True)
    write=REPORT/'audit.json';write.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
