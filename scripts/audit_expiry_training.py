"""Check completed expiry-training artifacts without loading raw chains."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_contract_walkforward import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/contracts/expiry_trained'))
    parser.add_argument('--report',type=Path,default=Path('reports/expiry_trained'))
    args=parser.parse_args();root=args.root;report=args.report
    inputs=json.loads((root/'inputs.json').read_text())
    for path,expected in inputs['code'].items():assert digest(path)==expected,path
    cfg=inputs['settings'];assert cfg['hold_to_expiry']
    records=[]
    for d in inputs['definitions']:
        folder=root/d['pair'];panel=pd.read_parquet(folder/'states.parquet')
        with np.load(folder/'labels.npz',allow_pickle=False) as z:
            daily=z['daily'];ends=pd.DatetimeIndex(z['ends']);vegas=z['vegas'];statuses=z['statuses']
        assert daily.shape==(len(panel),9,cfg['max_dte']+2)
        np.testing.assert_allclose(daily[np.isfinite(daily).all(axis=(1,2)),4],0,atol=1e-8)
        reward=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg['entry_vega_per_leg'],vegas[:,[0,-1]],
            out=np.full((len(panel),2),np.nan),where=vegas[:,[0,-1]]>0)
        # Sample dates deterministically, including the end of the state panel.
        for i in np.unique(np.linspace(0,len(panel)-1,min(8,len(panel)),dtype=int)):
            date=panel.index[i]
            for days in [91,365]:
                for k,side in enumerate(['short','long']):
                    use=(ends<date)&(ends>=date-pd.Timedelta(days=days))&(vegas[:,[0,-1]][:,k]>0)&np.isfinite(reward[:,k])
                    assert panel.iloc[i][f'count_{side}_{days}d']==use.sum()
                    np.testing.assert_allclose(panel.iloc[i][f'profit_{side}_{days}d'],reward[use,k].mean() if use.any() else 0,atol=1e-12)
        actions=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0])
        if 'beta_screen' in inputs:
            rule=inputs['beta_screen']
            expected=np.isfinite(panel.beta) & panel.correlation.ge(rule['correlation_floor'])
            if rule.get('enabled',True):
                expected &= panel.beta.between(rule['minimum'],rule['maximum'])
            np.testing.assert_array_equal(panel.eligible,expected)
            for model in ['rl_sized','nn_fixed']:
                assert panel.loc[actions.index[actions[model].ne(0)],'eligible'].all()
        assert np.isin(actions.rl_sized,np.arange(-1,1.01,.25)).all()
        assert np.isin(actions.nn_fixed,[-1,0,1]).all()
        for r in json.loads((folder/'status.json').read_text()):
            if r['status']!='complete':continue
            dest=folder/r['model'];nav=pd.read_parquet(dest/'daily_nav.parquet')
            try:t=pd.read_csv(dest/'trades.csv')
            except pd.errors.EmptyDataError:t=pd.DataFrame()
            if len(t):
                assert set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
                exits=t[t.reason.eq('expiry_settlement')]
                assert exits.commission.eq(0).all() and exits.execution_penalty.eq(0).all()
                entry=t[t.reason.eq('entry')]
                np.testing.assert_allclose(entry.commission,entry.quantity.abs()*cfg['option_commission'],atol=1e-9)
                np.testing.assert_allclose(t.execution_penalty.sum(),nav.option_model_penalty.sum(),atol=1e-7)
                assert t.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
        records.append(dict(pair=d['pair'],label=d['label'],label_rows=len(panel),
            complete_labels=int((statuses=='complete').sum()),unresolved_labels=int(np.char.startswith(statuses,'unresolved').sum()),
            audit='passed'))
    pd.DataFrame(records).to_csv(report/'audit.csv',index=False)
    f=pd.read_csv(report/'fits.csv');nn=f[f.model.eq('nn_fixed')]
    summary=dict(pairs_checked=len(records),label_rows=sum(r['label_rows'] for r in records),
        unresolved_labels=sum(r['unresolved_labels'] for r in records),
        nn_validation_fits=int(nn.converged.notna().sum()),nn_validation_warnings=int(nn.converged.eq(False).sum()),
        nn_recent_fits=int(nn.recent_converged.notna().sum()),nn_recent_warnings=int(nn.recent_converged.eq(False).sum()))
    (report/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)


if __name__=='__main__':main()
