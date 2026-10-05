"""Replay trusted local saved episode policies and verify original daily NAV."""
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scripts import run_contract_episodes as implementation
from scripts.contract_engine import Settings
from scripts.contract_q_engine import evaluate,LONG,SHORT,WAIT
from scripts.run_contract_q import market_slice,summarize,constant_policy


class LocalModels(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='__main__' and name in ('Forecasts','EpisodePolicy'):
            return getattr(implementation,name)
        return super().find_class(module,name)


def main():
    root=Path('artifacts/contract_episodes')
    path=Path('data/q_contracts/AMD_NVDA/prepared.pkl')
    manifest=json.loads((root/'manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['data_sha256']:
        raise ValueError('Replay data hash differs')
    with (root/'models.pkl').open('rb') as f: models=LocalModels(f).load()
    bundle=pd.read_pickle(path); cfg=Settings(**bundle['settings'])
    selected=json.loads((root/'selection.json').read_text())['selected']
    summary=[]
    ablation=[]
    for year in (2024,2025):
        m=market_slice(bundle,f'{year}-01-01',f'{year+1}-01-01',cfg)
        for family,name in selected.items():
            p=constant_policy('cash') if name=='cash' else models[name]
            nav,error=evaluate(m,p)
            prior=pd.read_csv(root/f'{year}_selected_{family}.csv')
            np.testing.assert_allclose(nav.nav,prior.nav,atol=1e-7,rtol=1e-12)
            summary.append(dict(year=year,family=family,chosen=name,**summarize(nav,error,cfg)))
        # Post-hoc diagnostic only: identical fixed-policy entry dates and a
        # previously fitted Q exit rule. Never used for model selection.
        chosen=selected['fixed']
        if chosen!='cash':
            fixed=models[chosen]
            q=models[f'q_exit_h{fixed.horizon}_buffer0']
            original=pd.read_csv(root/f'{year}_selected_fixed.csv')
            entry_dates={pd.Timestamp(r.signal_date):LONG if r.action=='enter_long' else SHORT
                         for r in original[original.status.eq('entered')].itertuples()}
            def identical_entries(m,t,book):
                return entry_dates.get(m.dates[t],WAIT) if book is None else q(m,t,book)
            nav,error=evaluate(m,identical_entries)
            nav.to_csv(root/f'{year}_posthoc_same_entries_q_exits.csv')
            ablation.append(dict(year=year,**summarize(nav,error,cfg)))
    # Additional validation-only diagnosis; never used to reselect settings.
    m=market_slice(bundle,'2023-01-01','2024-01-01',cfg)
    diagnostics={}
    for name,p in models.items():
        idx=[t for t in range(m.n-p.horizon-1) if LONG in m.legal(t,None)]
        x=np.array([p.forecasts.features(m,t,None) for t in idx])
        values=np.column_stack([model.predict(x) for model in p.entry_models.values()])
        diagnostics[name]=dict(eligible_decisions=len(idx),
            positive_entry_predictions=int((values.max(axis=1)>p.threshold).sum()),
            mean_best_entry_dollars=float(values.max(axis=1).mean()),
            largest_entry_dollars=float(values.max()))
    result=dict(replay=summary,validation_values=diagnostics,posthoc_same_entries_q_exits=ablation)
    (root/'replay_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
