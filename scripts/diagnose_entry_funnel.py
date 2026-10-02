"""Count observed eligibility gates; do not infer raw proposals from final actions."""
import json
from pathlib import Path
import pandas as pd
from scripts.test_contract_entry_gates import gate_actions
from scripts.diagnose_current_contract_losses import read_csv


def main():
    root=Path('artifacts/contracts/expiry_vix_momentum/vix')
    out=Path('reports/current_pair_selection')
    rows=[];fits=[];rejections=[]
    for pair in json.loads(Path('reports/expiry_vix_momentum/matched_pairs.json').read_text()):
        folder=root/pair
        panel=pd.read_parquet(folder/'states.parquet')
        actions=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0])
        actions.index=pd.DatetimeIndex(actions.index)
        state=panel.reindex(actions.index)
        f=read_csv(folder/'fits.csv')
        for model,variant in [('rl_sized','memory_both'),('nn_fixed','memory_year')]:
            eligible=state.eligible.astype(bool)
            memory=gate_actions(pd.Series(1.,index=state.index),state,variant).ne(0)|gate_actions(pd.Series(-1.,index=state.index),state,variant).ne(0)
            own=f[f.model.isin([model,'both'])] if len(f) else f
            enabled=set(pd.to_datetime(own.loc[own.status.eq('enabled'),'fit']).dt.to_period('M')) if len(own) else set()
            enabled_mask=state.index.to_period('M').isin(enabled)
            dec=read_csv(folder/model/'decisions.csv')
            stages={'covered_signal_dates':pd.Series(True,index=state.index),'beta_correlation':eligible,
                'plus_memory_either_direction':eligible&memory,'plus_validation_enabled':eligible&memory&enabled_mask,
                'nonzero_final_action':actions[model].ne(0)}
            assert not (stages['nonzero_final_action']&~stages['plus_validation_enabled']).any()
            for stage,mask in stages.items():rows.append(dict(pair=pair,model=model,stage=stage,dates=int(mask.sum())))
            for status,count in own.status.value_counts().items() if len(own) else []:
                fits.append(dict(pair=pair,model=model,status=status,months=count))
            if len(dec):
                for status,count in dec[dec.action.ne(0)].status.value_counts().items():
                    rejections.append(dict(pair=pair,model=model,status=status,count=count))
    result=pd.DataFrame(rows);result.to_csv(out/'entry_funnel.csv',index=False)
    f=pd.DataFrame(fits);f.to_csv(out/'fit_gates.csv',index=False)
    r=pd.DataFrame(rejections);r.to_csv(out/'execution_gates.csv',index=False)
    print(result.groupby(['model','stage']).agg(dates=('dates','sum'),pairs=('dates',lambda x:(x>0).sum())).to_string())
    print(f.groupby(['model','status']).months.sum().to_string())
    print(r.groupby(['model','status'])['count'].sum().to_string())


if __name__=='__main__':main()
