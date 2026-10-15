"""Summarize exact entry explanations and funded admission mechanisms.

Uses the frozen, previously audited sizing policies. Future profits are used
only to score completed outcomes, never as features or a new selection rule.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('reports/allocation_interpretability_099')
AUDIT = Path('reports/retained_entry_explanations')
PAIR = Path('artifacts/contracts/p25_validation_split/replays')
STRADDLE = Path('artifacts/contracts/straddle_policy_refinement')
SOURCES = {
    'put_bandit': PAIR/'screened_selected',
    'put_fixed50': Path('data/submission/artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed50'),
    'put_fixed75': PAIR/'screened_fixed75',
    'put_fixed100': Path('data/submission/artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed100'),
    'straddle_bandit': STRADDLE/'cases/performance60/replays/tree18',
    'straddle_fixed50': Path('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18'),
    'straddle_fixed75': Path('artifacts/contracts/straddle_capped_books/fixed75/replays/tree18'),
    'straddle_fixed100': Path('artifacts/contracts/atm_rv_straddle_full/replays/straddle100'),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads((AUDIT/'protocol.json').read_text())
    for path, signature in audit['sources'].items():
        assert digest(path)==signature, 'Stale entry explanation: '+path
    stats, overlap, admission, daily = [], [], [], []
    for strategy in ('put','straddle'):
        baseline=None
        for suffix in ('bandit','fixed50','fixed75','fixed100'):
            case=strategy+'_'+suffix; folder=SOURCES[case]
            decisions=pd.read_csv(folder/'decisions.csv',usecols=['signal_date','pair','wing','level','status','risk_limits'],parse_dates=['signal_date'])
            decisions['period']=np.where(decisions.signal_date.dt.year<=2021,'validation','evaluation')
            decisions['size']=decisions.level/4
            if baseline is None:baseline=decisions.copy()
            else:
                joined=baseline.merge(decisions,on=['signal_date','pair','wing'],suffixes=('_bandit','_fixed'),validate='one_to_one')
                assert len(joined)==len(baseline)==len(decisions)
                for period,g in joined.groupby('period_bandit'):
                    a=g.status_bandit.eq('entered');b=g.status_fixed.eq('entered')
                    overlap.append(dict(strategy=strategy,control=suffix,period=period,
                                        common_entries=int((a&b).sum()),bandit_only=int((a&~b).sum()),
                                        fixed_only=int((~a&b).sum()),neither=int((~a&~b).sum())))
            for (period,size,status),g in decisions.groupby(['period','size','status']):
                admission.append(dict(strategy=strategy,policy=suffix,period=period,size=size,status=status,proposals=len(g)))
            book=pd.read_parquet(folder/'daily_nav.parquet',columns=['nav','excess_return','cash_benchmark'],use_threads=False)
            for period,start,end in [('validation','2018-01-01','2021-12-31'),('evaluation','2022-01-01','2025-08-29')]:
                x=book.loc[start:end,'excess_return']
                if x.index[0]==book.index[0]:x=x.iloc[1:]
                stats.append(dict(strategy=strategy,policy=suffix,period=period,
                                  annualized_excess_sharpe=float(np.sqrt(252)*x.mean()/x.std(ddof=1)),
                                  annualized_mean_excess_pct=float(252*x.mean()*100),
                                  annualized_excess_volatility_pct=float(np.sqrt(252)*x.std(ddof=1)*100),
                                  entries=int(((decisions.period==period)&decisions.status.eq('entered')).sum())))
            for date,row in book.iterrows():
                daily.append(dict(strategy=strategy,policy=suffix,date=date,excess_return=row.excess_return))
        print('Checked shared proposals and admissions: '+strategy,flush=True)
    pd.DataFrame(stats).to_csv(OUT/'bandit_risk_return.csv',index=False)
    pd.DataFrame(overlap).to_csv(OUT/'bandit_admission_overlap.csv',index=False)
    pd.DataFrame(admission).to_csv(OUT/'bandit_admission_by_size.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/'bandit_daily_excess.csv',index=False)
    # A quote-available diagnostic subset, not the full proposal universe.
    # Labels include extra observations admitted by the causal fixed-half book.
    meta=pd.read_parquet(STRADDLE/'training.parquet',columns=['date','secid','end','full_profit','finite_label'],use_threads=False)
    meta=meta.loc[meta.finite_label].copy()
    scores=pd.read_csv(AUDIT/'straddle_score_explanations.csv',usecols=['date','secid','period','size','status','fitted_context'],parse_dates=['date'])
    sample=meta.merge(scores,on=['date','secid'],validate='one_to_one')
    sample=sample.loc[(sample.period.eq('validation')&sample.end.le('2021-12-31'))|
                      (sample.period.eq('evaluation')&sample.end.le('2025-08-29'))]
    outcomes=[]
    for (period,size),g in sample.groupby(['period','size']):
        outcomes.append(dict(period=period,size=size,completed_observations=len(g),
                             mean_fullsize_profit=float(g.full_profit.mean()),
                             median_fullsize_profit=float(g.full_profit.median()),
                             profitable_fraction=float((g.full_profit>0).mean()),
                             mean_profit_per_reference_vega=float(g.full_profit.mean()/450)))
    pd.DataFrame(outcomes).to_csv(OUT/'straddle_completed_outcome_by_size.csv',index=False)
    pair=pd.read_csv(AUDIT/'put_pair_path_importance.csv')
    pair['path_share']=pair.path_visits/pair.groupby('period').path_visits.transform('sum')
    pair.to_csv(OUT/'put_tree_usage.csv',index=False)
    inputs=[Path(__file__),AUDIT/'protocol.json',AUDIT/'put_pair_rules.csv',AUDIT/'straddle_score_explanations.csv',
            STRADDLE/'training.parquet',Path('reports/straddle_policy_refinement/comparison.csv')]
    for root in SOURCES.values():inputs.extend([root/'decisions.csv',root/'daily_nav.parquet'])
    (OUT/'bandit_protocol.json').write_text(json.dumps(dict(
        exact_explanations_sources_reverified=True,put_proposals=608,straddle_proposals=35774,
        sources={str(p):digest(p) for p in inputs},
        meaning='Entry-size choices and funded admission are separate. Label-based outcome buckets use a quote-available retrospective subset, not all proposals, and do not establish causal size effects. Existing within-date straddle shuffles preserve daily proposed action counts but can change funded admission.'),indent=2)+'\n')


if __name__=='__main__':main()
