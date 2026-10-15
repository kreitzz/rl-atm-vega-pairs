"""Reproduce and explain both frozen entry-sizing policies, month by month.

Exact tree paths explain the put-pair context. Exact additive ridge-score terms
explain straddle context; the learned context-to-size map is shown separately.
Neither explanation attributes economic causality to a correlated input.
"""
import argparse
import csv
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('reports/retained_entry_explanations')
PAIR = Path('artifacts/contracts/p25_feature_search')
STRADDLE = Path('artifacts/contracts/straddle_policy_refinement')


def explain_pair():
    import joblib
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__)]
    path = Path('artifacts/contracts/p25_validation_split/replays/screened_selected/decisions.csv')
    sources += [path, PAIR/'signal_rows.parquet', PAIR/'replays/control/scored_allocations.parquet']
    decisions = pd.read_csv(path, parse_dates=['signal_date'])
    rows = pd.read_parquet(PAIR/'signal_rows.parquet', use_threads=False)
    expected = pd.read_parquet(PAIR/'replays/control/scored_allocations.parquet',
                               columns=['date','pair','wing','learned_level'], use_threads=False)
    expected = expected.rename(columns={'learned_level':'recorded_level'})
    rows = rows.merge(expected, on=['date','pair','wing'], validate='one_to_one')
    rows = rows.merge(decisions[['signal_date','pair','wing','level','status','risk_limits']],
                      left_on=['date','pair','wing'], right_on=['signal_date','pair','wing'], validate='one_to_one')
    assert len(rows) == 608
    pair_records, split_records = [], []
    for month, frame in rows.groupby(rows.date.dt.to_period('M')):
        model_path = PAIR/'models/control'/f'{month.start_time.date()}.joblib'
        meta_path = model_path.with_suffix('.json'); sources += [model_path, meta_path]
        model = joblib.load(model_path); meta = json.loads(meta_path.read_text())
        if meta['last_training_outcome'] is not None:
            assert pd.Timestamp(meta['last_training_outcome']) < month.start_time
        names = model['features']; tree = model['tree']
        for row in frame.itertuples(index=False):
            values = np.array([getattr(row, n) for n in names])
            conditions = []
            if tree is None:
                state, leaf = 0, 0
            else:
                leaf = int(tree.apply(pd.DataFrame(values[None],columns=names))[0]); state = int(np.searchsorted(model['leaves'], leaf))
                node = 0
                while tree.tree_.children_left[node] != tree.tree_.children_right[node]:
                    j = int(tree.tree_.feature[node]); threshold = float(tree.tree_.threshold[node])
                    # sklearn converts the predictor input to float32.
                    left = np.float32(values[j]) <= threshold
                    conditions.append(f'{names[j]}={values[j]:.6g} '+('<=' if left else '>')+f' {threshold:.6g}')
                    split_records.append(dict(date=row.date,pair=row.pair,period='validation' if row.date.year<=2021 else 'evaluation',
                        feature=names[j],raw_value=values[j],threshold=threshold,went_left=bool(left)))
                    node = int(tree.tree_.children_left[node] if left else tree.tree_.children_right[node])
                assert node == leaf
            level = int(model['choices'][state])
            assert level == row.recorded_level == row.level
            pair_records.append(dict(date=row.date,pair=row.pair,wing=row.wing,period='validation' if row.date.year<=2021 else 'evaluation',
                model_fit=month.start_time,last_completed=meta['last_training_outcome'],
                training_samples=meta['training_rows'],leaf=leaf,size=level/4,status=row.status,
                risk_limits=row.risk_limits,rule=' AND '.join(conditions) if conditions else 'Single default context'))
    pd.DataFrame(pair_records).to_csv(OUT/'put_pair_rules.csv', index=False)
    splits = pd.DataFrame(split_records); splits.to_csv(OUT/'put_pair_path_splits.csv', index=False)
    splits.groupby(['period','feature']).size().rename('path_visits').reset_index().to_csv(OUT/'put_pair_path_importance.csv', index=False)
    del rows, decisions, expected
    gc.collect()
    metadata = dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))}, proposals=len(pair_records), exact_level_replication=True)
    (OUT/"put_pair_protocol.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Reproduced608 put-pair sizes",flush=True)


def explain_straddle():
    OUT.mkdir(parents=True,exist_ok=True)
    sources = [Path(p) for p in json.loads((OUT/"put_pair_protocol.json").read_text())["sources"]]
    pair_records = pd.read_csv(OUT/"put_pair_rules.csv")
    path = STRADDLE/'cases/performance60/replays/tree18/decisions.csv'
    allocation_path = STRADDLE/'cases/performance60/allocations/tree18_test.parquet'
    sources += [path, allocation_path, STRADDLE/'proposals.parquet']
    decisions = pd.read_csv(path, parse_dates=['signal_date'])
    names = json.loads((STRADDLE/'models/performance60/2022-01.json').read_text())['features']
    context = pd.read_parquet(STRADDLE/'proposals.parquet', columns=['date','pair','secid']+names,use_threads=False)
    allocations = pd.read_parquet(allocation_path, columns=['date','secid','learned_level'], use_threads=False)
    context = context.merge(allocations, on=['date','secid'], validate='one_to_one')
    context = context.merge(decisions[['signal_date','pair','level','status','risk_limits']],
        left_on=['date','pair'],right_on=['signal_date','pair'],validate='one_to_one')
    assert len(context) == 35774
    importance = defaultdict(lambda: np.zeros(3))
    count = 0
    with (OUT/'straddle_score_explanations.csv').open('w', newline='') as output:
        writer = None
        for month, frame in context.groupby(context.date.dt.to_period('M')):
            path = STRADDLE/'models/performance60'/f'{month}.json'; sources.append(path)
            model = json.loads(path.read_text()); names = model['features']
            if model['last_completed'] is not None:
                assert pd.Timestamp(model['last_completed']) < month.start_time
            values = frame[names].to_numpy(float)
            active = len(model['choices']) > 1
            if active:
                design = np.column_stack([np.ones(len(frame)),np.clip((values-model['center'])/model['scale'],-model['clip'],model['clip'])])
                terms = design*np.array(model['beta']); scores = terms.sum(axis=1)
                state = np.searchsorted(model['cuts'],scores,side='left')
                level = np.asarray(model['choices'])[state]
                predicted = design@model['beta']
                np.testing.assert_allclose(scores,predicted,rtol=1e-12,atol=1e-8)
            else:
                terms = np.zeros((len(frame),len(names)+1)); scores = np.full(len(frame),np.nan)
                state = np.zeros(len(frame),int); level = np.full(len(frame),model['choices'][0])
            np.testing.assert_array_equal(level,frame.learned_level)
            np.testing.assert_array_equal(level,frame.level)
            for i, row in enumerate(frame.itertuples(index=False)):
                period = 'validation' if row.date.year<=2021 else 'evaluation'
                record = dict(date=row.date,secid=row.secid,period=period,status=row.status,
                    size=level[i]/4,score=scores[i],state=int(state[i]),
                    state_size_map=json.dumps([v/4 for v in model['choices']]),
                    cuts=json.dumps(model.get('cuts',[])),last_completed=model['last_completed'],
                    training_samples=model['training_samples'],fitted_context=active,
                    intercept=terms[i,0] if active else np.nan)
                for j, name in enumerate(names):
                    record['raw_'+name] = values[i,j]
                    record['score_term_'+name] = terms[i,j+1] if active else np.nan
                    flip = False
                    if active:
                        counterfactual = scores[i]-terms[i,j+1]
                        counterstate = int(np.searchsorted(model['cuts'],counterfactual,side='left'))
                        flip = model['choices'][counterstate] != level[i]
                        importance[(period,name)] += [abs(terms[i,j+1]),float(flip),1]
                    record['size_changes_when_centered_'+name] = flip
                if writer is None:
                    writer = csv.DictWriter(output,fieldnames=list(record)); writer.writeheader()
                writer.writerow(record); count += 1
            gc.collect()
    assert count == 35774
    pd.DataFrame([dict(period=k[0],feature=k[1],mean_abs_score_contribution=v[0]/v[2],
        center_ablation_size_flip_rate=v[1]/v[2],fitted_proposals=int(v[2])) for k,v in importance.items()]).to_csv(OUT/'straddle_feature_importance.csv',index=False)
    all_sizes = pd.concat([pd.DataFrame(pair_records)[['period','size','status']].assign(strategy='put_pair'),
                          context[['date','learned_level','status']].assign(period=np.where(context.date.dt.year<=2021,'validation','evaluation'),
                              size=context.learned_level/4,strategy='straddle')[['period','size','status','strategy']]])
    all_sizes.groupby(['strategy','period','size','status']).size().rename('proposals').reset_index().to_csv(OUT/'size_and_admission.csv',index=False)
    protocol = dict(put_proposals=608,straddle_proposals=count,exact_level_replication=True,
        put='Exact root-to-leaf conditions followed by the CEM-selected leaf-to-size map. Path frequency is model usage, not a profitability effect or impurity importance.',
        straddle='Exact standardized ridge-score contributions, followed by score thresholds and the learned size map. A larger score does not necessarily imply larger size because the action map is not constrained to be monotonic.',
        default='Months without a fitted multi-context model use the saved default size and are not assigned invented feature contributions.',
        timing='Every saved model is checked for completed training outcomes before its monthly cutoff. Row dates are signal closes; actual execution occurs in the following session.',
        admission='Model size and proposal admission are separate. All proposals, including rejected ones, are included and carry their saved admission status.',
        interpretation='No new fitting. Center ablations describe sensitivity conditional on other inputs, not causal feature effects. Correlated inputs can substitute for each other. Performance-based validation of explanations is a separate question.',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (OUT/'FINDINGS.txt').write_text('FROZEN ENTRY POLICY EXPLANATIONS\n\n'+json.dumps({k:v for k,v in protocol.items() if k!='sources'},indent=2)+'\n')
    print('Reproduced608 put-pair sizes and35774 straddle sizes; explanations saved',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["pair","straddle"],required=True)
    args=parser.parse_args()
    explain_pair() if args.phase=="pair" else explain_straddle()
