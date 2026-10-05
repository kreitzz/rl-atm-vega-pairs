"""Audit the single-hurdle ablation without changing simulation outputs."""
import json
import numpy as np
import pandas as pd
from scripts.run_atm65_ablation import (OUT, REPORT, NN, CURVE, MODELS, PERIODS,
                                      SHORT, LONG, ATM, full_size, source_folder, read_actions)
from scripts.run_contract_walkforward import digest
from scripts.retrain_atm_commission import check_fits
from scripts.curve_trading.policy import gate
from scripts.curve_trading.config import COMBOS

def main():
    assert json.loads((OUT/'task_status.json').read_text())['status'] == 'complete'
    inputs = json.loads((OUT/'inputs.json').read_text())
    for path, expected in inputs['sources'].items(): assert digest(path) == expected, path
    coverage = json.loads((REPORT/'coverage.json').read_text())
    comparison = pd.read_csv(REPORT/'comparison.csv'); pairs = pd.read_csv(REPORT/'pairs.csv')
    assert comparison.pairs.eq(coverage['common_complete_pairs']).all()
    rosters = [set(group.pair) for _, group in pairs.groupby(['model','period'])]
    assert all(r == rosters[0] for r in rosters)
    ledger_error = 0.; windows = unresolved = recovered = 0
    for d in inputs['definitions']:
        for model in MODELS:
            folder = OUT/d['pair']/model
            for period, _ in PERIODS:
                status = json.loads((folder/period/'status.json').read_text())
                assert status['training_hurdle'] == .65 and status['execution_commission'] == .03
                if status['status'] == 'incomplete':
                    assert status['error'] and 'entries' not in status and not status.get('cash_only')
                    unresolved += 1; continue
                assert status['status'] == 'complete'; windows += 1
                ledger_error = max(ledger_error, status.get('ledger_error', 0.))
                if status['entries']:
                    trades = pd.read_csv(folder/period/'trades.csv'); entry = trades.reason.eq('entry')
                    np.testing.assert_allclose(trades.loc[entry,'commission'], abs(trades.loc[entry,'quantity'])*.03, atol=1e-10)
                    assert trades.loc[~entry,'commission'].eq(0).all()
        source = NN/d['pair']/'fee_065'
        if not (OUT/d['pair']/'fits.csv').exists(): continue
        state = pd.read_parquet(source/'states.parquet')
        with np.load(source/'gates.npz') as z:
            gates = np.zeros((len(COMBOS),len(state),2), bool); gates[ATM] = z['gates']
            available = np.zeros((len(COMBOS),len(state)), bool); available[ATM] = z['available']
        own = {}
        for model in MODELS:
            actions = read_actions(OUT/d['pair']/model/'actions.csv')
            assert actions.isin([0,*range(SHORT,LONG+1)]).all()
            if model != 'nn_sized': assert actions.isin([0,SHORT,LONG]).all()
            pd.testing.assert_series_equal(actions, gate(actions,actions.index,state,gates,available),check_names=False)
            own[model] = actions
        fixed = read_actions(source/'actions.csv')
        np.testing.assert_array_equal(full_size(own['nn_sized']),fixed)
        saved = read_actions(CURVE/d['pair']/'atm_control/actions.csv')
        np.testing.assert_array_equal(full_size(saved),own['rl_fixed_frozen'])
        fits = pd.read_csv(OUT/d['pair']/'fits.csv')
        if fits.empty: continue
        check_fits(fits)
        enabled = fits.nn_sized_status.eq('enabled')
        recovered += int(enabled.sum())
        if enabled.any():
            assert fits.loc[enabled,'nn_fixed_directions_recovered'].eq(True).all()
            assert (pd.to_datetime(fits.loc[enabled,'nn_last_recent_outcome']) < pd.to_datetime(fits.loc[enabled,'fit'])).all()
        old = pd.read_csv(source/'fits.csv').set_index('fit')
        assert fits.set_index('fit').nn_sized_status.equals(old.status)
    if coverage['common_complete_pairs'] == 345:
        original = pd.read_csv('reports/atm_nn_cost_hurdles/comparison.csv')
        for model in ['nn_fixed','rl_sized']:
            for period,_ in PERIODS:
                a = comparison[comparison.model.eq(model)&comparison.period.eq(period)].iloc[0]
                b = original[original.model.eq(model)&original.period.eq(period)&original.training_hurdle.eq(.65)].iloc[0]
                np.testing.assert_allclose(a.annualized_excess_sharpe,b.annualized_excess_sharpe,atol=1e-12)
                assert a.entries == b.entries
    result = dict(status='complete', complete_windows=windows, unresolved_windows=unresolved,
                  common_complete_pairs=coverage['common_complete_pairs'], original_nn_directions_verified=True,
                  frozen_rl_directions_verified=True, enabled_nn_refits=recovered, largest_ledger_error=ledger_error,
                  training_hurdle=.65, actual_commission=.03)
    (REPORT/'audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
