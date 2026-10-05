"""Check fixed execution costs and nested memory permissions in the sweep."""
import json
import numpy as np
import pandas as pd
from scripts.run_atm_nn_cost_hurdles import OUTPUT, REPORT, HURDLES, PERIODS, EXECUTION_FEE, fee_name, SOURCE
from scripts.curve_trading.policy import ATM, gate
from scripts.curve_trading.config import COMBOS
from pathlib import Path
from scripts.run_contract_walkforward import digest
from scripts.run_curve_trading import BASE


def main():
    assert json.loads((OUTPUT / 'task_status.json').read_text())['status'] == 'complete'
    inputs = json.loads((OUTPUT / 'inputs.json').read_text())
    protocol = json.loads((OUTPUT / 'followup_protocol.json').read_text())
    HURDLES = protocol['new_hurdles']
    for path, expected in protocol['code_and_sources'].items():
        assert digest(path) == expected, path
    for path, expected in inputs['code_and_sources'].items():
        assert digest(path) == expected, path
    rows = []; max_ledger_error = max_memory_error = 0.; ledgers = 0
    for d in inputs['definitions']:
        for fee in HURDLES:
            folder = OUTPUT / d['pair'] / fee_name(fee)
            for period, _ in PERIODS:
                status = json.loads((folder / period / 'status.json').read_text())
                assert status['status'] == 'complete'
                assert status['execution_commission'] == EXECUTION_FEE and status['training_hurdle'] == fee
                max_ledger_error = max(max_ledger_error, status.get('ledger_error', 0.))
                ledgers += 1
                if status['entries']:
                    trades = pd.read_csv(folder / period / 'trades.csv')
                    entry = trades[trades.reason.eq('entry')]
                    np.testing.assert_allclose(entry.commission, abs(entry.quantity) * EXECUTION_FEE, atol=1e-10)
                    assert trades.loc[~trades.reason.eq('entry'), 'commission'].eq(0).all()
    fits = pd.read_csv(REPORT / 'fits.csv')
    from scripts.retrain_atm_commission import check_fits
    check_fits(fits)
    pairs_checked = 0
    for d in inputs['definitions']:
        for fee in HURDLES:
            folder = OUTPUT / d['pair'] / fee_name(fee)
            if not (folder / 'actions.csv').exists(): continue
            own = pd.read_csv(folder / 'actions.csv', index_col=0, parse_dates=[0]).code.astype(int)
            assert own.isin([0, 1 + ATM * 8, 1 + ATM * 8 + 7]).all()
            state = pd.read_parquet(folder / 'states.parquet')
            with np.load(folder / 'gates.npz') as z:
                gates = np.zeros((len(COMBOS), len(state), 2), bool)
                gates[ATM] = z['gates']
                available = np.zeros((len(COMBOS), len(state)), bool)
                available[ATM] = z['available']
            pd.testing.assert_series_equal(own, gate(own, own.index, state, gates, available), check_names=False)
            if fee == .03:
                ref = Path('artifacts/contracts/atm_retrained_commission_003') / d['pair']
            elif fee in [.15, .30]:
                ref = Path('artifacts/contracts/atm_cost_hurdles') / d['pair'] / fee_name(fee)
            elif fee in [1., 1.5]:
                ref = Path('artifacts/contracts/atm_higher_cost_hurdles') / d['pair'] / fee_name(fee)
            else:
                ref = None
            if ref is not None:
                reference = pd.read_parquet(ref / 'states.parquet')
                pd.testing.assert_frame_equal(state, reference)
                with np.load(ref / 'gates.npz') as z:
                    np.testing.assert_array_equal(gates[ATM], z['gates'])
            else:
                reference = pd.read_parquet(BASE / d['pair'] / '30d/states.parquet')
                pd.testing.assert_frame_equal(state, reference, check_exact=False, atol=1e-6)
                with np.load(SOURCE / d['pair'] / 'gates_control.npz') as z:
                    np.testing.assert_array_equal(gates[ATM], z['gates'][ATM])
            pairs_checked += 1
    result = dict(status='complete', pairs=len(inputs['definitions']),
                  sleeve_windows_checked=ledgers, matched_state_gate_pairs=pairs_checked,
                  fixed_size_actions_only=True, monthly_fits=len(fits),
                  nn_nonconverged_proposal_fits=int(fits.converged.eq(False).sum()),
                  nn_nonconverged_refits=int(fits.recent_converged.eq(False).sum()),
                  largest_ledger_error=max_ledger_error,
                  execution_commission=EXECUTION_FEE)
    (REPORT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
