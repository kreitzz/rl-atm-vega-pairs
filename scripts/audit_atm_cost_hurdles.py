"""Check fixed execution costs and nested memory permissions in the sweep."""
import json
import numpy as np
import pandas as pd
from scripts.run_atm_cost_hurdles import OUTPUT, REPORT, HURDLES, PERIODS, EXECUTION_FEE, fee_name, SOURCE
from scripts.curve_trading.policy import ATM
from scripts.run_contract_walkforward import digest


def main():
    assert json.loads((OUTPUT / 'task_status.json').read_text())['status'] == 'complete'
    inputs = json.loads((OUTPUT / 'inputs.json').read_text())
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
        first = OUTPUT / d['pair'] / fee_name(HURDLES[0])
        if not (first / 'gates.npz').exists(): continue
        gates = {}
        with np.load('artifacts/contracts/atm_retrained_commission_003/' + d['pair'] + '/gates.npz') as z:
            dates = z['dates']; gates[.03] = z['gates']
        for fee in HURDLES:
            folder = OUTPUT / d['pair'] / fee_name(fee)
            with np.load(folder / 'gates.npz') as z:
                np.testing.assert_array_equal(z['dates'], dates); gates[fee] = z['gates']
            check = json.loads((folder / 'reference_check.json').read_text())
            assert check['original_curve_gates_match']
            max_memory_error = max(max_memory_error, check['original_memory_max_error'])
        with np.load(SOURCE / d['pair'] / 'gates_control.npz') as z:
            np.testing.assert_array_equal(z['dates'], dates)
            gates[.65] = z['gates'][ATM]; available = z['available'][ATM]
        eligibility = pd.read_parquet(first / 'states.parquet', columns=['eligible']).eligible.to_numpy(bool)
        valid = eligibility[:, None] & available[:, None]
        for lower, higher in zip([.03, .15, .30], [.15, .30, .65]):
            assert not (gates[higher] & ~gates[lower]).any(), 'Higher hurdle added a memory permission'
        for fee, g in gates.items():
            rows.append(dict(pair=d['pair'], label=d['label'], training_hurdle=fee,
                             eligible_direction_days=int((g & valid).sum())))
    pd.DataFrame(rows).to_csv(REPORT / 'memory_permissions.csv', index=False)
    pd.DataFrame(rows).groupby('training_hurdle').eligible_direction_days.sum().to_csv(REPORT / 'memory_permission_totals.csv')
    result = dict(status='complete', pairs=len(inputs['definitions']), sleeve_windows_checked=ledgers,
                  memory_pairs_checked=len(rows)//4, nested_memory_gates=True,
                  largest_ledger_error=max_ledger_error, largest_original_memory_error=max_memory_error,
                  execution_commission=EXECUTION_FEE)
    (REPORT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
