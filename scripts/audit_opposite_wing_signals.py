"""Final checks and a concise readout without changing frozen experiment outputs."""
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.curve_trading.config import COMBOS
from scripts.run_atm65_ablation import read_actions,write_json
from scripts.run_contract_walkforward import digest
from scripts.research_opposite_wing_signals import OUT,REPORT,SILOS,RULES,FEATURES

CHECKS=Path('reports/opposite_wing_signals_checks')


def main():
    started=time.monotonic();CHECKS.mkdir(parents=True,exist_ok=True)
    sources={}
    manifests=[OUT/'inputs.json',OUT/'market_sources.json',
        Path('artifacts/contracts/opposite_wing_exploration/inputs.json'),
        Path('artifacts/contracts/opposite_wing_exploration/market_sources.json'),
        Path('artifacts/contracts/tail_put25_residual_rl/inputs.json'),
        Path('artifacts/contracts/tail_put25_residual_rl/market_sources.json'),
        REPORT/'output_hashes.json']
    for manifest in manifests:
        record=json.loads(manifest.read_text());values=record.get('sources',record)
        for path,sha in values.items():
            assert path not in sources or sources[path]==sha, f'Conflicting input hash: {path}'
            sources[path]=sha
    for path,sha in sources.items():assert digest(path)==sha,f'Changed source/output: {path}'
    defs=json.loads(Path('artifacts/contracts/tail_put25_pooled/inputs.json').read_text())['definitions']
    books=entries=0
    for d in defs:
        folder=OUT/'pairs'/d['pair']
        assert len(json.loads((folder/'coverage.json').read_text()))==2
        for cp,bucket in SILOS:
            dest=folder/f'{cp}{bucket}';combo=COMBOS.index((30,cp,bucket,bucket))
            permitted=[0,1+combo*8,1+combo*8+7]
            plain=read_actions(dest/'spread/actions.csv')
            native=read_actions(dest/'spread_native_memory/actions.csv')
            assert plain.index.equals(native.index)
            assert np.isin(plain,permitted).all() and np.isin(native,permitted).all()
            assert (native.eq(0)|native.eq(plain)).all()
            for rule in RULES:
                status=json.loads((dest/rule/'full/status.json').read_text())
                assert status['status']=='complete'
                assert status.get('ledger_error',0)<1e-5
                books+=1;entries+=status['entries']
    assert books==345*2*2
    correlations=pd.read_csv(REPORT/'signal_correlations.csv')
    assert len(correlations)==2*2*3*len(FEATURES)
    comparison=pd.read_csv(REPORT/'strategy_comparison.csv')
    assert int(comparison.entries.sum())==entries
    coverage=pd.read_csv(REPORT/'coverage.csv')
    coverage.groupby('silo')[['observed_rows','eligible_rows','legacy_rows','resolved_legacy','unresolved_evaluation']].sum().to_csv(CHECKS/'coverage_totals.csv')
    candidates=['wing_iv_gap','vol_factor_residual_gap','risk_reversal_gap','right_steepness_gap','har_vrp_gap','wing_term_gap']
    shortlist=correlations[correlations['sample'].eq('broad')&correlations.feature.isin(candidates)]
    shortlist.to_csv(CHECKS/'signal_shortlist.csv',index=False)
    readout='Opposite-wing signal exploration, 2018 through August2025. All Sharpes are annualized daily excess over cash.\n\n'
    readout+=comparison[['model','entries','active_pairs','annualized_excess_sharpe','net_excess_pnl']].to_string(index=False)+'\n\n'
    readout+='Directional correlations are equal-pair within-pair rank associations with completed, standalone delta-hedged contract outcomes; negative means the lower signal favors long A / short B. These are descriptive associations, not incremental strategy returns.\n'
    readout+=shortlist[['silo','period','feature','rho','partial','pairs','ci_low','ci_high']].to_string(index=False)+'\n\n'
    readout+='Candidate priorities: C25 absolute wing-IV gap and market-vol-adjusted gap; risk reversal and right-wing steepness as small additions. Oriented absolute IV gap also has a consistent relationship to profitability conditional on the existing spread-rule direction. HAR-RV adds little stable evidence here; realized skew/kurtosis are unstable. Large P75 term correlations have only two qualifying pairs in each broad period and are unsuitable as a general sizing signal.\n\n'
    readout+='The strongest P75 fixed-rule result turns negative after excluding its three biggest winning pairs. Both wings remain weak after native profitability memory. Low daily correlations with the .792 put25 policy are useful for investigation but do not establish a profitable combination. P75 and C25 are alternative implementations of the same upside wing, not independent smile sleeves. No new RL was fitted and no strategy combination was selected.\n\n'
    readout+='Next experiment: a separate sizing-only learner per wing, with pair-level allocations and shared training where pair histories are short; a bounded ladder around the fixed rule, all observed states scored, optional-feature missing indicators and no model-availability activation veto. Evaluate whether smile/factor features add beyond absolute IV gap. Combine with ATM/put25 only after standalone sizing succeeds, replaying one common inventory book with shared-name risk limits.\n\n'
    readout+='Limitations: inspected history, multiple comparisons and overlapping cohorts; chronological subsamples are not a new holdout. Frozen pair/calendar conditioning, initial vega neutrality only, daily delta hedging, no early assignment, no shared-name risk consolidation between current pair sleeves. Canadian dividends use a declaration-time fixed-FX approximation. Quote coverage is substantially thinner for P75. Corporate-action identifier correction is documented in artifacts/contracts/opposite_wing_signals/corporate_accounting.json; frozen .756/.792 references were flat in all affected pairs at that close.\n'
    (CHECKS/'findings.txt').write_text(readout)
    write_json(CHECKS/'audit.json',dict(status='complete',source_and_output_files_verified=len(sources),
        books_checked=books,entries=entries,contract_types_and_buckets_matched=True,
        native_rule_is_subset=True,elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print((CHECKS/'audit.json').read_text(),flush=True)


if __name__=='__main__':main()
