"""Finish broad pooled evaluation without deleting pairs or changing policies."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import pandas as pd

import scripts.run_tail_put25_pooled as experiment
from scripts.curve_trading.policy import Replay
from scripts.option_flex.pooled_tail_engine import simulate
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.run_tail_put25_rule_sizing import PutMarket
from scripts.run_atm65_ablation import write_json, read_actions
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import audit, scores
from scripts.run_contract_walkforward import digest
from scripts.curve_trading.config import config
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.black_scholes import fit_quotes

BASE = experiment.OUT
OUT = Path('artifacts/contracts/tail_put25_pooled_complete')
REPORT = Path('reports/tail_put25_pooled_complete')
EVENTS = [
    dict(secid=103676, announced='2023-08-25', effective='2023-10-02',
         source='https://investors.danaher.com/2023-08-25-Danaher-Declares-pro-Rata-Dividend-of-Veralto-Common-Stock-and-Announces-Expected-When-issued-Trading-of-Veralto-Common-Stock'),
    dict(secid=106899, announced='2025-01-17', effective='2025-01-21',
         source='https://www.miaxglobal.com/alert/2025/01/17/miax-exchange-group-options-markets-corporate-action-alert-lennar'),
    dict(secid=109956, announced='2020-04-01', effective='2020-04-15',
         source='https://www.vaneck.com/globalassets/home/us/insights/media-coverage/press-releases/vaneck-announces-reverse-share-split-of-five-etfs.pdf'),
    dict(secid=127107, announced='2020-03-23', effective='2020-03-30',
         source='https://www.nasdaq.com/press-release/state-street-global-advisors-announces-reverse-splits-of-two-etfs-2020-03-23'),
]


class AccountingReplay(Replay):
    def run(self, codes, calendar):
        # Reuse the reference contract lookup/action translation, swapping only
        # the scoped lifecycle engine. The module binding is restored immediately.
        import scripts.curve_trading.policy as policy
        original = policy.simulate
        policy.simulate = lambda *args, **kwargs: simulate(*args, **kwargs, corporate_events=EVENTS)
        try:
            return super().run(codes, calendar)
        finally:
            policy.simulate = original


def save(dest, nav, trades, decisions, cfg, stock, d):
    checks = audit(nav, trades, stock, cfg, d['secids'])
    assert abs(nav.option_value.iloc[-1]) + abs(nav.stock_value.iloc[-1]) < 1e-7
    allowed = {'entry', 'split_adjustment', 'expiry_settlement', 'announced_corporate_action_exit'}
    assert set(trades.reason).issubset(allowed)
    paid = trades.reason.isin(['entry', 'announced_corporate_action_exit'])
    np.testing.assert_allclose(trades.loc[paid, 'commission'], abs(trades.loc[paid, 'quantity']) * cfg.option_commission)
    assert trades.loc[~paid, 'commission'].eq(0).all()
    entered = decisions[decisions.status.eq('entered')]
    if len(entered):
        assert (abs(entered.vega_a-entered.vega_b) / entered[['vega_a','vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
    nav.to_parquet(dest / 'daily_nav.parquet'); trades.to_csv(dest / 'trades.csv', index=False)
    decisions.to_csv(dest / 'decisions.csv', index=False)
    result = dict(status='complete', pair=d['pair'], label=d['label'], entries=len(entered),
        commission=float(trades.commission.sum()), option_cost=float(nav.option_execution_cost.sum()),
        hedge_cost=float(nav.hedge_cost.sum()), decision_counts=decisions.status.value_counts().to_dict(),
        corporate_action_exit_legs=int(trades.reason.eq('announced_corporate_action_exit').sum()),
        modeled_held_marks=int(nav.causal_held_mark_fallbacks.sum()),
        risk_breach_sessions=int(nav.post_trade_risk_breach.sum()), **scores(nav, cfg.capital), **checks)
    write_json(dest / 'status.json', result)


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    signature = json.loads((BASE / 'inputs.json').read_text()); defs = signature['definitions']
    sources = {}; cfg = replace(config(), option_commission=.03)
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    market = Market(cal); loader = PutMarket(market, cfg, sources)
    affected = {e['secid'] for e in EVENTS} | {112463, 108969}
    distributions = pd.read_parquet(ROOT / 'distributions.parquet')
    supplement = (distributions.secid.eq(108969) & distributions.distr_type.eq('5') &
        distributions.currency.eq('USD') & distributions.adj_factor.eq(1) & distributions.approx_flag.eq('0') &
        distributions.liquid_flag.eq('0') & distributions.cancel_flag.eq('0') & distributions.amount.ge(0))
    working_distributions = distributions.copy()
    working_distributions.loc[supplement, 'distr_type'] = '1'
    previous_hashes = json.loads(Path('reports/tail_put25_pooled/output_hashes.json').read_text())
    write_json(OUT / 'protocol.json', dict(events=EVENTS, models_frozen=True,
        prior_experiment='reports/tail_put25_pooled/audit.json',
        note='All pair books retained. Freeze learned actions, repair evaluation lifecycle equally for both models. Pre-announced adjusted-deliverable events close both legs at preceding session close, with unchanged costs including exit costs; no new entry on that close. Otherwise hold to own expiry. A missing held mark uses last known IV/current spot/current rate/declared dividends for valuation and hedging only; never fabricated entry liquidity. LEN stock cfadj has a 2% metadata jump on 2019-05-31 despite unchanged option strike/deliverable and unchanged reported share count; remove this jump in a working copy without changing raw stock prices. Issuer 2019 10-K reports no 2019 stock dividend. COP declared supplemental cash dividends (distribution type5, USD, unadjusted, noncancelled, nonliquidating, exact) enter known dividend PV and stock cash accounting, like ordinary cash. Refitted Greeks use observed quote midpoint and only already declared dividends; observed bid/ask remain unchanged. Training/actions are not refit after these evaluation repairs.',
        len_evidence='https://www.sec.gov/Archives/edgar/data/920760/000162828020000559/len-20191130x10k.htm'))
    for n, d in enumerate(defs, 1):
        folder = OUT / d['pair']; folder.mkdir(exist_ok=True)
        shutil.copy2(BASE / d['pair'] / 'coverage.json', folder / 'coverage.json')
        if not set(d['secids']).intersection(affected):
            for model in experiment.MODELS:
                target = folder / model
                if not target.exists():
                    shutil.copytree(BASE / d['pair'] / model, target)
            continue
        loader.retain(d['secids']); assets = [loader.asset(sid) for sid in d['secids']]
        for model in experiment.MODELS:
            source = BASE / d['pair'] / model
            target = folder / model; target.mkdir(exist_ok=True)
            status = json.loads((source / 'full/status.json').read_text())
            if status.get('cash_only'):
                if not (target / 'full').exists(): shutil.copytree(source, target, dirs_exist_ok=True)
                continue
            codes = read_actions(source / 'actions.csv')
            shutil.copy2(source / 'actions.csv', target / 'actions.csv')
            dest = target / 'full'; dest.mkdir(exist_ok=True)
            if (dest / 'status.json').exists() and json.loads((dest / 'status.json').read_text())['status'] == 'complete':
                continue
            q, s, sel, divs = [pd.concat([a[c] for a in assets], ignore_index=True) for c in ['q','s','sel','divs']]
            if 108969 in d['secids']:
                extra = working_distributions.loc[supplement].copy()
                for col in ['ex_date', 'declare_date']: extra[col] = pd.to_datetime(extra[col])
                divs = pd.concat([divs, extra], ignore_index=True)
                rows = q.secid.eq(108969) & q.bs_status.eq('unsupported_distribution')
                if rows.any():
                    fitted = fit_quotes(q.loc[rows].drop(columns=['close','dte'], errors='ignore'),
                        s, pd.read_parquet(ROOT / 'zero_curve.parquet'), working_distributions)
                    for col in ['bs_iv','bs_rate','known_dividend_pv','bs_status']:
                        q.loc[rows, col] = fitted[col].to_numpy()
                    q.loc[rows, 'delta'] = fitted.bs_delta.to_numpy()
                    q.loc[rows, 'vega'] = fitted.bs_vega.to_numpy()
            fix = s.secid.eq(106899) & s.date.ge('2019-05-31')
            s.loc[fix, 'cfadj'] /= 1.02
            engine = AccountingReplay(q, s, sel, d['secids'], divs, cfg)
            try:
                nav, trades, decisions = engine.run(codes, cal[cal >= '2018-01-01'])
                mark_book(nav, market.rates, cfg.capital)
                save(dest, nav, trades, decisions, cfg, s, d)
            except UnresolvedPosition as exc:
                write_json(dest / 'status.json', dict(status='incomplete', pair=d['pair'], label=d['label'], error=str(exc)))
            print(f'Accounting replay {n}/{len(defs)} {d["label"]} {model}: {json.loads((dest / "status.json").read_text())["status"]}', flush=True)
            del engine, q, s, sel, divs; gc.collect()
    shutil.copy2('reports/tail_put25_pooled/fits.csv', REPORT / 'fits.csv')
    experiment.OUT, experiment.REPORT = OUT, REPORT
    incomplete, coverage = experiment.report(defs, market, cfg)
    for path, sha in previous_hashes.items():
        assert digest(path) == sha, f'Frozen pre-repair outputs changed: {path}'
    statuses = [json.loads(p.read_text()) for p in OUT.glob('*/*/full/status.json')]
    record = dict(status='complete' if not incomplete else 'accounting_incomplete', unresolved=incomplete,
        requested_pairs=len(defs), observed_pairs=int(coverage.evaluated_observed_rows.gt(0).sum()),
        scored_pairs=int(coverage.scored_rows.fillna(0).gt(0).sum()),
        observed_rows=int(coverage.evaluated_observed_rows.sum()), scored_rows=int(coverage.scored_rows.fillna(0).sum()),
        corporate_action_exit_legs=sum(s.get('corporate_action_exit_legs', 0) for s in statuses),
        modeled_held_marks=sum(s.get('modeled_held_marks', 0) for s in statuses),
        learned_actions_and_prior_outputs_verified=len(previous_hashes),
        elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations='Inspected sample. Original policies/labels frozen before accounting repair. Known nonstandard corporate actions are exceptions to hold-to-expiry. Missing held marks modeled causally; no synthetic entry quotes. LEN adjustment-factor correction is metadata-only; quoted prices unchanged. Separate capped pair sleeves, not globally netted inventory.')
    write_json(REPORT / 'audit.json', record)
    (REPORT / 'findings.txt').write_text(pd.read_csv(REPORT / 'comparison.csv').to_string(index=False) + '\n\n' + json.dumps(record, indent=2) + '\n\n' + json.dumps(json.loads((OUT / 'protocol.json').read_text()), indent=2) + '\n')
    hashes = {str(p): digest(p) for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file() and p.name != 'output_hashes.json'}
    hashes.update({str(p): digest(p) for p in [Path(__file__), Path('scripts/option_flex/pooled_tail_engine.py')]})
    write_json(REPORT / 'output_hashes.json', hashes)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
