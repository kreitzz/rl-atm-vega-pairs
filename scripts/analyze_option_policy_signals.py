"""Conditional quality of walk-forward trade proposals and their sizing ladder."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.option_flex_ablation import OUTPUT, REPORT, BASE, FLEX, VARIANTS, ACTIONS, FLAT, definitions, settings
from scripts.analyze_option_signals import correlation


def main():
    opportunities = pd.read_parquet('artifacts/contracts/option_signal_diagnostics/outcomes.parquet')
    records = []; substitution = []; paired = []; memory_samples = []; cfg = settings()
    for d in definitions():
        source = FLEX/d['pair']/'labels.npz'
        if not source.exists(): continue
        panel = pd.read_parquet(FLEX/d['pair']/'states.parquet')
        own = opportunities[opportunities.pair.eq(d['pair'])]
        for combo, group in own.groupby('combination'):
            states = panel.reindex(pd.DatetimeIndex(group.date))
            for side in ['short', 'long']:
                passed = (states[f'{combo}_profit_{side}_91d'].gt(0) & states[f'{combo}_count_{side}_91d'].ge(21)
                          & states[f'{combo}_profit_{side}_365d'].gt(0) & states[f'{combo}_count_{side}_365d'].ge(63))
                memory_samples.append(pd.DataFrame(dict(pair=d['pair'], combination=combo, side=side,
                    year=group.date.dt.year.to_numpy(), gate_pass=passed.to_numpy(), net_per_vega=group[side+'_net'].to_numpy())))
        with np.load(source, allow_pickle=False) as z:
            daily = z['daily']; vegas = {c: z[c+'_vegas'] for c in ['CC', 'PP']}
        original = pd.read_csv(BASE/d['pair']/'actions.csv', index_col=0, parse_dates=[0]).rl_sized
        try: old_dec = pd.read_csv(BASE/d['pair']/'rl_sized/decisions.csv')
        except pd.errors.EmptyDataError: old_dec = pd.DataFrame(columns=['status', 'signal_date'])
        try: put_dec = pd.read_csv(OUTPUT/d['pair']/'frozen_PP/decisions.csv')
        except pd.errors.EmptyDataError: put_dec = pd.DataFrame(columns=['status', 'signal_date'])
        put_status = put_dec.assign(signal_date=pd.to_datetime(put_dec.signal_date)).set_index('signal_date').status
        for date in pd.to_datetime(old_dec.loc[old_dec.status.eq('entered'), 'signal_date']):
            status = put_status.get(date, 'missing_decision')
            substitution.append(dict(pair=d['pair'], label=d['label'], date=date, put_status=status))
            if status != 'entered' or date not in panel.index: continue
            size = float(original.loc[date]); i = panel.index.get_loc(date); j = int(round((size+1)*4)); payoffs = {}
            for combo in ['CC', 'PP']:
                v = vegas[combo][i, j]; k = ACTIONS.index((combo, size))
                payoffs[combo] = daily[i, k].sum()*cfg.entry_vega_per_leg/v if v > 0 else np.nan
            if np.isfinite(list(payoffs.values())).all():
                paired.append(dict(pair=d['pair'], label=d['label'], date=date, size=size,
                                   cc_net_per_vega=payoffs['CC'], pp_net_per_vega=payoffs['PP'],
                                   pp_minus_cc=payoffs['PP']-payoffs['CC']))
        for model in ['reference_calls']+list(VARIANTS):
            if model == 'reference_calls':
                a = pd.read_csv(BASE/d['pair']/'actions.csv', index_col=0, parse_dates=[0]).rl_sized
                a = pd.Series([ACTIONS.index(('CC', float(v))) for v in a], index=a.index)
                folder = BASE/d['pair']/'rl_sized'
            else:
                folder = OUTPUT/d['pair']/model
                a = pd.read_csv(folder/'actions.csv', index_col=0, parse_dates=[0]).action_code
            try: dec = pd.read_csv(folder/'decisions.csv')
            except pd.errors.EmptyDataError: dec = pd.DataFrame(columns=['status', 'signal_date'])
            filled = set(pd.to_datetime(dec.loc[dec.status.eq('entered'), 'signal_date']))
            for date, k in a[a.ne(FLAT)].items():
                combo, size = ACTIONS[int(k)]; row = own[own.date.eq(date) & own.combination.eq(combo)]
                if row.empty: continue
                i = panel.index.get_loc(date); j = int(round((size+1)*4)); vega = vegas[combo][i, j]
                outcome = daily[i, int(k)].sum()*cfg.entry_vega_per_leg/vega if vega > 0 else np.nan
                if not np.isfinite(outcome): continue
                records.append(dict(model=model, pair=d['pair'], label=d['label'], date=date, combination=combo,
                    direction=int(np.sign(size)), size=abs(size), net_per_vega=outcome, actually_filled=date in filled,
                    better_direction=bool(np.sign(size)*row.directional_edge.iloc[0] > 0)))
    data = pd.DataFrame(records); rows = []; sizes = []
    memory = pd.concat(memory_samples, ignore_index=True)
    for columns, filename in [(['combination', 'side', 'gate_pass'], 'memory_gate_outcomes.csv'),
                              (['combination', 'side', 'gate_pass', 'year'], 'memory_gate_outcomes_by_year.csv')]:
        memory.assign(positive=memory.net_per_vega.gt(0)).groupby(columns).agg(
            observations=('net_per_vega', 'size'), mean_net_per_vega=('net_per_vega', 'mean'),
            median_net_per_vega=('net_per_vega', 'median'), fraction_positive=('positive', 'mean')).to_csv(REPORT/filename)
    sub = pd.DataFrame(substitution); sub.to_csv(REPORT/'frozen_put_entry_decisions.csv', index=False)
    sub.groupby('put_status').size().rename('attempts').to_csv(REPORT/'frozen_put_rejections.csv')
    paired = pd.DataFrame(paired); paired.to_csv(REPORT/'matched_substitution_shadow_payoffs.csv', index=False)
    for (model, sample), g in data.assign(sample='all_resolved_proposals').groupby(['model', 'sample']):
        for label, part in [(sample, g), ('actually_filled', g[g.actually_filled])]:
            rows.append(dict(model=model, sample=label, opportunities=len(part), active_pairs=part.pair.nunique(),
                size_vs_net_per_vega_spearman=correlation(part['size'], part.net_per_vega, 'spearman'),
                fraction_better_direction=part.better_direction.mean(), fraction_net_positive=part.net_per_vega.gt(0).mean(),
                mean_net_per_vega=part.net_per_vega.mean()))
            for size, h in part.groupby('size'):
                sizes.append(dict(model=model, sample=label, size=size, opportunities=len(h),
                    mean_net_per_vega=h.net_per_vega.mean(), median_net_per_vega=h.net_per_vega.median(),
                    fraction_net_positive=h.net_per_vega.gt(0).mean()))
    result = pd.DataFrame(rows)
    result.to_csv(REPORT/'policy_signal_quality.csv', index=False)
    pd.DataFrame(sizes).to_csv(REPORT/'policy_sizing_quality.csv', index=False)
    data.to_parquet(Path('artifacts/contracts/option_signal_diagnostics')/'policy_proposals.parquet', index=False)
    lines = ['# Policy direction and sizing diagnostics', '',
        'These diagnostics condition on nonflat walk-forward proposals with resolved standalone outcomes. They are not an unbiased test of all opportunities. Net payoff is measured per actual initial vega at the requested ladder size, avoiding the mechanical correlation between position size and dollar P&L. The payoff remains a standalone fixed-capital counterfactual; portfolio fills can have different integer quantities and net hedge/financing costs. The filled subset flags proposals that became portfolio entries, not a cohort attribution of the portfolio NAV.', '',
        '| Policy | Resolved proposals | Better direction | Positive net payoff | Size vs net-per-vega Spearman |',
        '|---|---:|---:|---:|---:|']
    for r in result[result['sample'].eq('all_resolved_proposals')].itertuples():
        lines.append(f'| {r.model} | {r.opportunities} | {r.fraction_better_direction:.1%} | {r.fraction_net_positive:.1%} | {r.size_vs_net_per_vega_spearman:.3f} |')
    lines += ['', '“Better direction” means the chosen long/short orientation beat its reverse. It need not have positive net payoff after costs. Constant full-size variants have no size correlation. Overlapping outcomes, repeated pairs and shared underlyings invalidate IID significance tests; none are reported.', '',
        f'Put substitution filled {sub.put_status.eq("entered").sum()} of {len(sub)} original entry dates. Among {len(paired)} common filled dates with complete standalone labels, mean put-minus-call net payoff per initial vega was {paired.pp_minus_cc.mean():.4f} (median {paired.pp_minus_cc.median():.4f}). This is an ex-post matched-date shadow-payoff diagnostic, not an implementable selection rule or full-portfolio P&L attribution.', '',
        '[All proposal/filled-subset metrics](option_flex_ablation/policy_signal_quality.csv) · [Outcomes by ladder size](option_flex_ablation/policy_sizing_quality.csv) · [Memory gate conditional outcomes](option_flex_ablation/memory_gate_outcomes.csv) · [Put entry rejections](option_flex_ablation/frozen_put_rejections.csv) · [Matched-date shadow payoffs](option_flex_ablation/matched_substitution_shadow_payoffs.csv) · [Feature relationships](option_signal_relationships.md)', '']
    REPORT.with_name('option_policy_signal_quality.md').write_text('\n'.join(lines))
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__': main()
