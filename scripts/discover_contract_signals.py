"""Small, purged chronological signal screen on cached actual-contract outcomes.

This is a standalone-payoff diagnostic, not a portfolio backtest or new holdout.
"""
from pathlib import Path
import hashlib
import json
import resource
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

SOURCE = Path('artifacts/contracts/option_signal_diagnostics/outcomes.parquet')
STATES = Path('artifacts/contracts/option_flex')
OUT = Path('artifacts/contracts/signal_discovery')
REPORT = Path('reports/signal_discovery')
VALUE = ['spread', 'rv_adjusted_value', 'spread_reversion', 'variance_value', 'relative_iv_rv']
CONTEXT = ['beta', 'correlation', 'vix_previous_close', 'momentum_12_1_a', 'momentum_12_1_b',
           'dte_a', 'dte_b', 'moneyness_a', 'moneyness_b', 'commission_vol_entry',
           'amd_earnings_days_capped', 'nvda_earnings_days_capped', 'iv_amd', 'iv_nvda',
           'trailing_rv_amd', 'trailing_rv_nvda']
SURFACE = [f'surface_{s}_{side}' for s in ['downside_skew', 'smile_curvature', 'term_slope', 'term_bend'] for side in ['a', 'b']]
MEMORY = [f'CC_{v}_{side}_{days}d' for v in ['profit', 'count'] for side in ['long', 'short'] for days in [91, 365]]
MODELS = {'spread_only': ['spread'], 'rv_value_only': ['rv_adjusted_value'], 'value': VALUE,
          'value_context': VALUE+CONTEXT, 'value_context_surface': VALUE+CONTEXT+SURFACE,
          'value_context_surface_memory': VALUE+CONTEXT+SURFACE+MEMORY}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_mask(data, cutoff):
    return (data.date < cutoff) & (data.end < cutoff) & (data.date >= cutoff-pd.DateOffset(years=5))


def predict(train, test, features):
    # Every transformation is fit only on fully completed training observations.
    x = train[features].to_numpy(float); xt = test[features].to_numpy(float)
    keep = np.isfinite(x).any(axis=0)
    x = x[:, keep]; xt = xt[:, keep]
    lo, hi = np.nanquantile(x, [.01, .99], axis=0)
    x = np.clip(x, lo, hi); xt = np.clip(xt, lo, hi)
    imp = SimpleImputer(strategy='median', add_indicator=True)
    x = imp.fit_transform(x); xt = imp.transform(xt)
    scale = StandardScaler(); x = scale.fit_transform(x); xt = scale.transform(xt)
    weights = 1/train.groupby('pair').pair.transform('size').to_numpy(float)
    weights *= len(weights)/weights.sum()
    model = Ridge(alpha=.1*len(train)).fit(x, train[['short_net', 'long_net']], sample_weight=weights)
    return model.predict(xt)


def summarize(g):
    chosen = g.action.ne(0)
    actual = np.where(g.action.gt(0), g.long_net, np.where(g.action.lt(0), g.short_net, 0.))
    traded = actual[chosen]
    direction = (g.pred_long-g.pred_short).corr(g.directional_edge, method='spearman') if g.pred_long.nunique()>1 else np.nan
    return dict(opportunities=len(g), trades=int(chosen.sum()), active_pairs=g.loc[chosen, 'pair'].nunique(),
                mean_net_per_opportunity=float(actual.mean()), mean_net_per_trade=float(traded.mean()) if len(traded) else np.nan,
                median_net_per_trade=float(np.median(traded)) if len(traded) else np.nan,
                profitable_fraction=float((traded>0).mean()) if len(traded) else np.nan,
                direction_spearman=direction)


def main():
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(SOURCE); data = data[data.combination.eq('CC')].copy()
    hashes = {}; frames = []
    for pair, g in data.groupby('pair', sort=True):
        path = STATES/pair/'states.parquet'; hashes[str(path)] = digest(path)
        p = pd.read_parquet(path).reindex(pd.DatetimeIndex(g.date))
        for col in list(p):
            if col+'_available' in p: p[col] = p[col].where(p[col+'_available'].astype(bool))
        p['variance_value'] = (p.trailing_rv_amd**2-p.iv_amd**2)-(p.trailing_rv_nvda**2-p.iv_nvda**2)
        p['relative_iv_rv'] = p.trailing_rv_amd/p.iv_amd-p.trailing_rv_nvda/p.iv_nvda
        features = list(dict.fromkeys(VALUE+CONTEXT+SURFACE+MEMORY))
        g = g.reset_index(drop=True)
        g[features] = p[features].replace([np.inf,-np.inf],np.nan).reset_index(drop=True)
        frames.append(g)
    data = pd.concat(frames, ignore_index=True).sort_values(['date','pair']).reset_index(drop=True)
    assert not data.duplicated(['date','pair']).any()
    assert (data.end > data.date).all()
    records = []; fits = []
    for period in pd.period_range('2020Q1', '2025Q3', freq='Q'):
        cutoff = period.start_time
        train = data.loc[training_mask(data, cutoff)]
        test = data.loc[data.date.dt.to_period('Q').eq(period)]
        if len(train)<500 or test.empty: continue
        assert train.end.max()<test.date.min()
        for name, features in {'training_mean': [], **MODELS}.items():
            if features: pred = predict(train, test, features)
            else:
                means = train.groupby('pair')[['short_net','long_net']].mean().mean().to_numpy()
                pred = np.tile(means, (len(test),1))
            out = test[['date','end','pair','label','short_net','long_net','directional_edge','nonoverlap']].copy()
            out['model'] = name; out['pred_short'] = pred[:,0]; out['pred_long'] = pred[:,1]
            out['action'] = np.where(pred.max(axis=1)>0, np.where(pred[:,1]>pred[:,0], 1, -1), 0)
            records.append(out)
            fits.append(dict(model=name, cutoff=cutoff, training_rows=len(train), training_pairs=train.pair.nunique(),
                             last_training_end=train.end.max(), first_test_date=test.date.min(), test_rows=len(test)))
        # Fixed economic directions, independent of fitted associations.
        for name, score in [('raw_iv_spread', -test.spread), ('raw_rv_value', test.rv_adjusted_value)]:
            out = test[['date','end','pair','label','short_net','long_net','directional_edge','nonoverlap']].copy()
            out['model']=name; out['pred_long']=score; out['pred_short']=-score
            out['action']=np.sign(score).fillna(0).astype(int); records.append(out)
        print(f'{period}: {len(train):,} completed training / {len(test):,} evaluation rows', flush=True)
    predictions = pd.concat(records, ignore_index=True)
    predictions.to_parquet(OUT/'predictions.parquet', index=False)
    pd.DataFrame(fits).to_csv(REPORT/'fit_audit.csv', index=False)
    summary=[]; annual=[]; pairs=[]
    for name,g in predictions.groupby('model'):
        for sample,h in [('all',g),('nonoverlap',g[g.nonoverlap])]:
            summary.append(dict(model=name,sample=sample,**summarize(h)))
            for year,v in h.groupby(h.date.dt.year): annual.append(dict(model=name,sample=sample,year=year,**summarize(v)))
            for pair,v in h.groupby('pair'): pairs.append(dict(model=name,sample=sample,pair=pair,label=v.label.iloc[0],**summarize(v)))
    result=pd.DataFrame(summary); result.to_csv(REPORT/'summary.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    coverage=pd.read_csv('reports/option_flex_ablation/signal_coverage.csv')
    coverage=coverage[coverage.combination.eq('CC')]
    coverage.to_csv(REPORT/'source_coverage.csv',index=False)
    manifest=dict(source_sha256=digest(SOURCE), source_manifest=json.loads(SOURCE.with_name('inputs.json').read_text()),
                  state_hashes=hashes,code_sha256=digest(Path(__file__)),models=MODELS,
                  ridge_penalty_per_row=.1, refit='quarterly', train_years=5, outcome='net dollars per initial mean leg vega',
                  period=['2020-01-01','2025-07-10'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Actual-contract signal discovery','',
      'Chronological calls/calls signal screen, before the RL policy or profitability-memory gate selects trades. Existing historical pair membership, beta 0.85–1.15 and correlation ≥0.5 filters remain. Calls-only reference features match the contract type. This does not test a broader unscreened universe or independently forecast individual legs.','',
      'Quarterly pooled ridge regressions predict the two directional net standalone payoffs. Each fit uses at most five years of observations whose entire outcomes ended strictly before the quarter. Training starts in 2018; evaluation runs from 2020 through July 10, 2025. Equal total training weight per pair; fixed regularization; train-only imputation, clipping and scaling. Choose the direction with the larger prediction only when predicted net payoff is positive. No threshold or feature-set winner is selected on these results. Raw-rule baselines always choose their economic direction, without a fitted cost hurdle. Cash/no trading earns zero excess payoff.','',
      'Labels include the existing option-entry penalty, commissions, daily delta hedging, stock execution, borrow and financing through expiry. Numbers below are net dollars per initial mean leg vega, not capital returns or portfolio Sharpe. Vega is matched initially. Standalone labels omit overlapping-book capacity limits.','',
      '| Model | Trades | Net / trade | Net / opportunity | Nonoverlap net / trade | Direction rank correlation |',
      '|---|---:|---:|---:|---:|---:|']
    for r in result[result['sample'].eq('all')].itertuples():
        thin=result[result.model.eq(r.model)&result['sample'].eq('nonoverlap')].iloc[0]
        lines.append(f'| {r.model} | {r.trades:,} | {r.mean_net_per_trade:.3f} | {r.mean_net_per_opportunity:.3f} | {thin.mean_net_per_trade:.3f} | {r.direction_spearman:.3f} |')
    lines += ['',f'Source coverage: {coverage.entered_resolved_opportunities.sum():,} resolved calls/calls opportunities out of {coverage.screened_dates.sum():,} screened pair-dates. Evaluation is conditional on both directional labels resolving; this is future outcome availability, not an executable filter. Missing histories can bias this sample. All models share the same evaluable dates.', '',
      'This is repeatedly explored research history, not an untouched holdout. Outcomes overlap and pairs share stocks. Nonoverlap rows thin each pair by outcome end date as a sensitivity diagnostic; they do not eliminate cross-pair dependence or constitute an executable scheduling rule. Earnings features retain the old 31-day cap, so they cannot accurately represent all expiry-specific event exposure. Surface marks remain candidate state variables only, never executable premiums. No annualized Sharpe is inferred from terminal cohort payoffs.','',
      '[Annual diagnostics](signal_discovery/annual.csv) · [Pair diagnostics](signal_discovery/pairs.csv) · [Fit timing audit](signal_discovery/fit_audit.csv) · [Original IV reconciliation](expiry_signal_gap.md)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(result.to_string(index=False),flush=True)

if __name__=='__main__': main()
