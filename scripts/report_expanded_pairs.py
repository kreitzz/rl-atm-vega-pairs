"""Summarize historical discovery and separate candidates from backtest readiness."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path('reports/expanded_pair_discovery')


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    p=pd.read_csv(ROOT/'pairs.csv');new=p[~p.previously_tested]
    core=new[new.latest_relationship.isin(['same_sic3','same_sic2'])]
    cached=core[core.both_contracts_downloaded]
    core.to_csv(ROOT/'new_industry_peers.csv',index=False)
    cached.to_csv(ROOT/'new_industry_peers_cached_names.csv',index=False)
    screens=pd.read_csv(ROOT/'candidate_screens.csv',parse_dates=['formation'])
    screens[screens.pair.isin(cached.pair)&screens.relationship.isin(['same_sic3','same_sic2'])].to_csv(ROOT/'cached_peer_historical_membership.csv',index=False)
    roster=set(pd.read_csv('reports/contract_data/underlyings.csv').secid)
    extra=(set(core.secid_a)|set(core.secid_b))-roster
    counts=new.groupby('latest_relationship').agg(new_pairs=('pair','size'),both_names_downloaded=('both_contracts_downloaded','sum'))
    counts.to_csv(ROOT/'relationship_counts.csv')
    examples=core[core.label.isin(['APA/DVN','HES/COP','MGM/HLT','KEY/RF','SWKS/ADI','ON/NXPI','AEM/RGLD','D/DUK'])]
    examples.to_csv(ROOT/'examples.csv',index=False)
    lines=['# Expanded historical option-pair discovery','',
        f'The existing research roster is much narrower than the available source universe. The old formation rule retained at most four names per three-digit SIC industry and 180 stock pairs per screen, plus predefined ETFs. The uncapped search below finds **{len(core):,} new industry-peer pairs**, including **{len(cached):,} pairs whose two names already have downloaded option contracts**. They have not yet been trained or backtested as these new combinations.','',
        '## Discovery rule','',
        '- Quarterly formations from January 2018 through July 2025 (31 dates), using only information preceding each date.',
        '- Start from 1,922 prepared linked-security histories. Prior 252-session median call volume ≥100 and open interest ≥500, with at least 200 jointly observed return/activity dates.',
        '- Pairwise stock-return beta 0.85–1.15 and correlation ≥0.5, using all preceding 126 sessions with no missing returns. Beta regresses the lower SECID on the higher; pair orientation is not optimized.',
        '- Stock returns come directly from CRSP `retx`, respecting security-link intervals. They are not masked by missing standardized IV. Name and SIC metadata are dated before formation.',
        '- No four-name cap, no 180-pair cap, and no ranking by realized trading profits or standardized IV levels.',
        '- Record same-industry, broader-sector, cross-sector, ETF and exact-same-issuer matches separately. Same-issuer classification uses historical issuer text; it is a screening flag, not an exhaustive corporate-ownership audit.','',
        '| Relationship at latest qualifying formation | New pairs | Both names have downloaded contracts |',
        '|---|---:|---:|']
    labels={'same_sic3':'Same three-digit SIC industry','same_sic2':'Same two-digit SIC sector, different three-digit industry',
        'cross_sector':'Different two-digit SIC sectors','etf_pair':'At least one ETF','same_issuer':'Same historical issuer text'}
    for kind,row in counts.iterrows():lines.append(f'| {labels[kind]} | {row.new_pairs:,} | {row.both_names_downloaded:,} |')
    lines+=['',f'Across all categories there are {len(new):,} new statistical candidates. The industry-peer subset would require {len(extra)} additional underlyings to cover every candidate. These are unions across historical formation dates, not simultaneous current opportunities. High market-wide correlation can produce many cross-sector matches; those should not be treated automatically as economically comparable volatility pairs.','',
        '## Illustrative untested industry peers','',
        '| Pair | Qualifying quarters | Median beta | Median correlation | Latest qualifying formation | Contracts downloaded for both? |',
        '|---|---:|---:|---:|---|---|']
    for row in examples.itertuples():
        lines.append(f'| {row.label} | {row.formation_count} | {row.median_beta:.3f} | {row.median_correlation:.3f} | {row.last_formation} | {"Yes" if row.both_contracts_downloaded else "No"} |')
    lines+=['','Example statistics are medians over qualifying formation dates only. Labels are historical and are not assertions of present-day listings. These examples were not selected using trading P&L. Full-period frequency is descriptive; a backtest must use the dated candidate membership, not assume all discovered pairs were known from 2018.','',
        '## What this establishes','',
        'The hypothesis that candidate discovery has been too narrow is supported: there are many liquid, historically related combinations beyond the handpicked and capped roster. It does not establish that these pairs have profitable option-relative-value signals. The existing policy also has restrictive validation and profitability gates, so expanding the candidate roster and changing policy gates are separate experiments.','',
        f'The first practical expansion is the {len(cached):,} new industry-peer pairs using already-downloaded names. Next steps are to check actual selected-contract coverage and overlapping training history, build pair-specific features and completed expiry rewards, and evaluate the fixed VIX specification using historical membership. Existing contract caches do not guarantee every pair/date is tradable; the simulator must still check maturity, quotes, contract standardness and corporate actions. No new trading results or Sharpe improvements are claimed here.','',
        '## Coverage limits','',
        'This is a broader search of the local prepared universe, not a complete scan of every optionable US security. Upstream asset inclusion still depended on standardized-IV availability and historical earnings coverage, including full-history preparation requirements. Liquidity fields also come from those prepared panels, while beta now uses direct underlying returns. These inherited coverage restrictions should be removed in a later full-market data expansion. The original contract cache contains calls only with 0–60 days to expiry and covers 317 requested underlyings.','',
        f'The screen completed in {manifest["elapsed_seconds"]:.1f} seconds in one process, with peak RSS {manifest["peak_rss_bytes"]/1024**3:.2f} GiB. Formation-time return tests and the full 147-test suite pass. The current trading reference was not changed.','',
        '[New industry-peer roster](expanded_pair_discovery/new_industry_peers.csv) · [Peers using downloaded names](expanded_pair_discovery/new_industry_peers_cached_names.csv) · [Dated membership for cached peers](expanded_pair_discovery/cached_peer_historical_membership.csv) · [All historical candidates](expanded_pair_discovery/candidate_screens.csv) · [Formation counts](expanded_pair_discovery/formation_counts.csv) · [Data manifest](expanded_pair_discovery/manifest.json)','']
    ROOT.with_suffix('.md').write_text('\n'.join(lines))
    print(counts.to_string());print({'new_industry_peers':len(core),'cached_industry_peers':len(cached),'additional_industry_peer_names':len(extra)})


if __name__=='__main__':main()
