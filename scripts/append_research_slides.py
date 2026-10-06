"""Append current options research to the immutable teammate deck from Git.

Called by build_research_publication --append-only, under the resource guard.
No model fitting, backtests, browser processes, or original-slide rewrites.
"""
import hashlib
import html
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from scripts import build_research_publication as publication

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / 'slides'
APPENDIX = SLIDES / 'research_appendix'
FIGURES = APPENDIX / 'figures'
ORIGINAL_COMMIT = '8d3aa74'
BEGIN = '<!-- BEGIN APPENDED OPTIONS RESEARCH -->'
END = '<!-- END APPENDED OPTIONS RESEARCH -->'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build_append_only(replace_old_extension=False):
    original = subprocess.check_output(['git', 'show', f'{ORIGINAL_COMMIT}:slides/index.html'], cwd=ROOT)
    original_notes = subprocess.check_output(['git', 'show', f'{ORIGINAL_COMMIT}:slides/script.md'], cwd=ROOT)
    original_text = original.decode()
    original_sections = re.findall(r'<section\b.*?</section>', original_text, flags=re.S)
    kept_count = 7 if replace_old_extension else len(original_sections)
    first = kept_count + 1
    APPENDIX.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    sources = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in
               [Path(__file__), ROOT / 'scripts/build_research_publication.py']}

    def load(path):
        source = ROOT / path
        sources[path] = digest(source.read_bytes())
        return pd.read_csv(source)

    def select(frame, **conditions):
        mask = pd.Series(True, index=frame.index)
        for name, value in conditions.items():
            mask &= frame[name].eq(value)
        matches = frame.loc[mask]
        assert len(matches) == 1, (conditions, len(matches))
        return matches.iloc[0]

    def table(headers, rows):
        esc = lambda value: html.escape(str(value))
        return '<table class="data"><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'

    def points(*items):
        return '<ul class="points">' + ''.join(f'<li>{item}</li>' for item in items) + '</ul>'

    def small(text):
        return '<p class="footnote">' + text + '</p>'

    def chart(name, caption):
        return f'<figure><img src="research_appendix/figures/{name}.svg" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    def savechart(name, fig):
        fig.savefig(FIGURES / f'{name}.svg', bbox_inches='tight')
        fig.savefig(FIGURES / f'{name}.png', dpi=140, bbox_inches='tight')
        plt.close(fig)

    capital = load('reports/p25_capital_sensitivity/comparison.csv')
    training = load('reports/p25_variance_validation/initial_train_walk_forward.csv')
    management = load('reports/p25_long_management/comparison.csv')
    six_month = load('reports/long_tenor_fitted_q/comparison.csv')
    native = load('reports/native_year_entry/comparison.csv')
    native_bootstrap = load('reports/native_year_entry/paired_comparison.csv')
    short_annual = load('reports/p25_long_management/annual.csv')
    six_annual = load('reports/long_tenor_fitted_q/annual.csv')
    native_annual = load('reports/native_year_entry/annual.csv')
    smile = load('reports/smile_sizing/comparison.csv')
    feature_summary_path = 'reports/p25_variance_validation/research_summary.txt'
    sources[feature_summary_path] = digest((ROOT / feature_summary_path).read_bytes())
    all_three = load('reports/native_year_entry/replacement_blend/walk_forward_24M_sharpe_proportional/comparison.csv')
    two = load('reports/native_year_entry/replacement_blend/walk_forward_24M_30D_180D/comparison.csv')
    sizing = [select(capital, case=case) for case in ['10m_fixed50', '10m_fixed100', '10m_learned']]
    train_fixed = select(training, period='initial_training', policy='fixed100')
    train_learned = select(training, period='initial_training', policy='learned')
    best_native = select(native, case='forecast_rl_sized')
    native_fixed = select(native, case='forecast_variance_fixed50')
    matched_native = select(native, case='forecast_fixed50_same_sized_skips')
    matched_interval = select(native_bootstrap, case='forecast_rl_sized',
                              control='forecast_fixed50_same_sized_skips', block_sessions=252)
    short_hold = select(management, case='30D_locked_hold')
    short_managed = select(management, case='30D_manage_36M')
    mid_hold = select(six_month, tenor=180, case='fixed_50_hold')
    mid_managed = select(six_month, tenor=180, case='fixed50_fitted_q_conservative_aligned_runoff')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9.8, 3.1))
    scores = [r.annualized_excess_sharpe for r in sizing]
    bars = ax.barh(['Fixed 50%', 'Fixed 100%', 'Learned 50/75/100%'], scores,
                  color=['#eb6834', '#4a3aa7', '#2a78d6'])
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f'{v:.3f}' for v in scores], padding=7)
    ax.set_xlim(0, 1.15)
    publication.style(ax)
    ax.set_xlabel('Annualized daily cash-excess Sharpe')
    fig.tight_layout()
    savechart('contract_sizing', fig)
    feature_rows = []
    # The feature report separates its standard candidate and objective follow-ups.
    # Read the feature rows from the source table actually printed in the report.
    text = (ROOT / feature_summary_path).read_text()
    for key, label in [('variance_premium', 'Expected variance premium'), ('control', 'Prior learned state'),
                       ('volume_oi', 'Volume / open interest'), ('smile_shape', 'Smile shape'),
                       ('employee', 'Employee sentiment'), ('transcripts', 'Transcripts')]:
        match = re.search(r'^\s*' + key + r'\s+learned\s+([\d.\-]+)', text, flags=re.M)
        assert match is not None, key
        feature_rows.append((label, float(match.group(1))))
    fig, ax = plt.subplots(figsize=(9.8, 3.5))
    bars = ax.barh([x[0] for x in feature_rows], [x[1] for x in feature_rows],
                  color=['#2a78d6'] + ['#898781'] * 5)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f'{x[1]:.3f}' for x in feature_rows], padding=5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Annualized daily cash-excess Sharpe — original research reference')
    publication.style(ax)
    fig.tight_layout()
    savechart('feature_research', fig)

    slides = []
    def add(kicker, title, subtitle, body, notes):
        slides.append(dict(kicker=kicker, title=title, subtitle=subtitle, body=body, notes=notes))

    add('Research update', 'From the ATM proxy to actual option contracts',
        'The preceding slides remain the original presentation; this section extends the work',
        points('Keep the equity-pairs motivation and the original ATM experiment as the starting point.',
               'Test actual quotes, contract inventory, daily delta hedges, financing and costs.',
               'Explore smile regions separately, then test learned sizing and management within each strategy.',
               '<b>Current strongest result:</b> short-dated downside puts with adaptive sizing.'),
        'All original slides are preserved. Their proxy metrics describe the earlier experiment. The appended section reports the later contract-level work, with a different accounting convention. Do not compare the proxy and funded Sharpes as a controlled ablation.')

    add('Trade and accounting', 'Trade the contracts; measure returns against cash',
        'Actual puts selected by expiry and delta, with a shared cash-and-stock ledger',
        table(['Item', 'Contract-level convention'], [
            ['Instrument', 'Same option type; target absolute delta 0.25'],
            ['Initial neutrality', 'Integer contract vega matched within 10%; daily stock delta hedge'],
            ['Actual costs', '$0.03 / option contract + 0.1 vol point per leg; stock execution 1 bp'],
            ['Financing', 'Floating cash return; stock borrow spread 0.5%'],
            ['Metric', 'Daily cash-excess Sharpe × sqrt(252), sample standard deviation'],
        ]) + small('$10m funded equity. Vega is matched at entry, not continuously. Management pays additional costs when it trades.'),
        'The hedge, option positions, dividends and financing are reconciled in the actual ledger. Cash interest can make funded returns positive even in a negative excess-return year. Vega neutrality does not remove gamma, jump or changing volatility-factor exposure. Training commissions remain stricter at 65 cents; actual replay commissions are 3 cents.')

    add('Signal before sizing', 'Separate the trading signal from the allocation decision',
        'Broad research universe; each maturity keeps its own economic entry definition',
        points('Short-dated P25 state adds expected variance premium, historical normalization and RV / IV ratios.',
               'Relative-value, profitability memory and observable pair-risk inputs determine the qualified proposals.',
               'Compare fixed and learned sizes on common signals, contracts, costs and risk limits.',
               'The new one-year entry study independently forecasts next-year variance and generates monthly opportunities.'),
        'The short strategy retains 345 candidate pair sleeves, 608 qualified proposals, and 33 model inputs. Its monthly fits use completed prior outcomes. The one-year study uses the same researched roster, but does not copy short-dated entry dates, directions, sizes or gates. Pair calendars and the research universe are retrospective.')

    add('What the learner does', 'Learn size within a focused strategy',
        'Use precise algorithm labels: contextual allocation and sequential management are distinct',
        points('<b>30-day sizing:</b> tree states plus direct policy search; actions are 50%, 75% or 100%.',
               '<b>Six-month management:</b> fitted Q learns weekly hold, half-size or close decisions.',
               '<b>New one-year sizing:</b> quarterly completed-history fits choose 0%, 25%, 50% or 100%.',
               'Execution remains subject to quotes, integer contracts and common account risk checks.'),
        'The successful short-dated method is a contextual tree plus CEM optimized for portfolio Sharpe, not a neural sequential Q learner. The new native one-year method is a small-state contextual episodic direct policy optimization. The six-month fitted-Q experiment uses Bellman updates and is sequential. Claim adaptive policy value where the matched benchmark improves, rather than treating every method as the same RL algorithm.')

    add('30-day sizing', 'Adaptive sizing improves the fixed-vega controls',
        'Same qualified proposals; 2018–29 August 2025 at $10m',
        chart('contract_sizing', 'Common proposal set and risk limits. Size changes integer contracts, overlap and which entries fit in the account.') +
        table(['Sizing', 'Entries', 'Above-cash dollars'],
              [[label, int(r.entries), f'${r.net_excess_pnl:,.0f}'] for label, r in zip(['Fixed 50%', 'Fixed 100%', 'Learned sizing'], sizing)]),
        f'The learned hold-to-expiry policy improves Sharpe from {sizing[1].annualized_excess_sharpe:.3f} to {sizing[2].annualized_excess_sharpe:.3f}. In the initial {int(train_fixed.training_examples)}-example pre-2018 training fit, fixed-full and learned Sharpes are {train_fixed.annualized_sharpe:.3f} and {train_learned.annualized_sharpe:.3f}. Training uses aggregated paths and stricter costs, so compare policies within each period; do not compare absolute train and funded-test levels. Sizing changes admission and overlapping inventory.')

    management_rows = [
        ['30-day', short_hold, short_managed],
        ['90-day', select(management, case='90D_locked_hold'), select(management, case='90D_locked_manage_36M')],
        ['Six-month', mid_hold, mid_managed],
    ]
    add('Managing open positions', 'Exit decisions help some maturities more than others',
        'Hold versus learned management, using the controls defined within each study',
        table(['Maturity', 'Hold Sharpe', 'Managed Sharpe', 'Intervention'], [
            [label, f'{a.annualized_excess_sharpe:.3f}', f'{b.annualized_excess_sharpe:.3f}', intervention]
            for (label, a, b), intervention in zip(management_rows, ['Trim / paired exit', 'Trim / paired exit', 'Conservative fitted Q'])
        ]) + points('The 90-day locked comparison keeps the original 116 entries.',
                    'Six-month entry is fixed 50%; learned management adds a modest improvement.'),
        'These are separate within-study controls, not a single common entry policy across maturities. The 30-day managed checkpoint reaches 1.029. The locked 90-day management improves a negative hold strategy to about zero. Six-month conservative fitted Q improves .592 to .610; the 95% interval for that improvement includes zero. Aggressive management variants performed worse and remain in the reports.')

    add('Independent one-year entry', 'A native long-horizon signal supports a small sizing gain',
        'Forecast next-year variance; do not reuse short-dated entries',
        table(['One-year policy', 'Sharpe', 'Entries'], [
            ['Forecast direction; fixed 50%', f'{native_fixed.annualized_excess_sharpe:.3f}', int(native_fixed.entries)],
            ['Forecast direction; learned size / skip', f'{best_native.annualized_excess_sharpe:.3f}', int(best_native.entries)],
            ['Same learned skips; fixed 50%', f'{matched_native.annualized_excess_sharpe:.3f}', int(matched_native.entries)],
            ['Learn direction and size together', f'{select(native, case="rl_direction_sized").annualized_excess_sharpe:.3f}', int(select(native, case='rl_direction_sized').entries)],
        ]) + small(f'The sizing-only gain versus matched fixed 50% is {matched_interval.sharpe_gain:+.3f}; its 252-session bootstrap 95% interval is approximately [{matched_interval.unadjusted_gain_95_low:+.3f}, {matched_interval.unadjusted_gain_95_high:+.3f}].'),
        'The independent study includes all 47,610 pair/month states and 26,910 evaluation states, with 763 eligible opportunities. It uses a quarterly pooled ridge variance forecast trained only on completed 252-session targets. Learning direction as well as size failed; freezing forecast direction and learning size was better. This replaces the legacy one-year component in the blend. The improvement is exploratory and statistically uncertain.')

    wing_rows = [
        ['25-delta puts, downside', select(smile, model='put25_frozen').annualized_excess_sharpe],
        ['75-delta puts, upside strike', select(smile, model='P75_floor').annualized_excess_sharpe],
        ['25-delta calls, upside strike', select(smile, model='C25_floor').annualized_excess_sharpe],
    ]
    add('Smile exploration', 'The downside-put signal does not transfer uniformly',
        'Earlier smile experiment: separate strike regions, not extra actions in one policy',
        table(['Instrument / policy', 'Annualized excess Sharpe'],
              [[label, f'{score:.3f}'] for label, score in wing_rows]) +
        points('The opposite wing was weaker under the tested signals and quote screens.',
               'The current 30-day, six-month and one-year strategies all target 25-delta puts.'),
        'These figures belong to the earlier smile-sizing report and its original funding and eligibility conventions. They are exploratory strategy comparisons, not common-universe strike ablations and not the current $10m maturity checkpoints. A 25-delta put is the downside wing; its comparable-strike call is approximately 75 delta. Higher-delta puts and lower-delta calls describe the opposite wing. Preserve the negative findings without claiming universal call/put asymmetry.')

    short_years = short_annual.loc[short_annual.case.eq('30D_manage_36M')].set_index('year')
    mid_years = six_annual.loc[six_annual.tenor.eq(180) & six_annual.case.eq('fixed50_fitted_q_conservative_aligned_runoff')].set_index('year')
    native_years = native_annual.loc[native_annual.case.eq('forecast_rl_sized')].set_index('year')
    assert short_years.index.equals(mid_years.index) and short_years.index.equals(native_years.index)
    add('Year-by-year evidence', 'Profits and Sharpe vary materially across years',
        'Current managed 30-day, conservative six-month, and native one-year checkpoints',
        table(['Year', '30D return', '30D above cash (pp)', '30D Sharpe', '6M Sharpe', '1Y Sharpe'], [
            [str(year) + ('*' if year == 2025 else ''), f'{row.funded_return_pct:.2f}%',
             f'{row.return_above_cash_pp:+.2f}', f'{row.annualized_excess_sharpe:.2f}',
             f'{mid_years.loc[year].annualized_excess_sharpe:.2f}',
             f'{native_years.loc[year].annualized_excess_sharpe:.2f}']
            for year, row in short_years.iterrows()
        ]) + small('*2025 through 29 August. Percentage returns are calendar-period returns, not annualized. Sharpes are annualized; funded returns include interest on cash.'),
        'Do not suppress losing years. The 30-day managed policy is negative in 2019 and 2024 on a cash-excess basis; its 2022 gains are large. The native one-year policy has different weak years. These are separate researched strategies, so differences across columns are not a controlled maturity-only effect. Return percentage points above cash distinguish the trade signal from idle-cash yield.')

    add('Feature research', 'More features usually failed to improve the retained signal',
        'Same short-dated feature experiment; include successful and unsuccessful additions',
        chart('feature_research', 'Original funding reference and matched short-dated proposals. These are researched configurations, not new held-out tests.'),
        'Only the expected-variance-premium addition improves the prior learned state in this experiment. Term structure, smile, volume/open-interest, employee sentiment and transcripts generally do not add Sharpe to this checkpoint. The .961745 to 1.006305 gain has an interval that includes zero. Do not confuse the feature study funding reference with the later $10m strategy, whose retained sizing Sharpe is 1.003.')

    portfolio_rows = []
    for label, frame, case in [
        ['All three: equal weights', all_three, 'equal_weights'],
        ['All three: positive-Sharpe weights', all_three, 'sharpe_proportional24M_quarterly'],
        ['All three: covariance optimizer', all_three, 'rolling24M_quarterly_optimizer'],
        ['30D + 6M: equal weights', two, 'equal_weights'],
        ['30D + 6M: covariance optimizer', two, 'rolling24M_quarterly'],
    ]:
        a = select(frame, case=case, period='full_including_equal_warmup')
        b = select(frame, case=case, period='after_allocation_warmup')
        portfolio_rows.append([label, f'{a.annualized_excess_sharpe:.3f}', f'{b.annualized_excess_sharpe:.3f}'])
    add('Portfolio allocation', 'Past-performance weighting trails simple equal weights',
        'Quarterly fits on prior 24 months; equal weights during the 2018–2019 warmup',
        table(['Allocation', '2018–Aug 2025', '2020–Aug 2025'], portfolio_rows) +
        small('Annualized excess Sharpes. The retrospective full-period optimized blend reaches 1.107; it is a hindsight benchmark, not the historical allocation result.'),
        'Positive-Sharpe weights normalize each sleeve’s positive trailing Sharpe and assign zero to nonpositive ones; all-nonpositive fits use equal weights. Quarter-start returns are excluded from all allocation fits, verified with future perturbation and truncation checks. These are blends of saved after-cost returns; shared positions, joint risk admission and additional reallocation costs are not replayed. Component selection is still retrospective.')

    add('Interpretation', 'Adaptive decisions help within specific strategies',
        'Separate economic signal, learned sizing, sequential management, and portfolio allocation',
        points('<b>Strongest evidence:</b> P25 short-dated sizing improves common fixed-vega controls.',
               '<b>Smaller gains:</b> six-month management and native one-year sizing; uncertainty remains broad.',
               '<b>Limits:</b> weak opposite-wing results, failed direction learning, and weaker historical portfolio optimization.',
               '<b>Next test:</b> freeze the design and evaluate new dates with prospective pair formation.'),
        'The report should not claim that RL always wins, that all smile regions have the same signal, or that the contract strategy has a newly matched neural-network comparison. Price and Greek quality, American assignment, execution impact, jump risk, margin and retrospective research selection remain limitations. Vendor data remain subject to their existing licenses.')

    add('Reproducibility', 'Keep the original study and the extension together',
        'Original slides preserved; appended results linked to their saved source reports',
        points('<a href="../paper/options_relative_value.pdf">Ten-page paper</a> · <a href="options_research_appendix.pdf">Appended research slides as PDF</a>',
               '<a href="../reports/p25_capital_sensitivity/comparison.csv">Fixed versus learned sizing</a> · <a href="../reports/p25_long_management/comparison.csv">Short / 90-day management</a>',
               '<a href="../reports/long_tenor_fitted_q/findings.txt">Six-month management</a> · <a href="../reports/native_year_entry/findings.txt">Native one-year entry and sizing</a>',
               '<a href="../reports/native_year_entry/replacement_blend/walk_forward_24M_sharpe_proportional/comparison.csv">Past-only allocation results</a> · <a href="research_appendix/append_audit.json">Append-only preservation audit</a>'),
        'The paper is the previous short-dated write-up; this append does not silently rewrite it with the later long-tenor results. Source hashes record every numeric report used. No fitting or backtests were rerun to prepare these slides. The original proxy deck remains historical context, and its metric must not be conflated with the later funded contract accounting.')

    if replace_old_extension:
        # Latest user instruction supersedes the earlier append-only request.
        # Keep paper slides 2–7 verbatim, update title context, and remove the
        # old proxy extension together with its obsolete result-chart script.
        replacement_text = original_text
        for section in original_sections[kept_count:]:
            assert replacement_text.count(section) == 1
            replacement_text = replacement_text.replace(section, '', 1)
        old_title = original_sections[0]
        new_title = old_title.replace(
            'From discretized price-spread trading to sizing vega-neutral IV-spread pairs',
            'From equity pairs to learned sizing and management of actual option contracts')
        new_title = new_title.replace('extended to our own ATM vega pairs project.',
                                      'extended to our contract-level options relative-value research.')
        new_title = new_title.replace('<div class="source">rl-atm-vega-pairs</div>',
            '<div class="source">$10m funded equity · 2018–29 August 2025 · Annualized cash-excess Sharpe</div>' +
            '<div class="stat-row" style="margin-top:24px;width:100%;max-width:1000px">' +
            ''.join(f'<div class="stat-tile"><div class="num">{row.annualized_excess_sharpe:.3f}</div>' +
                    f'<div class="lbl">{label}</div></div>' for label, row in
                    [('30-day managed puts', short_managed), ('Six-month managed puts', mid_managed),
                     ('Native one-year sized puts', best_native)]) + '</div>')
        assert old_title != new_title
        replacement_text = replacement_text.replace(old_title, new_title, 1)
        stale_chart_start = replacement_text.index('  // Ours: combined-basket Sharpe')
        stale_chart_end = replacement_text.index('</script>', stale_chart_start)
        replacement_text = replacement_text[:stale_chart_start] + replacement_text[stale_chart_end:]
        assert "bar('ours-chart'" not in replacement_text
        slides[0].update(title='Our extension: contract-level option relative value',
            subtitle='Three current P25 strategies, reported separately with their own controls',
            body=table(['Current strategy', 'Annualized excess Sharpe', 'Entries'], [
                ['30-day learned sizing + management', f'{short_managed.annualized_excess_sharpe:.3f}', int(short_managed.entries)],
                ['Six-month conservative fitted-Q management', f'{mid_managed.annualized_excess_sharpe:.3f}', int(mid_managed.entries)],
                ['Native one-year forecast + learned sizing', f'{best_native.annualized_excess_sharpe:.3f}', int(best_native.entries)],
            ]) + small('2018–29 August 2025; actual contract accounting at $10m. These are separate researched strategies, not maturity-only ablations.'),
            notes='The paper discussion is preserved. The outdated ATM proxy extension is replaced by current actual-contract research. Each checkpoint is compared with its own matched control; the same economic entry signal is not assumed across maturities.')
        slides[-1].update(subtitle='Current metrics linked to saved reports and the updated research slides',
            body=slides[-1]['body'].replace('Appended research slides as PDF', 'Current options research (PDF)').replace(
                'Append-only preservation audit', 'Presentation update audit'),
            notes='The current deck preserves the teammate paper discussion and replaces the obsolete proxy strategy section. Sources record all reported metrics. The existing ten-page paper covers the short-dated study and has not been rewritten by this presentation update. No models or backtests were rerun.')
    else:
        replacement_text = original_text
    total = kept_count + len(slides)
    css = '''<style>
    .research-update .body{gap:12px}.research-update figure{margin:0;text-align:center}
    .research-update figure img{width:100%;max-width:1050px;max-height:34vh;object-fit:contain}
    .research-update figcaption{font-size:12px;color:var(--text-muted);margin-top:6px}
    .research-update .data{font-size:16px}.research-update .data th{font-size:12px}
    .research-update .points li{font-size:19px}.research-update a{color:var(--series-1)}
    .research-update .research-notes{display:none;font-size:15px;line-height:1.5;padding:16px;background:#fff8e9}
    body.show-research-notes .research-update .research-notes{display:block}
    .research-update .notes-button{position:absolute;top:18px;right:28px;font:13px system-ui;color:var(--series-1);background:white;border:1px solid var(--border);border-radius:6px;padding:6px 10px;cursor:pointer}
    @media(max-width:800px){.research-update{padding:40px 24px 50px}.research-update .data{font-size:12px}.research-update .data th{font-size:10px}.research-update .points li{font-size:17px}}
    </style>'''
    blocks = []
    for number, slide in enumerate(slides, first):
        blocks.append(f'<section class="slide research-update" id="research-slide-{number}">' +
            '<button class="notes-button" onclick="document.body.classList.toggle(\'show-research-notes\')">Research notes</button>' +
            f'<div class="kicker">{html.escape(slide["kicker"])}</div>' +
            f'<h1>{html.escape(slide["title"])}</h1><h2>{html.escape(slide["subtitle"])}</h2>' +
            f'<div class="body">{slide["body"]}</div>' +
            f'<aside class="research-notes">{html.escape(slide["notes"])}</aside>' +
            '<div class="pageno" data-n></div></section>')
    insertion = '\n' + BEGIN + '\n' + css + '\n' + '\n'.join(blocks) + '\n' + END + '\n'
    marker = '\n</div>\n\n<div class="progress"'
    assert replacement_text.count(marker) == 1
    joined = replacement_text.replace(marker, insertion + marker)
    # Removing exactly the inserted bytes recovers the whole original file,
    # including its styles, order, inline diagrams and JavaScript.
    assert joined.replace(insertion, '', 1) == replacement_text
    all_sections = re.findall(r'<section\b.*?</section>', joined, flags=re.S)
    if replace_old_extension:
        assert all_sections[1:kept_count] == original_sections[1:kept_count]
        assert len(all_sections) == total
        assert 'Independent-pair Sharpe' not in joined
        assert 'Vega-neutral ATM straddle pairs' not in joined
        assert '1.461' not in joined and "bar('ours-chart'" not in joined
    else:
        assert all_sections[:len(original_sections)] == original_sections
    notes_heading = 'Current contract-level extension' if replace_old_extension else 'Appended contract-level research'
    notes_suffix = '\n\n---\n\n# ' + notes_heading + '\n\n' + ''.join(
        f'## Slide {number} — {slide["title"]}\n\n{slide["notes"]}\n\n'
        for number, slide in enumerate(slides, first))
    joined_notes = original_notes + notes_suffix.encode()
    assert joined_notes.startswith(original_notes)
    soup = BeautifulSoup(joined, 'html.parser')
    asset_hashes = {}
    for image in soup.find_all('img'):
        path = SLIDES / image['src']
        assert path.is_file(), path
        asset_hashes[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    for name in ['paper_table3.png', 'paper_table4.png']:
        original_image = subprocess.check_output(
            ['git', 'show', f'{ORIGINAL_COMMIT}:slides/img/{name}'], cwd=ROOT)
        assert (SLIDES / 'img' / name).read_bytes() == original_image
    layout = publication.render_deck_pdf(slides, out=APPENDIX, first_slide_number=first,
                                         total_slides=total, figure_dir=FIGURES)
    appendix_pdf = APPENDIX / 'options_relative_value_slides.pdf'
    assert len(PdfReader(appendix_pdf).pages) == len(slides)
    archive = SLIDES / 'archive'
    archive.mkdir(exist_ok=True)
    for name in ['index.html', 'script.md', 'options_relative_value_slides.pdf']:
        path = SLIDES / name
        if path.exists() and (replace_old_extension or (BEGIN.encode() not in path.read_bytes() and b'# Appended contract-level research' not in path.read_bytes())):
            backup = archive / f'before_append_{digest(path.read_bytes())[:12]}_{name}'
            if not backup.exists():
                shutil.copy2(path, backup)
    (archive / 'teammate_original_8d3aa74.html').write_bytes(original)
    (archive / 'teammate_original_script_8d3aa74.md').write_bytes(original_notes)
    for name in ['paper_table3.png', 'paper_table4.png']:
        (archive / 'img').mkdir(exist_ok=True)
        shutil.copy2(SLIDES / 'img' / name, archive / 'img' / name)
    (APPENDIX / 'slide_content.json').write_text(json.dumps(slides, indent=2, ensure_ascii=False))
    (SLIDES / 'index.html').write_text(joined)
    (SLIDES / 'script.md').write_bytes(joined_notes)
    shutil.copy2(appendix_pdf, SLIDES / 'options_research_appendix.pdf')
    audit = dict(original_commit=ORIGINAL_COMMIT, original_slides=len(original_sections),
        mode='replace_obsolete_extension' if replace_old_extension else 'append_only',
        paper_slides_2_through_7_byte_identical=True,
        title_context_updated=replace_old_extension,
        obsolete_proxy_slides_removed=len(original_sections)-kept_count,
        appended_slides=len(slides), first_appended_slide=first, total_slides=total,
        original_complete_html_sha256=digest(original), original_script_sha256=digest(original_notes),
        removing_append_recovers_original_exactly=not replace_old_extension,
        original_slide_sections_byte_identical=not replace_old_extension,
        original_speaker_script_preserved_as_exact_prefix=True, models_or_backtests_rerun=False,
        paper_changed=False, combined_format='HTML', appendix_pdf_pages=len(slides),
        appended_slide_layout_checks=layout, sources=sources, image_hashes=asset_hashes,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (APPENDIX / 'append_audit.json').write_text(json.dumps(audit, indent=2))
    (APPENDIX / 'findings.txt').write_text((
        f'CURRENT PRESENTATION\n\nKept the original paper discussion (slides 2–7) byte-for-byte. Updated the title to current contract-level results. '
        f'Replaced {len(original_sections)-kept_count} old extension/conclusion slides with {len(slides)} current research slides, starting at slide {first}. '
        'Removed the obsolete ATM proxy result-chart script. The original speaker script is preserved as the prefix, followed by current research notes. '
        'All previous deck versions are archived by content hash. No models, backtests, or paper were changed.\n\n'
        'slides/index.html is the complete updated deck. slides/options_research_appendix.pdf contains the current options section.\n'
        if replace_old_extension else
        f'APPEND-ONLY PRESENTATION\n\nPreserved all {len(original_sections)} original teammate slides from Git {ORIGINAL_COMMIT} '
        f'and appended {len(slides)} research slides ({first}–{total}). '
        'Removing the appended block exactly recovers the original HTML. The original speaker script is an unchanged prefix. '
        'The old rewritten presentation is archived by content hash.\n\n'
        'slides/index.html is the complete combined deck. slides/options_research_appendix.pdf contains only the new slides. '
        'The older slides/options_relative_value_slides.pdf is the previous rewritten PDF, not the combined deck. '
        'No paper, models, backtests, original inline diagrams or original table images were changed.\n'))
    print(json.dumps({key: audit[key] for key in ['original_slides', 'appended_slides', 'total_slides',
          'original_slide_sections_byte_identical', 'peak_rss_bytes']}, indent=2), flush=True)
    return audit
