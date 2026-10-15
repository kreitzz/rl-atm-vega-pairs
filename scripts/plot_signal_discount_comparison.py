"""Plot saved capital-allocation results; no fitting or portfolio replay.

Main figures highlight the two previously discussed Q settings. Companion
figures retain the complete gamma=0.99 sweep, including the validation winner.
"""
import csv
import json
from pathlib import Path

SOURCE = Path('reports/signal_allocation_discount')
OUT = Path('reports/signal_reward_comparison')
SHARPE = 'annualized_excess_sharpe'
PROFIT = 'profit_above_cash'


def read(name):
    with (SOURCE / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    OUT.mkdir(parents=True, exist_ok=True)
    fixed = read('fixed_results.csv')
    q = [r for r in read('q_results.csv') if float(r['gamma']) == .99]
    selected = next(r['case'] for r in json.loads(
        (SOURCE / 'selection.json').read_text())['selected'] if r['gamma'] == .99)
    exports = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 12, 'pdf.fonttype': 42})
    for period, title in [('validation', '2018–2021 validation'),
                          ('evaluation', '2022–August 2025 evaluation')]:
        controls = sorted([r for r in fixed if r['period'] == period],
                          key=lambda r: float(r['straddle_target']))
        trials = [r for r in q if r['period'] == period]
        highlights = sorted([r for r in trials if float(r['risk']) in (.1, .2)],
                            key=lambda r: float(r[PROFIT]))
        assert len(controls) == 7 and len(highlights) == 2 and len(trials) == 5
        for r in controls + highlights:
            exports.append({'period': period, 'policy': r['case'],
                            'put_weight': 1-float(r['straddle_target']) if r['kind']=='fixed' else '',
                            'straddle_weight': r.get('straddle_target', ''),
                            'variance_penalty': r.get('risk', ''),
                            'annualized_excess_sharpe': r[SHARPE],
                            'profit_above_cash': r[PROFIT]})
        for full in (False, True):
            fig, ax = plt.subplots(figsize=(11, 6.8))
            fig.subplots_adjust(left=.10, right=.97, bottom=.23, top=.77)
            gray, blue = '#6b7785', '#086ba5'
            ax.plot([float(r[PROFIT])/1e6 for r in controls],
                    [float(r[SHARPE]) for r in controls], 'o-', color=gray,
                    lw=2.2, ms=6, label='Fixed capital splits', zorder=2)
            for r in controls:
                w = float(r['straddle_target'])
                ax.annotate(f'{1-w:.0%} / {w:.0%}',
                            (float(r[PROFIT])/1e6, float(r[SHARPE])),
                            xytext=(0, -19), textcoords='offset points',
                            ha='center', fontsize=9, color=gray)
            if full:
                for r in trials:
                    x, y = float(r[PROFIT])/1e6, float(r[SHARPE])
                    ax.scatter(x, y, s=65, facecolors='white', edgecolors=blue,
                               linewidths=1.5, zorder=3)
                    if float(r['risk']) not in (.1, .2):
                        offset = ({0.: (40, 28), .025: (-112, 40), .05: (-115, 7)}[float(r['risk'])]
                                  if period == 'validation' else (6, 8))
                        ax.annotate(f'penalty {float(r["risk"]):g}', (x, y),
                                    xytext=offset, textcoords='offset points', fontsize=9, color=blue)
                    if r['case'] == selected:
                        ax.scatter(x, y, s=175, facecolors='none', edgecolors='#bf7524',
                                   lw=1.6, label='Highest validation Sharpe', zorder=4)
            ax.plot([float(r[PROFIT])/1e6 for r in highlights],
                    [float(r[SHARPE]) for r in highlights], 'o-', color=blue,
                    lw=2.8, ms=8, zorder=5, label='Dynamic Q: two highlighted settings')
            for r in highlights:
                offset = ((-12, 50) if float(r['risk']) == .2 else (25, 0)) if period == 'validation' else (12, 13)
                ax.annotate(f'Q · penalty {float(r["risk"]):g}\n'
                            f'${float(r[PROFIT])/1e6:.3f}m · Sharpe {float(r[SHARPE]):.3f}',
                            (float(r[PROFIT])/1e6, float(r[SHARPE])),
                            xytext=offset, textcoords='offset points',
                            fontsize=10, color=blue, weight='bold', linespacing=1.5)
            ax.set(xlim=(0, 2.6), ylim=(.15 if full else .40, 1.96),
                   xlabel='Cumulative profit above cash ($ millions)',
                   ylabel='Annualized excess Sharpe ratio')
            ax.xaxis.set_major_locator(MultipleLocator(.5))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.1f}m'))
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#dce2e8', lw=.8)
            ax.set_axisbelow(True)
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), frameon=False,
                      fontsize=10, ncol=2, borderaxespad=0)
            fig.text(.10, .94, 'Dynamic Q versus fixed capital allocation', fontsize=20, weight='bold')
            fig.text(.10, .885, title + ' · Discount γ = 0.99', fontsize=13, color='#425466')
            fig.text(.10, .145, 'Fixed labels = put-pair / short-straddle capital weights. Both use the best retained inner bandits.', fontsize=9)
            fig.text(.10, .105, 'Same total budget and resizing-cost model; Q changes the split weekly. Lines only connect measured settings.', fontsize=9, color='#536572')
            note = ('Full Q sweep shown; gold ring marks the highest-validation-Sharpe setting.' if full else
                    'Two researched Q settings highlighted; neither is the highest-validation-Sharpe selection. Full sweep saved separately.')
            fig.text(.10, .065, note, fontsize=8.5, color='#536572')
            fig.text(.10, .028, 'Previously researched historical periods. Strategy-index allocation, rather than a shared integer-contract inventory replay.', fontsize=8, color='#536572')
            stem = ('all_q_settings_' if full else 'fixed_vs_q_099_') + period
            for ext in ('png', 'pdf', 'svg'):
                fig.savefig(OUT / f'{stem}.{ext}', dpi=180, facecolor='white')
            plt.close(fig)
    with (OUT / 'fixed_vs_q_099.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exports[0]))
        writer.writeheader(); writer.writerows(exports)
    print('Saved validation and evaluation graphs, full-sweep companion graphs, and comparison CSV.', flush=True)


if __name__ == '__main__':
    main()
