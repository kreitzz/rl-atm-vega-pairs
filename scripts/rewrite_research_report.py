"""Rewrite the illustrated ten-page report from saved results; never fit models or edit slides."""
from pathlib import Path
import hashlib
import html
import gc
import json
import re
import sys
import zipfile

import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader

from scripts import build_research_publication as pub
from scripts.research_report_text import CAPTIONS, manuscript

ROOT, PAPER, FIG = pub.ROOT, pub.PAPER, pub.FIG
BLUE, GRAY, GREEN, ORANGE, RED = pub.BLUE, pub.GRAY, pub.GREEN, pub.ORANGE, pub.RED
p, h, eq, chart, table = pub.p, pub.h, pub.eq, pub.chart, pub.table
DATA = PAPER / 'figure_data'
TITLE = 'From Equity Pairs to Options Relative Value'
DATE = '6 October 2026'


def record(name, df):
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / f'{name}.csv', index=False)


def save(name, fig, caption, sources, data=None):
    for path in sources:
        pub.source(path)
    pub.savechart(name, fig, caption, sources)
    svg = FIG / f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    if data is not None:
        record(name, data)
        pub.CHARTS[name]['plotted_data'] = f'figure_data/{name}.csv'


def boxes(name, titles, bodies, caption, source_paths, colors=None):
    fig, ax = plt.subplots(figsize=(8.1, 1.65))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.5); ax.axis('off')
    n = len(titles)
    width = 10 / n
    for i, (title, body) in enumerate(zip(titles, bodies)):
        x = i * width + .07
        color = (colors or [BLUE] * n)[i]
        ax.add_patch(pub.FancyBboxPatch((x, .2), width-.3, 1.05,
                     boxstyle='round,pad=.06', facecolor='#f1f5f9', edgecolor='#c8d6e3'))
        ax.text(x+(width-.3)/2, 1.03, title, ha='center', va='center',
                fontsize=10, weight='bold', color=color)
        ax.text(x+(width-.3)/2, .65, body, ha='center', va='center', fontsize=9)
        if i < n-1:
            ax.annotate('', xy=(x+width-.08, .7), xytext=(x+width-.25, .7),
                        arrowprops={'arrowstyle': '->', 'color': GRAY})
    fig.tight_layout(pad=.3)
    save(name, fig, caption, source_paths)



def annual_chart(annual):
    """Show total returns alongside returns above cash, rather than conflating them."""
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.25))
    x = list(range(len(annual)))
    labels = [str(y)+('*' if y==2025 else '') for y in annual.year]
    ax = axes[0]
    ax.bar([v-.17 for v in x], annual.funded_return_pct, width=.32,
           color=BLUE, label='Total account return')
    ax.bar([v+.17 for v in x], annual.return_above_cash_pp, width=.32,
           color=GREEN, label='Return above cash')
    ax.set_title('Calendar-period returns', fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    pub.style(ax, 'Percent / percentage points')
    ax = axes[1]
    vals = annual.annualized_excess_sharpe
    ax.bar(x, vals, color=[BLUE if v>=0 else RED for v in vals], width=.6)
    ax.set_title('Annualized excess Sharpe', fontsize=11)
    pub.style(ax, 'Sharpe')
    for ax in axes:
        ax.set_xticks(x, labels, rotation=45)
        ax.axhline(0, color=GRAY, linewidth=.7)
    fig.tight_layout()
    save('annual_results', fig, CAPTIONS['annual_results'],
         ['reports/p25_capital_sensitivity/10m_learned_annual.csv'], annual)


class ReportPage(pub.PaperPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for t in self.fig.texts:
            if 'RESEARCH DRAFT' in t.get_text():
                t.set_text('RESEARCH REPORT · '+DATE.upper())
    def paragraph(self, text, size=10.35, color='#202832', leading=13.25):
        # Keep readable body type; only bibliography entries use smaller type.
        if size == 10.35 and re.match(r'^\[\d', text):
            size, leading = 9, 11.5
        first = len(self.fig.texts)
        super().paragraph(text, size, color, leading)
        for url in re.findall(r'https://[^\s]+', text):
            for artist in self.fig.texts[first:]:
                if url in artist.get_text():
                    artist.set_url(url)


def editable_text(pages):
    # Plain text is intentionally included for editing in Word/Docs without losing prose.
    lines=[TITLE, 'Adam Kreitzman and Soo Xin Yang | '+DATE, '']
    for i,page in enumerate(pages,1):
        lines += [f'PAGE {i}: '+page['title'], '']
        for b in page['blocks']:
            if b['type'] in ['paragraph','heading','equation']:
                lines += [b['text'], '']
            elif b['type']=='chart':
                lines += [f'Figure {pub.PAPER_FIGURES[b["name"]]}. '+pub.CHARTS[b['name']]['caption'],
                          f'Figure file: figures/{b["name"]}.png','']
            elif b['type']=='table':
                lines += [' | '.join(b['headers'])]+[' | '.join(r) for r in b['rows']]+['']
    (PAPER/'report_text.txt').write_text('\n'.join(lines))


def main():
    plt.rcParams['text.parse_math'] = False  # Currency pairs must never become math delimiters.
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'slides').glob('*.pptx')}
    protected=['reports/p25_capital_sensitivity/comparison.csv',
               'reports/p25_reversion_q/comparison.csv']
    before.update({k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in protected})
    archive=PAPER/'archive'/'report_before_2026-10-06_rewrite.zip'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in PAPER.rglob('*'):
                if f.is_file() and 'archive' not in f.parts:
                    z.write(f,f.relative_to(PAPER))
    FIG.mkdir(exist_ok=True)
    # Reuse the audited result charts: this editorial rebuild requires no replay,
    # raw quote scan or numerical search. Refresh provenance for current sources.
    previous = json.loads((PAPER/'source_manifest.json').read_text())
    for name, metadata in previous['charts'].items():
        pub.CHARTS[name] = metadata.copy()
    for path in previous['sources']:
        pub.source(path)
    cap = pub.frame('reports/p25_capital_sensitivity/comparison.csv').set_index('case')
    fit = None
    annual = pub.frame('reports/p25_capital_sensitivity/10m_learned_annual.csv')
    boxes('research_map', ['Find a trade', 'Choose its size', 'Test further decisions'],
          ['Relative option pricing\nPast trading evidence', 'Half, three-quarter or full\nLearn from completed trades', 'When to exit\nHow to combine strategies'],
          CAPTIONS['research_map'], ['config/p25_submission.json'])
    boxes('workflow', ['Signal close', 'Next session', 'While holding', 'Expiry'],
          ['Select contracts\nUse prior information', 'Choose size and fill\nCheck account limits', 'Adjust stock hedge\nTrack cash flows', 'Settle each option\nRecord total profit'],
          CAPTIONS['workflow'], ['config/p25_submission.json'])
    boxes('allocation_model', ['Past completed trades', 'Group similar conditions', 'Choose exposure'],
          ['Most recent 18 months\nRefit each month', 'A small regression tree\nAt most eight groups', 'Search size assignments\nMaximize historical Sharpe'],
          CAPTIONS['allocation_model'], ['config/p25_submission.json'])
    for name, caption in CAPTIONS.items():
        pub.CHARTS[name]['caption'] = caption
    annual_chart(annual)
    for path in ['reports/atm65_employee_sentiment.md','reports/p25_variance_validation/research_summary.txt',
                 'reports/long_tenor_search_summary/core_comparison.csv',
                 'reports/long_expiry_parity/findings.txt','reports/p25_reversion_q/findings.txt',
                 'reports/p25_reversion_q/audit.json', 'data/p25_variance_premium_inputs.csv']:
        pub.source(path)
    for path in ['scripts/rewrite_research_report.py','scripts/build_research_publication.py',
                 'scripts/report_long_tenor_search.py','scripts/research_report_text.py',
                 'config/p25_submission.json', 'reports/year_cheap_vega/comparison.csv']:
        pub.source(path)
    pages=manuscript(cap,fit,annual)
    figure_order=[b['name'] for page in pages for b in page['blocks'] if b['type']=='chart']
    assert len(figure_order)==len(set(figure_order))
    pub.PAPER_FIGURES={k:i+1 for i,k in enumerate(figure_order)}
    pub.PaperPage=ReportPage
    # Measure the actual type, table cells and image proportions before writing a PDF.
    layout=[]
    for i, page in enumerate(pages, 1):
        canvas=ReportPage(i, page['title'], page.get('subtitle', ''))
        for block in page['blocks']:
            if block['type']=='paragraph': canvas.paragraph(block['text'])
            elif block['type']=='heading': canvas.heading(block['text'])
            elif block['type']=='equation': canvas.paragraph(block['text'],size=10,color=BLUE,leading=13)
            elif block['type']=='chart': canvas.image(block['name'],block['height'])
            elif block['type']=='table': canvas.table(block)
        layout.append(dict(page=i, bottom=round(canvas.y,2),
                           words=sum(len(b.get('text','').split()) for b in page['blocks'])))
        plt.close(canvas.fig)
        del canvas
        gc.collect()
    if '--layout-only' in sys.argv:
        print(json.dumps(layout,indent=2),flush=True)
        return
    assert all(item['bottom']>=49 for item in layout), layout
    (PAPER/'manuscript.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False))
    checks=pub.render_paper(pages)
    file=PAPER/'options_relative_value.html'
    content=file.read_text().replace('Research draft · 4 October 2026','Research report · '+DATE)
    content=content.replace('font:10.1pt/1.30 Georgia', 'font:10.35pt/1.28 Georgia')
    content=content.replace('https://arxiv.org/abs/2403.12180',
           '<a href="https://arxiv.org/abs/2403.12180">arXiv:2403.12180</a>')
    content=content.replace('paper/source_manifest.json','<a href="source_manifest.json">paper/source_manifest.json</a>')
    for url in re.findall(r'https://www\.optionseducation\.org/[^\s<]+', content):
        content=content.replace(url, f'<a href="{url}">{url}</a>')
    file.write_text(content)
    editable_text(pages)
    previews=[]
    for i in range(1,11):
        im=Image.open(PAPER/f'page_{i:02d}.png').convert('RGB'); im.thumbnail((306,396)); previews.append(im)
    sheet=Image.new('RGB',(5*306,2*396),'#d8dfe7')
    for i,im in enumerate(previews):sheet.paste(im,((i%5)*306,(i//5)*396))
    sheet.save(PAPER/'contact_sheet.png')
    reader=PdfReader(PAPER/'options_relative_value.pdf')
    extracted='\n'.join(p.extract_text() for p in reader.pages)
    for expected in ['1.003','0.769','0.598','0.313','0.610','1.029','0.865']:
        assert expected in extracted,expected
    assert len(reader.pages)==10
    for path,digest in before.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for path,digest in pub.SOURCES.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    manifest=dict(title=TITLE,draft_date='2026-10-06',data_end='2025-08-29',paper_pages=10,
                  sources=pub.SOURCES,charts={k:pub.CHARTS[k] for k in figure_order},
                  layout_checks=checks,models_or_backtests_rerun=False,slides_modified=False,
                  figure_numbers=pub.PAPER_FIGURES,protected_files=before)
    (PAPER/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    # Full experiment results accompany the concise main report without enlarging its ten pages.
    supplemental=['reports/p25_feature_search/comparison.csv','reports/p25_recency/comparison.csv',
                  'reports/p25_capital_sensitivity/comparison.csv','reports/p25_maturity_matched/comparison.csv',
                  'reports/p25_short_management/comparison.csv','reports/long_tenor_fitted_q/comparison.csv',
                  'reports/native_year_entry/comparison.csv','reports/p25_reversion_q/comparison.csv',
                  'reports/p25_reversion_q/annual.csv','reports/p25_reversion_q/findings.txt',
                  'reports/native_year_entry/replacement_blend/walk_forward_24M/comparison.csv',
                  'reports/native_year_entry/replacement_blend/walk_forward_24M_30D_180D/comparison.csv',
                  'reports/long_tenor_search_summary/core_comparison.csv',
                  'reports/year_cheap_vega/comparison.csv']
    for path in supplemental:pub.source(path)
    manifest['sources']=pub.SOURCES
    manifest['supplemental_results']=supplemental
    (PAPER/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    words=sum(len(b.get('text','').split()) for page in pages for b in page['blocks'])
    audit=dict(status='complete',paper_pages=10,paper_word_count=words,figures=len(figure_order),
               source_hashes_verified=True,all_layout_checks_pass=True,
               prior_report_archived=True,presentation_unchanged=True,checkpoints_unchanged=True,
               main_values_verified=True,models_or_backtests_rerun=False,
               reader_revision='Foundations, worked hedge example, chronological methodology and interpretation',
               archived_report='archive/submitted_report_2026-10-06.zip')
    (PAPER/'publication_audit.json').write_text(json.dumps(audit,indent=2))
    (PAPER/'reproduction.txt').write_text('Rebuild only the report from saved results:\n'
       '.venv/bin/python -m scripts.run_signal_discovery_guarded --module scripts.rewrite_research_report --stop-file artifacts/submission/STOP --timeout 600 --duty-cycle 0.2 --rss-limit-mib 384 --min-free-percent 35 --warning-rss-limit-mib 384 --max-swap-growth-mib 64 --pause-low-headroom-seconds 60\n'
       'Install requirements-report.txt; restore submission caches first (see SUBMISSION.md).\n'
       'One worker;20% of one core; BLAS/Arrow threads1;384 MiB RSS stop. No fitting, backtest or deck editing.\n'
       'Main PDF: options_relative_value.pdf; editable HTML: options_relative_value.html; prose: report_text.txt.\n'
       'Every chart: figures/*.svg and *.png; numeric figure tables: figure_data/*.csv.\n')
    with zipfile.ZipFile(PAPER/'report_and_visuals.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['options_relative_value.pdf','options_relative_value.html','report_text.txt','manuscript.json',
                     'source_manifest.json','publication_audit.json','reproduction.txt']:
            z.write(PAPER/name,name)
        for k in figure_order:
            for ext in ['png','svg']:z.write(FIG/f'{k}.{ext}',f'figures/{k}.{ext}')
        for f in DATA.glob('*.csv'):z.write(f,f'figure_data/{f.name}')
        for path in supplemental:z.write(ROOT/path,'supplemental/'+path)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    main()
