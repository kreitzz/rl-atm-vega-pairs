"""Build the integrated paper while preserving the original paper's exposition.

Only saved results are read. Each figure and page is rendered in a fresh guarded
process. PDF, DOCX and LaTeX share one manuscript; earlier papers are untouched.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
import zipfile
import unicodedata
from pathlib import Path

DEST=Path('paper/unified_learning_2026_10_16')
FIG=DEST/'figures';RENDER=DEST/'rendered_pages'
FACTOR=Path('reports/signal_learning_factorial')
VAR=Path('reports/signal_allocation_variance')
INTERP=Path('reports/allocation_interpretability_099')
BENCH=Path('reports/paper_allocation_benchmarks')
BLUE='#17364d';TEAL='#087f9c';GRAY='#657681'
MAIN_PAGES=16
APPENDIX_PAGES=5
CAPTIONS={
 'learning_pipeline':'Two exposure decisions. Entry policies size proposals within each signal; a separately fitted Q policy chooses the capital split between funded strategy indices.',
 'signal_example':'AMD minus NVDA selected-put IV in 2019. Shading marks excursions outside historical percentile boundaries, not executed trades. Gaps are unavailable quotes. Daily contract selection makes this a signal series, not the mark of one held contract.',
 'refit_timeline':'Monthly put-policy refitting. Only completed outcomes whose signal dates fall in the displayed 18-month window enter each fit. The selected model settings remain fixed; the tree and size assignments update.',
 'feature_validation':'Original feature exploration at the 18-month baseline. Scores use 2018–2021 validation and the archived $345 million research account; funded policy comparisons use $10 million. The three highest scores tie exactly.',
 'entry_policy':'The original tree/CEM construction. The three-state maps illustrate candidates; actual trees have up to eight states. Each candidate selects action-specific completed paths; the best 19 update sampling probabilities. The retained map sizes next month’s proposals.',
 'entry_curves':'Profit above cash since January 2022. Upper right: total straddle profits. Lower right: incremental straddle profit, bandit minus fixed full sizing. Series subtract their 2021 closing values and retain carried positions. Differences reflect both sizing and admission.',
 'entry_explanations':'Put bars show the share of visited tree nodes; straddle bars show size changes when one input is centered. The denominators differ. Both are descriptive policy-usage measures.',
 'allocation_risk':'Bandit-managed source portfolios; nine-input Q state and γ = 0.99. Gray labels are fixed straddle percentages; the remainder goes to put pairs. Lines connect discrete policies. Both axes have the same scales across periods; 2025 ends in August.',
 'allocation_schedule':'Weekly target decisions during evaluation, before next-close execution. The remaining weight goes to put pairs. These are requested strategy allocations, not drifted portfolio holdings or individual trade sizes.',
 'conditional_risk':'Realized straddle returns after λ = 0.20 Q decisions. Straddle-only choices cover 220 sessions; put-only choices cover 678. The four mixed-allocation intervals are omitted. Source costs are included; outer resizing charges are excluded.',
 'factorial':'Cell values are annualized excess Sharpe at λ = 0.20. Fixed entry size is 100%; fixed capital allocation is 50/50. Q is independently fitted on each entry family’s actual returns and costs.',
}
NUMBERS={name:i+1 for i,name in enumerate(CAPTIONS)}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def money(value):return f'${float(value)/1e6:.3f}m'


def pick(rows,**keys):
    selected=[r for r in rows if all(str(r[k])==str(v) for k,v in keys.items())]
    assert len(selected)==1,(keys,len(selected))
    return selected[0]


def table(name):
    put=read('reports/p25_validation_split/periods.csv')
    straddle=read('reports/straddle_policy_refinement/comparison.csv')
    rows=[]
    if name=='entry_results':
        for strategy,source,selected in [('Put pair',put,'screened_selected'),('Straddle',straddle,'performance60')]:
            for case,label in [(('screened_fixed50' if strategy=='Put pair' else 'fixed50'),'50%'),(('screened_fixed75' if strategy=='Put pair' else 'fixed75'),'75%'),(('screened_fixed100' if strategy=='Put pair' else 'fixed100'),'100%'),(selected,'Bandit')]:
                v=pick(source,case=case,period='validation');e=pick(source,case=case,period='retrospective_holdout' if strategy=='Put pair' else 'evaluation')
                rows.append([strategy+' · '+label,f'{float(v["annualized_excess_sharpe"]):.3f}',f'{float(e["annualized_excess_sharpe"]):.3f}',e['entries'],money(e['profit_above_cash'])])
        return ['Entry policy','2018–21 S','2022–25 S','Later entries','Later profit'],rows
    if name=='admission':
        data=read(INTERP/'bandit_admission_overlap.csv')
        for strategy,label in [('put','Put pairs'),('straddle','Straddles')]:
            r=pick(data,strategy=strategy,control='fixed100',period='evaluation')
            rows.append([label,r['common_entries'],r['bandit_only'],r['fixed_only']])
        return ['Later entry overlap vs full','Both','Bandit only','Fixed only'],rows
    if name=='risk_preferences':
        data=read(VAR/'results.csv')
        for risk in [0.,.1,.2,1.]:
            v=pick(data,variance_penalty=str(risk),period='validation');e=pick(data,variance_penalty=str(risk),period='evaluation')
            rows.append([f'λ = {risk:g}',f'{float(v["annualized_excess_sharpe"]):.3f}',money(v['profit_above_cash']),f'{float(e["annualized_excess_sharpe"]):.3f}',money(e['profit_above_cash'])])
        return ['Reward','2018–21 S','2018–21 profit','2022–25 S','2022–25 profit'],rows
    if name=='conventional_allocators':
        data=read(BENCH/'all_results.csv')
        choices={r['family']:r['case'] for r in read(BENCH/'validation_selection.csv')}
        cases=[('fixed_0.5','Fixed 50/50'),
               (choices['minimum_variance'],'Minimum variance'),
               (choices['maximum_sharpe'],'Maximum Sharpe'),
               (choices['mean_variance_lambda0.2'],'Mean–variance, λ = .20'),
               ('Q_lambda0.2','Q, λ = .20'),
               (choices['mean_variance_lambda1'],'Mean–variance, λ = 1'),
               ('Q_lambda1','Q, λ = 1')]
        for case,label in cases:
            v=pick(data,case=case,period='validation');e=pick(data,case=case,period='evaluation')
            rows.append([label,f'{float(v["annualized_excess_sharpe"]):.3f}',
                         f'{float(e["annualized_excess_sharpe"]):.3f}',money(e['profit_above_cash']),
                         f'{float(e["annualized_excess_volatility_pct"]):.3f}'])
        return ['Capital allocator','2018–21 S','2022–25 S','Later profit','Later σ (%)'],rows
    if name=='factorial':
        data=read(FACTOR/'four_way.csv')
        for cell,label in [('Fixed entry / fixed allocation','Fixed / fixed'),('Bandit entry / fixed allocation','Bandit / fixed'),('Fixed entry / Q allocation','Fixed / Q'),('Bandit entry / Q allocation','Bandit / Q')]:
            v=pick(data,cell=cell,period='validation',comparison_risk='0.2');e=pick(data,cell=cell,period='evaluation',comparison_risk='0.2')
            rows.append([label,f'{float(v["annualized_excess_sharpe"]):.3f}',f'{float(e["annualized_excess_sharpe"]):.3f}',money(e['profit_above_cash'])])
        return ['Entry / capital policy','2018–21 S','2022–25 S','Later profit'],rows
    if name=='timing_controls':
        data=read(INTERP/'q_controls.csv')
        for key,label in [('same_mean_target_hindsight','Fixed at Q’s mean target'),('saved_gamma_zero','One-period learner, γ = 0'),('actual_q','Sequential Q, γ = 0.99')]:
            v=pick(data,case='q_lambda0.2_g0.99',control=key,period='validation');e=pick(data,case='q_lambda0.2_g0.99',control=key,period='evaluation')
            rows.append([label,f'{float(v["annualized_excess_sharpe"]):.3f}',f'{float(e["annualized_excess_sharpe"]):.3f}',money(e['profit_above_cash'])])
        return ['Allocation control, λ = .20','2018–21 S','2022–25 S','Later profit'],rows
    if name=='fixed75_allocation':
        data=read(FACTOR/'all_results.csv')
        for allocator,case,label in [('fixed','fixed_0.5','Fixed 50/50'),('Q','q_lambda0.2','Q, λ = .20')]:
            v=pick(data,entry_policy='fixed75',allocator=allocator,case=case,period='validation')
            e=pick(data,entry_policy='fixed75',allocator=allocator,case=case,period='evaluation')
            rows.append([label,f'{float(v["annualized_excess_sharpe"]):.3f}',f'{float(e["annualized_excess_sharpe"]):.3f}',money(e['profit_above_cash']),f'{abs(float(e["max_cash_relative_drawdown_pct"])):.2f}%'])
        return ['Fixed-75% entries','2018–21 S','2022–25 S','Later profit','Drawdown'],rows
    if name=='annual_sharpes':
        p=read('reports/p25_validation_split/annual.csv');a=read('reports/straddle_policy_refinement/annual_comparison.csv');q=read(VAR/'annual.csv')
        for year in range(2018,2026):
            x=pick(p,case='screened_selected',year=year);y=pick(a,case='performance60',year=year)
            z=pick(q,variance_penalty='0.2',year=year);w=pick(q,variance_penalty='1.0',year=year)
            rows.append([str(year)+('*' if year==2025 else ''),*[f'{float(r["annualized_excess_sharpe"]):.3f}' for r in [x,y,z,w]]])
        return ['Year','Put bandit','Straddle bandit','Q: λ = .20','Q: λ = 1'],rows
    if name=='size_risk_sensitivity':
        data=read(FACTOR/'all_results.csv')
        for entry,label in [('fixed50','Fixed 50%'),('fixed75','Fixed 75%'),('fixed100','Fixed 100%'),('bandit','Bandit')]:
            rows.append([label,*[f'{float(pick(data,entry_policy=entry,allocator="Q",variance_penalty=str(risk),period="evaluation")["annualized_excess_sharpe"]):.3f}' for risk in [0.,.1,.2,1.]]])
        return ['Entry policy + Q','λ = 0','λ = .10','λ = .20','λ = 1'],rows
    if name=='original_sizing_controls':
        basic=read('reports/p25_strategy_audit/sizing/periods.csv')
        direct=read('reports/p25_direct_policy/funded/periods.csv')
        for source,case,label in [(basic,'fixed50','Constant 50%'),(basic,'fixed75','Constant 75%'),
                                  (basic,'fixed100','Constant 100%'),(basic,'tree_cem','Tree + CEM'),
                                  (basic,'rank_terciles','Spread-rank sizing'),(basic,'ridge_terciles','Ridge-score sizing'),
                                  (direct,'direct_linear','Direct linear policy'),(direct,'direct_additive','Direct additive policy')]:
            v=pick(source,case=case,period='validation');e=pick(source,case=case,period='evaluation')
            rows.append([label,f'{float(v["annualized_excess_sharpe"]):.3f}',f'{float(e["annualized_excess_sharpe"]):.3f}',e['entries'],money(e['profit_above_cash'])])
        return ['Put sizing policy','2018–21 S','2022–25 S','Later entries','Later profit'],rows
    if name=='original_annual_economics':
        source=read('reports/p25_validation_split/annual.csv')
        for year in range(2022,2026):
            r=pick(source,case='screened_selected',year=year)
            rows.append([str(year)+('*' if year==2025 else ''),f'{float(r["funded_return_pct"]):.3f}%',
                         f'{float(r["return_above_cash_pp"]):.3f}',f'{float(r["annualized_excess_sharpe"]):.3f}',r['entries']])
        return ['Year','Account return','Above cash (pp)','Excess Sharpe','Entries'],rows
    raise ValueError(name)


def prepare():
    DEST.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);RENDER.mkdir(exist_ok=True)
    pages=json.loads((DEST/'manuscript_template.json').read_text())
    for page in pages:
        for block in page['blocks']:
            if block['type']=='data_table':
                block['headers'],block['rows']=table(block.pop('name'));block['type']='table'
    assert len(pages)==MAIN_PAGES+APPENDIX_PAGES
    for page in pages:
        for block in page['blocks']:
            if block['type']=='equation':block['latex']=block['latex'].replace('\\\\','\\')
    dump(DEST/'manuscript.json',pages)
    sources=['reports/p25_validation_split/periods.csv','reports/p25_validation_split/annual.csv','reports/p25_validation_split/holdout_intervals.csv',
             'reports/straddle_policy_refinement/comparison.csv','reports/straddle_policy_refinement/annual_comparison.csv',
             str(INTERP/'q_controls.csv'),str(INTERP/'q_switch_conditional_results.csv'),str(INTERP/'bandit_admission_overlap.csv'),
             str(INTERP/'put_tree_usage.csv'),'reports/retained_entry_explanations/straddle_feature_importance.csv',
             str(VAR/'results.csv'),str(VAR/'annual.csv'),str(VAR/'convergence_results.csv'),str(FACTOR/'all_results.csv'),str(FACTOR/'four_way.csv'),
             'artifacts/contracts/straddle_policy_refinement/models/performance60/2025-01.json',
             'paper/figure_data/signal_example.csv','reports/p25_validation_split/validation_candidates.csv',
             str(VAR/'fixed_results.csv'),str(VAR/'decisions.csv'),
             'reports/p25_strategy_audit/sizing/periods.csv','reports/p25_direct_policy/funded/periods.csv',
             'reports/p25_strategy_audit/pairs/concentration.csv']
    sources.extend(str(BENCH/name) for name in ['all_results.csv','validation_selection.csv',
                   'cost_and_switching_diagnostics.csv','diagnostic_protocol.json','protocol.json',
                   'verification.json','fit_timing.csv','fit_moments.csv'])
    unchanged=['paper/options_relative_value_draft.pdf','paper/options_relative_value_draft.tex','paper/original_with_rl_extension/options_relative_value_extended.pdf','paper/original_with_rl_extension/research_extension.docx']
    sources.extend(str(Path(folder)/'daily_nav.parquet') for folder in [
        'artifacts/contracts/p25_validation_split/replays/screened_selected',
        'artifacts/paper_code/replays/fixed100',
        'artifacts/contracts/straddle_policy_refinement/cases/performance60/replays/tree18',
        'artifacts/contracts/atm_rv_straddle_full/replays/straddle100'])
    dump(DEST/'provenance.json',dict(main_pages=MAIN_PAGES,appendix_pages=APPENDIX_PAGES,source_hashes={s:digest(s) for s in sources},preserved_versions={s:digest(s) for s in unchanged},figures=CAPTIONS))
    write_tex(pages)
    (DEST/'README.txt').write_text('''UNIFIED PAPER: LEARNING EXPOSURE WITHIN AND ACROSS OPTIONS STRATEGIES

options_exposure_learning.pdf is the sixteen-page main paper.
options_exposure_learning_with_appendix.pdf contains the same main paper plus five technical appendix pages.
options_exposure_learning.docx is an editable copy of the complete manuscript.
options_exposure_learning.tex and manuscript.json preserve the same text, tables and figures.

The original ten-page paper and original-with-extension version are unchanged. This integrated revision restores the original economic motivation, paper discussion, worked vega example, actual spread illustration, feature exploration, training timeline and detailed tree/CEM figure. It adds the straddle signal and Q allocation as extensions of that foundation. Entry learning and allocation receive three dedicated pages each, with a separate four-way comparison. The previous integrated version is archived locally in ../unified_learning_2026_10_16_before_figure_revision.

To build the figures on another checkout, extract reproduction_inputs.zip into the repository root first. It supplies four derived daily portfolio ledgers and one saved context model, verified against provenance.json; it contains no raw option chains or credentials. Other required reports and the signal illustration are committed alongside the paper. This compact archive supports document reproduction, not a complete refit of the underlying strategies.

Page 15 adds conventional allocation benchmarks adapted from Caner et al. (2026): minimum variance, maximum Sharpe and mean–variance objectives. Sample and linear Ledoit-Wolf covariance estimators are compared using validation only within each objective family. All cases and the zero-penalty exception are retained in reports/paper_allocation_benchmarks. Same-schedule fee diagnostics distinguish allocation performance from incremental resizing charges. Realized volatility is reported alongside profit; policies do not have matched risk exposure.

Reproduce using scripts.build_unified_options_paper under scripts.run_signal_discovery_guarded. Run prepare; figure --figure NAME separately for each name in provenance.json; render --page N in a fresh worker for each of 21 pages; assemble; docx. scripts.rebuild_options_paper_guarded automates the figure, page and finalization stages sequentially. Figures are implemented in scripts.options_paper_figures at their printed width with 170-dpi PNGs and vector PDF/SVG exports. Preserve one worker, 20% of one core, 256 MiB worker and swap-growth caps. No model fitting is performed.

The PDF uses traditional academic typography: the original one-line title, centered authors and subtitle, black serif headings, plain ruled tables and centered page numbers. Charts are embedded as vectors; PNGs support previews and the editable Word export. scripts.options_paper_layout defines the matching PDF and Word style. LaTeX and Word may paginate differently in other software. Source report hashes, original-version preservation checks and layout checks are saved alongside the paper. All Sharpe ratios are annualized excess Sharpe; 2025 is partial. Data licensing remains applicable.
''')
    print(f'Prepared {MAIN_PAGES} main pages and {APPENDIX_PAGES} appendix pages',flush=True)


def figure(name):
    from scripts.options_paper_figures import build
    build(name, FIG)


def page_fingerprint(page):
    from scripts.options_paper_layout import STYLE_VERSION,FALLBACK_CHARS
    names=[b['name'] for b in page['blocks'] if b['type']=='chart']
    evidence=dict(style=STYLE_VERSION,page=page,
                  figures={name:{ext:digest(FIG/f'{name}.{ext}') for ext in ('png','pdf')} for name in names},
                  captions={name:CAPTIONS[name] for name in names})
    if any(c in b.get('text','') for b in page['blocks'] for c in FALLBACK_CHARS):
        evidence['unicode_font_fallback']='STIXGeneral + DejaVu Serif'
    return hashlib.sha256(json.dumps(evidence,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def render(page_number):
    if page_number is None:raise ValueError('Render one --page N')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scripts import build_research_publication as pub
    from scripts.options_paper_layout import page_class
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix'})
    pub.FIG=FIG;pub.CHARTS={k:dict(caption=v) for k,v in CAPTIONS.items()};pub.PAPER_FIGURES=NUMBERS
    pages=json.loads((DEST/'manuscript.json').read_text());page=pages[page_number-1]
    Page=page_class(pub.PaperPage)
    canvas=Page(page_number,page['title'])
    for block in page['blocks']:
        kind=block['type']
        if kind=='paragraph':canvas.paragraph(block['text'])
        elif kind=='heading':canvas.heading(block['text'])
        elif kind=='table':canvas.table(block)
        elif kind=='chart':canvas.image(block['name'],block['height'])
        elif kind=='equation':
            canvas.equation(block['latex'])
        elif kind=='reference':
            before=len(canvas.fig.texts);canvas.paragraph(block['text'],size=9,leading=11)
            url=re.search(r'https://\S+',block['text'])
            if url:
                for obj in canvas.fig.texts[before:]:obj.set_url(url.group())
    check=dict(page=page_number,bottom_content_points=canvas.y,overflow=canvas.y<49,
               content_sha256=page_fingerprint(page),rendered=False,vector_figures=canvas.figure_placements)
    dump(RENDER/f'layout_{page_number:02d}.json',check)
    # The PNG preview includes the charts. Final PDFs receive their vector
    # equivalents at these exact positions during assembly.
    for ax in canvas.figure_axes:ax.set_visible(False)
    with PdfPages(RENDER/f'page_{page_number:02d}.pdf') as pdf:pdf.savefig(canvas.fig)
    for ax in canvas.figure_axes:ax.set_visible(True)
    canvas.fig.savefig(RENDER/f'page_{page_number:02d}.png',dpi=110);plt.close(canvas.fig)
    check['rendered']=True
    dump(RENDER/f'layout_{page_number:02d}.json',check);print(json.dumps(check),flush=True)
    # The batch renderer collects all layout failures; assembly rejects any.


def assemble():
    from pypdf import PdfReader,PdfWriter,Transformation
    from scripts.options_paper_layout import TITLE
    pages=json.loads((DEST/'manuscript.json').read_text());meta=json.loads((DEST/'provenance.json').read_text())
    for path,expected in {**meta['source_hashes'],**meta['preserved_versions']}.items():assert digest(path)==expected,path
    checks=[json.loads((RENDER/f'layout_{i:02d}.json').read_text()) for i in range(1,len(pages)+1)]
    assert not any(x['overflow'] for x in checks)
    for page,check in zip(pages,checks):
        assert check.get('rendered') and check.get('content_sha256')==page_fingerprint(page),('Stale page',check['page'])
    for filename,indices in [('options_exposure_learning.pdf',range(1,MAIN_PAGES+1)),('technical_appendix.pdf',range(MAIN_PAGES+1,len(pages)+1)),('options_exposure_learning_with_appendix.pdf',range(1,len(pages)+1))]:
        writer=PdfWriter()
        for n in indices:
            page=PdfReader(RENDER/f'page_{n:02d}.pdf').pages[0]
            for placement in checks[n-1]['vector_figures']:
                figure=PdfReader(FIG/f'{placement["name"]}.pdf').pages[0]
                transform=Transformation().scale(placement['width']/float(figure.mediabox.width),
                                                  placement['height']/float(figure.mediabox.height))
                page.merge_transformed_page(figure,transform.translate(placement['x'],placement['y']))
            writer.add_page(page)
            writer.add_outline_item(pages[n-1]['title'],len(writer.pages)-1)
        writer.add_metadata({'/Title':TITLE,'/Author':'Adam Kreitzman and Soo Xin Yang'})
        with (DEST/filename).open('wb') as f:writer.write(f)
        writer.close()
    complete=PdfReader(DEST/'options_exposure_learning_with_appendix.pdf');assert len(complete.pages)==len(pages)
    text='\n'.join(p.extract_text() for p in complete.pages)
    for number,page in enumerate(pages):
        actual=complete.pages[number].extract_text()
        compact=lambda value:re.sub(r'\s+','',unicodedata.normalize('NFKC',str(value)))
        for block in page['blocks']:
            if block['type']=='table':
                for row in block['rows']:
                    for value in row:assert compact(value) in compact(actual),(number+1,value)
    for value in ['1.726','2.152','2.118','0.671','1.263','1.041']:assert value in text,value
    verification=dict(main_pages=MAIN_PAGES,appendix_pages=APPENDIX_PAGES,total_pages=len(pages),all_table_values_in_pdf=True,source_hashes_verified=True,
        vector_figures_embedded=sum(len(c['vector_figures']) for c in checks),
        original_versions_unchanged=True,layout_checks=checks,
        figure_hashes={str(FIG/f'{name}.{ext}'):digest(FIG/f'{name}.{ext}') for name in CAPTIONS for ext in ('png','pdf','svg')},
        output_hashes={str(p):digest(p) for p in DEST.glob('*.pdf')})
    dump(DEST/'verification.json',verification);print(f'Verified {MAIN_PAGES} main pages + {APPENDIX_PAGES} appendix pages; earlier papers unchanged',flush=True)


def write_tex(pages):
    from scripts.options_paper_layout import TITLE,SUBTITLE,AUTHORS,COURSE,DATE
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}',
             'λ':r'\ensuremath{\lambda}','γ':r'\ensuremath{\gamma}','σ':r'\ensuremath{\sigma}','Δ':r'\ensuremath{\Delta}',
             '²':r'\ensuremath{^2}','±':r'\ensuremath{\pm}','×':r'\ensuremath{\times}','−':'-','–':'--','—':'---','’':"'",'“':'``','”':"''"}
    def escape(text):return ''.join(mapping.get(c,c) for c in str(text))
    parts=[r'\documentclass[11pt,letterpaper]{article}',r'\usepackage[margin=0.75in]{geometry}',r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern,microtype,amsmath,amssymb,booktabs,array,float,graphicx,hyperref}',r'\hypersetup{hidelinks}',r'\setlength{\parindent}{0pt}',r'\setlength{\parskip}{0.4em}',
           r'\title{\textbf{'+escape(TITLE)+r'}\\[0.4em]\large '+escape(SUBTITLE)+'}',
           r'\author{'+escape(AUTHORS)+r'\\'+escape(COURSE)+'}',r'\date{'+escape(DATE)+'}',r'\begin{document}',r'\maketitle']
    for index,page in enumerate(pages):
        if index:parts.append(r'\clearpage')
        if index==MAIN_PAGES:parts.append(r'\appendix')
        if index:parts.append(r'\section*{'+escape(page['title'])+'}')
        for b in page['blocks']:
            kind=b['type']
            if index==0 and kind=='heading' and b['text']=='Abstract':
                parts.append(r'\begin{abstract}');continue
            if index==0 and kind=='heading' and b['text'].startswith('1.'):
                parts.append(r'\end{abstract}')
            if kind in ['paragraph','reference']:parts.append(escape(b['text'])+'\n')
            elif kind=='heading':parts.append(r'\subsection*{'+escape(b['text'])+'}')
            elif kind=='equation':parts.append(r'\['+b['latex']+r'\]')
            elif kind=='chart':parts.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth,height='+str(b['height'])+r'pt,keepaspectratio]{'+str(FIG/f'{b["name"]}.pdf')+r'}\caption{'+escape(CAPTIONS[b['name']])+r'}\end{figure}')
            elif kind=='table':
                parts.append(r'\begin{center}\small\resizebox{\textwidth}{!}{\begin{tabular}{'+'l'*len(b['headers'])+r'}\toprule')
                parts.append(' & '.join(escape(x) for x in b['headers'])+r'\\\midrule')
                parts.extend(' & '.join(escape(x) for x in row)+r'\\' for row in b['rows'])
                parts.append(r'\bottomrule\end{tabular}}\end{center}')
    parts.append(r'\end{document}');(DEST/'options_exposure_learning.tex').write_text('\n'.join(parts)+'\n')


def docx():
    from scripts import export_research_docx as word
    from scripts.options_paper_layout import format_word
    pages=json.loads((DEST/'manuscript.json').read_text());export=copy.deepcopy(pages)
    for page in export:
        for b in page['blocks']:
            if b['type']=='reference':b['type']='paragraph'
            if b['type']=='chart':
                import struct
                w,h=struct.unpack('>II',(FIG/f'{b["name"]}.png').read_bytes()[16:24])
                b['height']=504*h/w
    def verify(path,pages,charts,numbers):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None;document=word.ET.fromstring(z.read('word/document.xml'))
            paragraphs=[''.join(p.itertext()) for p in document.findall('.//w:p',word.NS)]
            for page in pages:
                assert page['title'] in paragraphs
                for b in page['blocks']:
                    if b['type'] in ['paragraph','heading','equation']:assert b['text'] in paragraphs
                    elif b['type']=='chart':assert z.read(f'word/media/{b["name"]}.png')==(FIG/f'{b["name"]}.png').read_bytes()
                    elif b['type']=='table':
                        for row in [b['headers']]+b['rows']:assert all(str(v) in paragraphs for v in row)
            assert len(document.findall('.//w:pageBreakBefore',word.NS))==len(pages)-1
    word.verify=verify;path=DEST/'options_exposure_learning.docx'
    word.export(export,{k:dict(caption=v) for k,v in CAPTIONS.items()},NUMBERS,path)
    with zipfile.ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
    parts=format_word(parts,word)
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    verify(path,export,CAPTIONS,NUMBERS)
    print('Verified editable Word manuscript with all tables and figures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=['prepare','figure','render','assemble','docx']);p.add_argument('--figure',choices=list(CAPTIONS));p.add_argument('--page',type=int)
    args=p.parse_args()
    if args.phase=='figure':figure(args.figure)
    elif args.phase=='render':render(args.page)
    else:globals()[args.phase]()
