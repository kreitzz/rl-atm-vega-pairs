"""Preserve the submitted paper exactly, then append a focused research extension.

The original PDF pages and LaTeX are copied without editing. Six follow-up pages
explain the second signal, entry sizing and sequential capital allocation.
Render one page per fresh guarded process; no model or portfolio is refitted.
"""
import argparse
import csv
import gc
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

DEST=Path('paper/original_with_rl_extension')
PRIOR=Path('paper/interpretability_extension_2026_10_15')
REPORT=Path('reports/allocation_interpretability_099')
ORIGINAL_PDF=Path('paper/options_relative_value_draft.pdf')
ORIGINAL_TEX=Path('paper/options_relative_value_draft.tex')
CAPTIONS={
 'bandit_explanations':'Entry-policy explanations. Put bars measure actual tree-path usage; straddle bars measure how often centering an input changes proposed size. The measures have different definitions and are not causal effects.',
 'learning_layers':'The extension adds a second signal and an outer allocation decision. The original put-pair framework remains the first strategy; its retained entry policy is frozen.',
 'q_evaluation':'Four risk preferences with the original nine-input Q state versus seven fixed capital splits, 2022–August 2025. Gray labels give put/straddle weights. Lines connect discrete results rather than a fitted frontier; the table also reports validation performance.',
 'q_context':'Penalty-0.20 Q: fraction of later decisions changed when one input is set to its monthly training center, holding the other actual state values fixed.',
 'q_schedule':'The saved Q policy mostly switches between the two strategy indices. These are capital weights, rather than the fraction of cash spent on options. Only fitted decisions are shown.',
 'q_timing':'Matched controls at penalty 0.20. The average-weight control uses the realized mean target retrospectively; the discount-zero learner uses the same reward, features, data and regularization as Q.',
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write_tex(pages):
    """Keep editable LaTeX synchronized with the rendered extension content."""
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
             '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
             'λ':r'$\lambda$','γ':r'$\gamma$','–':'--','—':'---','’':"'",'−':'-'}
    def escape(text):return ''.join(mapping.get(char,char) for char in str(text))
    parts=[r'\clearpage',r'\normalsize',r'\renewcommand{\thefigure}{E\arabic{figure}}',r'\setcounter{figure}{0}']
    for index,page in enumerate(pages):
        if index:parts.append(r'\clearpage')
        parts.append(r'\section*{'+escape(page['title'])+'}')
        for block in page['blocks']:
            kind=block['type']
            if kind=='paragraph':parts.append(escape(block['text'])+'\n')
            elif kind=='heading':parts.append(r'\subsection*{'+escape(block['text'])+'}')
            elif kind=='chart':
                name=block['name']
                parts.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{'+str(DEST/'figures'/f'{name}.pdf')+r'}\caption{'+escape(CAPTIONS[name])+r'}\end{figure}')
            elif kind=='table':
                parts.append(r'\begin{center}\small\resizebox{\textwidth}{!}{\begin{tabular}{'+'l'*len(block['headers'])+r'}\toprule')
                parts.append(' & '.join(escape(v) for v in block['headers'])+r'\\\midrule')
                parts.extend(' & '.join(escape(v) for v in row)+r'\\' for row in block['rows'])
                parts.append(r'\bottomrule\end{tabular}}\end{center}')
    (DEST/'research_extension.tex').write_text('\n'.join(parts)+'\n')
    original=ORIGINAL_TEX.read_text();assert original.count(r'\end{document}')==1
    (DEST/'options_relative_value_extended.tex').write_text(original.replace(r'\end{document}',r'\input{'+str(DEST/'research_extension.tex')+'}\n'+r'\end{document}'))


def prepare():
    from pypdf import PdfReader
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'original').mkdir(exist_ok=True);(DEST/'figures').mkdir(exist_ok=True);(DEST/'rendered_pages').mkdir(exist_ok=True)
    for source in (ORIGINAL_PDF,ORIGINAL_TEX):
        target=DEST/'original'/source.name
        if target.exists():assert digest(source)==digest(target)
        else:shutil.copy2(source,target)
    for name in CAPTIONS:
        for ext in ('png','pdf','svg'):
            source=(Path('reports/signal_allocation_variance')/f'risk_preferences_paper.{ext}' if name=='q_evaluation' else PRIOR/'figures'/f'{name}.{ext}')
            target=DEST/'figures'/f'{name}.{ext}'
            if source.exists():shutil.copy2(source,target)
            else:assert target.exists(),'Missing publication figure: '+str(target)
    original_pages=len(PdfReader(ORIGINAL_PDF).pages)
    p=lambda text:dict(type='paragraph',text=text)
    h=lambda text:dict(type='heading',text=text)
    c=lambda name,height:dict(type='chart',name=name,height=height)
    def table(headers,rows,widths):return dict(type='table',headers=headers,rows=rows,widths=widths)
    pages=[]
    def page(title,blocks):pages.append(dict(title=title,blocks=blocks))
    result=read('reports/straddle_policy_refinement/comparison.csv')
    rows=[]
    for key,label in [('fixed50','Fixed 50%'),('fixed75','Fixed 75%'),('fixed100','Fixed 100%'),('performance60','Contextual sizing')]:
        v=next(r for r in result if r['case']==key and r['period']=='validation')
        e=next(r for r in result if r['case']==key and r['period']=='evaluation')
        rows.append([label,f'{float(v["annualized_excess_sharpe"]):.3f}',f'{float(e["annualized_excess_sharpe"]):.3f}',e['entries'],f'${float(e["profit_above_cash"])/1e6:.3f}m'])
    page('Extension 1. Adding a second volatility signal',[
      p('The preceding paper studies one economic signal: relative implied volatility in approximately 30-day, 25-delta put pairs. Its conclusion proposes allocating across several signals. This follow-up implements a first version of that extension while retaining the original signal, actual-contract simulator and entry-sizing result.'),
      h('Persistent realized-versus-implied differences'),
      p('The second signal asks whether a name’s recent options have repeatedly implied more volatility than subsequently realized. For each name, we average completed 30-day realized volatility minus entry ATM call IV over the preceding year. Only outcomes finished before the decision enter this memory. We rank eligible names daily and sell the bottom decile: the names with the most persistent negative RV-minus-IV history. This simple persistence forecast avoids changing the original pair signal.'),
      p('We express the signal through approximately 30-day ATM straddles: equal-count short calls and puts with the same underlying, strike, expiry and adjustment. The combined reference vega is $450 per IV point. Delta is hedged daily and contracts normally run to expiry. The original call-IV input is retained when moving from calls to straddles. The simulator applies the same funded cash-flow accounting, evaluation execution charges, dividends, financing and admission framework as before.'),
      h('Learning size within the second signal'),
      p('A monthly ridge score partitions proposals into three contexts using 60 months of completed history. The inputs include IV, completed RV-minus-IV memory, VIX, recent IV/VIX changes and a causal fixed-half benchmark’s trailing performance and capital use. Exhaustive search over 27 maps assigns 50%, 75% or 100% size to each context to maximize daily training Sharpe. All 35,774 proposals remain; early months use half size until enough completed examples exist.'),
      table(['Straddle entry policy','2018–21 Sharpe','2022–25 Sharpe','Later entries','Later profit'],rows,[174,90,90,78,80]),
      p('The retained bandit improves the historical point estimate against the three fixed sizes. Straddles also generate substantially more dollar profit and return volatility than the original put-pair strategy. This difference creates an allocation problem: how selectively should the portfolio take the larger short-volatility exposure?')
    ])
    page('Extension 2. Interpreting entry-size decisions',[
      c('bandit_explanations',174),
      p('The original put learner can be explained by following its exact tree path and reading the size assigned to the resulting leaf. Completed 365-day profit appears in 18.7% of later visited decision nodes; mean IV, commission relative to vega and stock-return correlation account for 14.9%, 12.7% and 12.3%. These frequencies describe what the model uses, not whether each feature causes profit.'),
      p('For straddles, standardized ridge contributions add exactly to a context score; thresholds and the saved context-to-size map then determine exposure. Centering IV changes 36.6% of fitted later proposal sizes. Recent VIX change changes 15.3%, normalized RV-minus-IV memory 12.9% and VIX level 11.9%. The action map need not be monotonic in the score because it optimizes aggregate Sharpe. All 608 put-pair sizes and 35,774 straddle sizes reproduce the frozen policies exactly.'),
      table(['Later entries versus fixed full','Both enter','Bandit only','Fixed only'],[
        ['Original put-pair strategy','74','36','1'],['New straddle strategy','897','307','43']],[260,84,84,84]),
      p('The entry benefit combines exposure choices and admission, extending the original paper’s capacity diagnostic. Smaller commitments can leave room for later trades. The put bandit admits 110 later entries versus 75 at full size; the straddle bandit admits 1,204 versus 940. These are complete funded policies, rather than size comparisons on identical executed portfolios.'),
      p('There is also descriptive outcome separation. Among quote-available completed straddle labels, full-size profit averages $1,631 for later proposals assigned 100%, versus $16 for those assigned 50%. This subset is conditioned on available historical labels. Three within-date size shuffles reach later Sharpes of 0.834, 0.928 and 1.044, versus the bandit’s 1.041; one control wins. We therefore separate decision transparency and funded performance from stronger claims of predictive sizing.')
    ])
    page('Extension 3. Learning the capital split',[
      c('learning_layers',112),
      p('We freeze the retained put-pair and straddle entry policies and let an outer Q learner allocate one $10 million budget between their funded strategy indices. Actions choose put/straddle weights of 100/0, 90/10, 75/25, 50/50, 25/75, 10/90 or 0/100. Weights sum to one. All seven fixed benchmarks use the same menu and start at their own split. There is no outer leverage, cash action or additional portfolio stop.'),
      h('State and execution'),
      p('The state combines put-spread strength, the put implied-versus-realized variance-spread feature, straddle IV-versus-RV richness, completed straddle premium, relative strategy volatility, strategy return correlation, VIX, relative recent Sharpe and previous straddle weight. Market information uses the prior close. A weekly decision takes effect at the next close, leaving the first session at old weights. Holdings then drift and affect the cost of the next resize.'),
      p('The action-value function is quadratic in straddle weight, with coefficients linear in standardized state inputs. Monthly fitted Q iteration uses the preceding 104 completed weeks, requiring 52. The seven prior allocations crossed with seven actions are counterfactuals on the same week; they do not create independent observations. Q uses 50/50 before enough history is available, while fixed controls always use their own target.'),
      h('Reward and sequential value'),
      p('Weekly reward sums daily excess returns normalized by completed training-window reference volatility, then subtracts λ times squared normalized returns. λ controls risk preference. The separate discount γ = 0.99 weights continuation value. Ridge regularization and 200 Bellman updates are held fixed. A matched γ = 0 learner provides a one-period contextual-allocation control.'),
      p('This is fractional strategy-index allocation, preserving source returns and adding source-based resizing charges. It is not a shared integer-contract replay across both signals. The straddle sleeve introduces net short vega, so the combined portfolio is not vega neutral. The extension tests capital allocation across the existing strategies, while retaining the original contract-level study as its foundation.')
    ])
    fixed=read('reports/signal_allocation_discount/fixed_results.csv');q=read('reports/signal_allocation_variance/results.csv')
    rows=[]
    for risk in [0,.1,.2,1]:
        v=next(r for r in q if float(r['variance_penalty'])==risk and r['period']=='validation')
        e=next(r for r in q if float(r['variance_penalty'])==risk and r['period']=='evaluation')
        rows.append([f'λ = {risk:g}',f'{float(v["annualized_excess_sharpe"]):.3f}',f'${float(v["profit_above_cash"])/1e6:.3f}m',f'{float(e["annualized_excess_sharpe"]):.3f}',f'${float(e["profit_above_cash"])/1e6:.3f}m'])
    page('Extension 4. Fixed versus dynamic allocation',[
      c('q_evaluation',224),
      table(['Risk penalty','2018–21 Sharpe','2018–21 profit','2022–25 Sharpe','2022–25 profit'],rows,[100,103,103,103,103]),
      p('We report 2018–2021 validation and 2022–August 2025 walk-forward out-of-sample performance. Each monthly fit uses only outcomes completed before its decisions; earlier completed evaluation-period weeks can therefore enter later refits. The four settings illustrate risk preferences; they are not all validation winners. Features, source bandits, allocation choices, execution and discount remain identical.'),
      p('At penalty 0.20, Q reaches later Sharpe 1.726 and $1.306 million profit above cash. The highest fixed-split Sharpe is 1.612 at 90/10, earning $435,181. Penalty 0.10 retains more profit at Sharpe 1.491. Raising the penalty to 1.00 reaches Sharpe 2.152 while reducing profit to $1.141 million. Straddle-heavy fixed mixes still earn more dollars: these are different risk–return tradeoffs, rather than dominance on every objective.'),
      p('The zero-penalty policy provides a useful contrast. Its validation Sharpe is 1.427, but later Sharpe falls to 0.671. Without a variance penalty, the reward favors profit and the policy allocates more often to the higher-profit straddle signal. A strong validation Sharpe alone does not ensure that this risk preference transfers to a different period.')
    ])
    condition=read(REPORT/'q_switch_conditional_results.csv')
    rows=[]
    for choice,label in [('straddle_only','Q chose straddles'),('put_only','Q chose put pairs'),('interior_mix','Q chose a mixed split')]:
        r=next(r for r in condition if r['chosen_allocation']==choice and r['source_strategy']=='straddle')
        rows.append([label,f'{float(r["mean_daily_excess_bps"]):.2f}',f'{float(r["daily_excess_volatility_bps"]):.2f}',f'{float(r["conditional_annualized_sharpe"]):.2f}'])
    page('Extension 5. Why the switches help',[
      table(['Subsequent straddle performance','Mean/day (bps)','Daily vol. (bps)','Conditional Sharpe'],rows,[224,90,90,108]),
      p('The allocator’s effective distinction is when straddle carry compensates for its risk. After Q chooses straddles, their average excess return is higher and daily fluctuations lower than on dates when Q chooses put pairs. On the latter dates, the put strategy has conditional Sharpe 1.78. The allocator can therefore improve Sharpe while forgoing some average straddle profit.'),
      c('q_context',122),
      c('q_schedule',119),
      p('Of 191 later decisions, 141 choose put pairs alone and 46 straddles alone; only four choose interior splits. Mean straddle target is 24.9%. In later low-VIX weeks it averages 44.9%, versus 4.7% in high-VIX weeks. This median split describes observed behavior, not a fitted threshold rule; the earlier relationship differs. The inputs jointly influence decisions through monthly refits.'),
      p('The table uses source returns strictly after the decision’s execution close through the next execution close. Source costs are included; additional allocation-resizing charges are excluded. These are conditional statistics on noncontiguous historical dates, not separately deployable strategy Sharpes. Centering inputs is a policy sensitivity check, not a causal feature test.')
    ])
    controls=read(REPORT/'q_controls.csv');rows=[]
    for key,label in [('same_mean_target_hindsight','Fixed at Q’s mean target'),('saved_gamma_zero','One-period learner, γ = 0'),('actual_q','Sequential Q, γ = 0.99')]:
        v=next(r for r in controls if r['case']=='q_lambda0.2_g0.99' and r['control']==key and r['period']=='validation')
        e=next(r for r in controls if r['case']=='q_lambda0.2_g0.99' and r['control']==key and r['period']=='evaluation')
        rows.append([label,f'{float(v["annualized_excess_sharpe"]):.3f}',f'{float(e["annualized_excess_sharpe"]):.3f}',f'${float(e["profit_above_cash"])/1e6:.3f}m'])
    page('Extension 6. Timing, continuation and interpretation',[
      c('q_timing',155),
      table(['Penalty-0.20 policy or control','2018–21 Sharpe','2022–25 Sharpe','Later profit'],rows,[220,97,97,98]),
      p('A constant target equal to Q’s later mean straddle allocation gives Sharpe 1.387 and $841,251 profit, versus Q’s 1.726 and $1.306 million. The earlier comparison is 0.828 versus 1.215. Average exposure alone therefore does not explain the point estimate. These targets use the realized mean retrospectively and are diagnostic rather than validation-selected policies.'),
      p('Shifting the learned action schedule through 49 nonzero circular offsets, preserving its action counts and most serial structure, gives median later Sharpe 0.744; none reaches Q. Three of 49 earlier rotations reach or exceed the actual policy. Wealth and resizing costs are recomputed. This is descriptive evidence for timing conditional on the researched schedule, not a selection-adjusted significance result.'),
      p('The matched one-period learner already reaches later Sharpe 1.645. Q’s continuation term changes 9 of 191 later decisions when evaluated at the same actual states; it adds a smaller increment than the full dynamic-versus-fixed gap. Some feature-centering interventions also improve Sharpe slightly. The clearest interpretation is conditional allocation with a modest incremental sequential benefit, rather than proof that every input or long-horizon planning is essential.'),
      p('The reward changes behavior, not just the reported score. With no variance penalty, 138 of 191 later decisions choose straddles alone, versus 46 at penalty 0.20 and 31 at 1.00. Mean straddle targets are 73.2%, 24.9% and 16.9%. The zero-penalty policy loses $331,255 above cash in 2025; penalty 1.00 earns $75,335. Fixed 75% straddles still achieves later Sharpe 1.084, so timing as well as exposure contributes to the zero-penalty failure.'),
      p('Increasing Bellman updates from 200 to 500, 1,000 and 2,000 leaves every later allocation decision and result unchanged for all four settings. The weak zero-penalty result is therefore not explained by an insufficient iteration budget in this check.'),
      p('The extension separates signal profitability from capital allocation. Modest positive penalties retain strong Sharpe across both periods, while the profit-only objective transfers less reliably. Shared integer-contract replay and execution sensitivity are the next checks.')
    ])
    metadata=dict(original_pages=original_pages,extension_pages=len(pages),
                  original_sources={str(p):digest(p) for p in (ORIGINAL_PDF,ORIGINAL_TEX)},
                  source_reports={str(p):digest(p) for p in [REPORT/'q_switch_conditional_results.csv',REPORT/'q_protocol.json',REPORT/'bandit_protocol.json',Path('reports/signal_allocation_discount/results.csv'),Path('reports/signal_allocation_variance/results.csv'),Path('reports/signal_allocation_variance/decisions.csv')]},
                  original_preserved='The first original_pages PDF pages are copied intact, not redrawn. The original paper text is not rewritten. The extension follows the original paper and appendices.',
                  figures=CAPTIONS)
    (DEST/'manuscript.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False)+'\n')
    write_tex(pages)
    (DEST/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (DEST/'README.txt').write_text('''ORIGINAL PAPER WITH A FOCUSED RL EXTENSION

Read options_relative_value_extended.pdf. The original submitted paper appears first with its PDF pages preserved exactly. Six extension pages follow; the rewritten fourteen-page draft is not used as the base.

The extension covers the second volatility signal, entry-bandit interpretation, the Q capital allocator, fixed comparisons, and why its switching helps. research_extension.docx is an editable copy of the new extension only; it is not a replacement Word version of the original paper.

The original PDF and LaTeX are archived in original/. All original source files remain unchanged. provenance.json and verification.json document page and text preservation. No models were refitted or selected.

Reproduce with scripts.build_original_options_extension, running prepare, render --page N (one fresh guarded worker per extension page), assemble, and docx. Use the existing 20% single-core /256MiB memory and swap-growth caps. No user apps need to be closed.
''')
    print(f'Preserved {original_pages} original pages; prepared {len(pages)} extension pages',flush=True)


def render(page_number):
    if page_number is None:raise ValueError('Render exactly one page with --page N')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scripts import build_research_publication as pub
    pages=json.loads((DEST/'manuscript.json').read_text());meta=json.loads((DEST/'provenance.json').read_text())
    pub.FIG=DEST/'figures';pub.CHARTS={k:dict(caption=v) for k,v in CAPTIONS.items()};pub.PAPER_FIGURES={k:'E'+str(i+1) for i,k in enumerate(CAPTIONS)}
    class ExtensionPage(pub.PaperPage):
        def text(self,x,y,text,size=10.35,color='#202832',family='serif',weight='normal'):
            return self.fig.text(x/612,y/792,text,fontsize=size,color=color,family=family,weight=weight,va='top',parse_math=False)
    page=pages[page_number-1];canvas=ExtensionPage(meta['original_pages']+page_number,page['title'])
    for obj in canvas.fig.texts:
        if obj.get_text()=='RESEARCH DRAFT · 4 OCTOBER 2026':obj.set_text('FOLLOW-UP EXTENSION · 15 OCTOBER 2026')
    for block in page['blocks']:
        if block['type']=='paragraph':canvas.paragraph(block['text'])
        elif block['type']=='heading':canvas.heading(block['text'])
        elif block['type']=='chart':canvas.image(block['name'],block['height'])
        elif block['type']=='table':canvas.table(block)
    check=dict(extension_page=page_number,pdf_page=meta['original_pages']+page_number,bottom_content_points=canvas.y,overflow=canvas.y<49)
    folder=DEST/'rendered_pages'
    with PdfPages(folder/f'extension_{page_number:02d}.pdf') as pdf:pdf.savefig(canvas.fig)
    canvas.fig.savefig(folder/f'extension_{page_number:02d}.png',dpi=90);plt.close(canvas.fig);del canvas;gc.collect()
    (folder/f'layout_{page_number:02d}.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check),flush=True)
    if check['overflow']:raise ValueError('Extension page overflow')


def assemble():
    from pypdf import PdfReader,PdfWriter
    meta=json.loads((DEST/'provenance.json').read_text());original=PdfReader(ORIGINAL_PDF)
    for path,signature in meta['original_sources'].items():assert digest(path)==signature
    for path,signature in meta['source_reports'].items():assert digest(path)==signature
    checks=[json.loads((DEST/'rendered_pages'/f'layout_{i:02d}.json').read_text()) for i in range(1,meta['extension_pages']+1)]
    assert not any(c['overflow'] for c in checks)
    # Clone the entire original catalog, including its named link targets.
    # Page-only merging can retain the visible pages but drop internal links.
    writer=PdfWriter(clone_from=original);writer.add_outline_item('Original paper (preserved)',0)
    for i in range(1,meta['extension_pages']+1):
        writer.append(PdfReader(DEST/'rendered_pages'/f'extension_{i:02d}.pdf'),outline_item='Research extension' if i==1 else None)
    writer.add_metadata({'/Title':'From Equity Pairs to Options Relative Value — Original Paper with Research Extension','/Author':'Adam Kreitzman and Soo Xin Yang'})
    path=DEST/'options_relative_value_extended.pdf'
    with path.open('wb') as out:writer.write(out)
    writer.close();combined=PdfReader(path)
    assert len(combined.pages)==meta['original_pages']+meta['extension_pages']
    # Verify the original page content and resource streams after PDF merging.
    def streams(obj):
        if hasattr(obj,'get_object'):obj=obj.get_object()
        if hasattr(obj,'get_data'):return [obj.get_data()]
        return sum([streams(x) for x in obj],[])
    for i,page in enumerate(original.pages):
        assert page.extract_text()==combined.pages[i].extract_text()
        assert streams(page['/Contents'])==streams(combined.pages[i]['/Contents'])
        assert list(page.mediabox)==list(combined.pages[i].mediabox)
        old_annotations=page.get('/Annots',[])
        if hasattr(old_annotations,'get_object'):old_annotations=old_annotations.get_object()
        new_annotations=combined.pages[i].get('/Annots',[])
        if hasattr(new_annotations,'get_object'):new_annotations=new_annotations.get_object()
        assert len(old_annotations)==len(new_annotations)
    for name,destination in original.named_destinations.items():
        assert name in combined.named_destinations
        assert original.get_destination_page_number(destination)==combined.get_destination_page_number(combined.named_destinations[name])
    verification=dict(original_pages_preserved=meta['original_pages'],extension_pages=meta['extension_pages'],
                      total_pages=len(combined.pages),original_text_unchanged=True,original_page_content_streams_unchanged=True,
                      original_sources_unchanged=True,original_annotations_and_named_destinations_preserved=True,
                      layout_checks=checks,original_pdf_sha256=digest(ORIGINAL_PDF),output_sha256=digest(path))
    (DEST/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='layout_checks'}),flush=True)


def docx():
    from scripts import export_research_docx as word
    pages=json.loads((DEST/'manuscript.json').read_text())
    def verify(path,pages,charts,numbers):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            document=word.ET.fromstring(z.read('word/document.xml'))
            paragraphs=[''.join(p.itertext()) for p in document.findall('.//w:p',word.NS)]
            for page in pages:
                assert page['title'] in paragraphs
                for b in page['blocks']:
                    if b['type'] in ('paragraph','heading'):assert b['text'] in paragraphs
                    elif b['type']=='chart':assert z.read(f'word/media/{b["name"]}.png')==(DEST/'figures'/f'{b["name"]}.png').read_bytes()
                    elif b['type']=='table':
                        for row in [b['headers']]+b['rows']:assert all(str(v) in paragraphs for v in row)
            blocks=[b for p in pages for b in p['blocks']]
            assert len(document.findall('.//w:tbl',word.NS))==sum(b['type']=='table' for b in blocks)
            assert len(document.findall('.//w:drawing',word.NS))==sum(b['type']=='chart' for b in blocks)
            assert len(document.findall('.//w:pageBreakBefore',word.NS))==len(pages)-1
    word.verify=verify;path=DEST/'research_extension.docx'
    word.export(pages,{k:dict(caption=v) for k,v in CAPTIONS.items()},
                {k:'E'+str(i+1) for i,k in enumerate(CAPTIONS)},path)
    with zipfile.ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
    parts['word/header1.xml']=parts['word/header1.xml'].replace(b'6 OCTOBER 2026',b'RESEARCH EXTENSION - 15 OCTOBER 2026')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    print('Saved editable extension; original paper not rewritten',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['prepare','render','assemble','docx']);p.add_argument('--page',type=int);a=p.parse_args()
    if a.phase=='render':render(a.page)
    else:globals()[a.phase]()
