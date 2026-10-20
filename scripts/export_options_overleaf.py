"""Export the integrated manuscript as a self-contained editable Overleaf project.

No raw data or model fitting is required. Run --phase export after building the
paper figures. Optional --phase compile uses a supplied Tectonic executable;
run that phase through run_signal_discovery_guarded on the research Mac.
The ZIP excludes build outputs and can be uploaded with Overleaf's Upload Project.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

SOURCE=Path('paper/unified_learning_2026_10_16')
DEST=Path('paper/overleaf_options_exposure')
ARCHIVE=Path('paper/options_exposure_overleaf.zip')
KEYS=['ning2024','goyal2009','deboer2005','ernst2005','caner2026','alphaglass2026']

BIB=r'''@misc{ning2024,
  author={Ning, B. and Lee, K.},
  title={Advanced Statistical Arbitrage with Reinforcement Learning},
  year={2024}, eprint={2403.12180}, archivePrefix={arXiv},
  url={https://arxiv.org/abs/2403.12180}
}
@article{goyal2009,
  author={Goyal, Amit and Saretto, Alessio},
  title={Cross-section of option returns and volatility},
  journal={Journal of Financial Economics}, year={2009},
  volume={94}, pages={310--326}, doi={10.1016/j.jfineco.2009.01.001}
}
@article{deboer2005,
  author={de Boer, Pieter-Tjerk and Kroese, Dirk P. and Mannor, Shie and Rubinstein, Reuven Y.},
  title={A Tutorial on the Cross-Entropy Method},
  journal={Annals of Operations Research}, year={2005}, volume={134}, pages={19--67},
  url={https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf}
}
@article{ernst2005,
  author={Ernst, Damien and Geurts, Pierre and Wehenkel, Louis},
  title={Tree-Based Batch Mode Reinforcement Learning},
  journal={Journal of Machine Learning Research}, year={2005}, volume={6}, pages={503--556},
  url={https://jmlr.org/papers/v6/ernst05a.html}
}
@misc{caner2026,
  author={Caner, M. and Capponi, A. and Sun, N. and Tan, J. Y.},
  title={Designing Agentic {AI}-Based Screening for Portfolio Investment},
  year={2026}, eprint={2603.23300}, archivePrefix={arXiv},
  url={https://arxiv.org/abs/2603.23300}
}
@techreport{alphaglass2026,
  author={Bell, Sebastian and Kakhbod, Ali and Lettau, Martin and Nazemi, Abdolreza},
  title={{AlphaGlass}: Interpretable Characteristic-Based Portfolio Choice},
  institution={National Bureau of Economic Research},
  type={Working Paper}, number={35186}, year={2026}, month={May},
  doi={10.3386/w35186}, url={https://www.nber.org/papers/w35186}
}
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def escape(text):
    """Escape prose, retaining editable math, references and breakable repo paths."""
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
             '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
             'λ':r'\(\lambda\)','γ':r'\(\gamma\)','σ':r'\(\sigma\)',
             'Δ':r'\(\Delta\)','Γ':r'\(\Gamma\)','Θ':r'\(\Theta\)','Σ':r'\(\Sigma\)',
             '²':r'\(^{2}\)','₁':r'\(_1\)','₂':r'\(_2\)','𝑤':r'\(w\)',
             '±':r'\(\pm\)','√':r'\(\surd\)','≈':r'\(\approx\)','≥':r'\(\geq\)',
             '→':r'\(\to\)','∈':r'\(\in\)','½':r'\(\tfrac{1}{2}\)',
             '·':r'\(\cdot\)','−':'-','–':'--','—':'---','’':"'",'“':'``','”':"''"}
    def plain(value):return ''.join(mapping.get(c,c) for c in value)
    chunks=[]
    # Every other chunk is a citation or repository path. Avoid escaping the
    # backslashes of LaTeX generated in a preceding replacement pass.
    pattern=r'(\[[1-6]\]|(?:reports|artifacts|paper|scripts|data|config)/[A-Za-z0-9_./-]+)'
    for token in re.split(pattern,str(text)):
        if re.fullmatch(r'\[[1-6]\]',token):chunks.append(r'\cite{'+KEYS[int(token[1])-1]+'}')
        elif re.match(r'^(reports|artifacts|paper|scripts|data|config)/',token):chunks.append(r'\path{'+token+'}')
        else:chunks.append(plain(token))
    result=''.join(chunks)
    result=re.sub(r'Appendix ([A-F])',lambda m:r'Appendix~\ref{app:'+m[1].lower()+'}',result)
    result=result.replace('in Section 16',r'in Section~\ref{sec:16}')
    assert all(ord(c)<128 for c in result),result
    return result


def table_tex(block):
    cols=len(block['headers']);weights=block.get('widths') or [1]*cols
    numeric=[False]+[sum(bool(re.match(r'^[+$−\-]?\d',str(row[i]))) for row in block['rows'])>len(block['rows'])/2 for i in range(1,cols)]
    spec=[]
    for w,is_numeric in zip(weights,numeric):
        align=r'\raggedleft' if is_numeric else r'\raggedright'
        spec.append(r'>{\hsize='+f'{cols*w/sum(weights):.5f}'+r'\hsize\linewidth=\hsize'+align+r'\arraybackslash}X')
    parts=[r'\begin{table}[H]\centering\small']
    if block.get('caption'):parts.append(r'\caption{'+escape(block['caption'])+'}')
    parts.extend([r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.18}',
           r'\begin{tabularx}{\textwidth}{@{}'+''.join(spec)+r'@{}}\toprule',
           ' & '.join(r'\textbf{'+escape(x)+'}' for x in block['headers'])+r'\\\midrule'])
    parts.extend(' & '.join(escape(x) for x in row)+r'\\' for row in block['rows'])
    return '\n'.join(parts+[r'\bottomrule\end{tabularx}\end{table}'])


def export():
    from scripts.build_unified_options_paper import CAPTIONS,MAIN_PAGES
    from scripts.options_paper_layout import TITLE,SUBTITLE,AUTHORS,DATE
    pages=json.loads((SOURCE/'manuscript.json').read_text())
    for folder in ['sections','appendices','figures']:(DEST/folder).mkdir(parents=True,exist_ok=True)
    inputs=[];appendices=[];abstract=[];figures=[];block_count=0
    for index,page in enumerate(pages):
        if page['title']=='References':
            block_count+=len(page['blocks'])
            continue
        is_appendix=index>=MAIN_PAGES
        if is_appendix:
            letter=page['title'].split()[1].rstrip('.').lower()
            title=page['title'].split('. ',1)[1]
            label='app:'+letter;folder='appendices';prefix=letter
        else:
            title=page['title'].split('. ',1)[1] if index else 'From a useful signal to an exposure decision'
            label=f'sec:{index+1}';folder='sections';prefix=f'{index+1:02d}'
        slug=re.sub('[^a-z0-9]+','_',title.lower()).strip('_')[:65]
        filename=f'{folder}/{prefix}_{slug}.tex'
        content=([r'\clearpage'] if is_appendix else [])+[r'\section{'+escape(title)+r'}\label{'+label+'}']
        for b in page['blocks']:
            block_count+=1;kind=b['type']
            if kind=='reference':continue
            if kind=='heading' and b['text'] in ['Abstract','References']:continue
            if index==0 and kind=='paragraph' and not abstract:
                abstract.append(escape(b['text']));continue
            if index==0 and kind=='heading' and b['text'].startswith('1.'):continue
            if kind=='paragraph':content.append(escape(b['text'])+'\n')
            elif kind=='heading':
                if b['text']=='Conclusion':content.append(r'\Needspace{15\baselineskip}')
                content.append(r'\subsection*{'+escape(b['text'])+'}')
            elif kind=='equation':content.append(r'\begin{equation*}'+b['latex']+r'\end{equation*}')
            elif kind=='table':content.append(table_tex(b))
            elif kind=='chart':
                name=b['name'];figures.append(name)
                shutil.copy2(SOURCE/'figures'/f'{name}.pdf',DEST/'figures'/f'{name}.pdf')
                placement='H' if name=='feature_validation' else '!htbp'
                content.append(r'\begin{figure}['+placement+r']\centering'+
                    '\n'+r'\includegraphics[width=\linewidth]{'+name+r'.pdf}'+
                    '\n'+r'\caption{'+escape(CAPTIONS[name])+r'}\label{fig:'+name.replace('_','-')+'}'+
                    '\n'+r'\end{figure}')
        # Avoid flushing a half-empty float page at every short source section.
        # Keep figures within the corresponding economic/methodological block.
        if is_appendix or index in {3,7,9,12,15,18,19}:
            content.append(r'\FloatBarrier')
        (DEST/filename).write_text('\n\n'.join(content).rstrip()+'\n')
        (appendices if is_appendix else inputs).append(r'\input{'+filename[:-4]+'}')
    (DEST/'references.bib').write_text(BIB)
    preamble=r'''\documentclass[11pt,letterpaper]{article}
\usepackage[left=1in,right=1in,top=0.9in,bottom=0.9in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype,amsmath,booktabs,tabularx,array,graphicx,placeins,float}
\usepackage{titlesec,needspace}
\usepackage[font=small,labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false]{caption}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\linespread{1.04}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.3em plus 0.05em minus 0.02em}
\setlength{\emergencystretch}{2em}
\titleformat{\section}{\large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.7em}{}
\titlespacing*{\section}{0pt}{1.3\baselineskip}{0.55\baselineskip}
\titlespacing*{\subsection}{0pt}{0.9\baselineskip}{0.35\baselineskip}
\captionsetup{skip=7pt}
\captionsetup[table]{position=top,skip=6pt}
\setlength{\abovedisplayskip}{9pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{9pt plus 2pt minus 2pt}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
\raggedbottom
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\intextsep}{12pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{.9}
\renewcommand{\bottomfraction}{.8}
\renewcommand{\textfraction}{.1}
\renewcommand{\floatpagefraction}{.8}
\hypersetup{pdftitle={TITLE},
 pdfauthor={Adam Kreitzman and Soo Xin Yang}}
\begin{document}
\begin{center}
\makebox[\textwidth][c]{{\fontsize{19}{23}\selectfont\bfseries TITLE}}
\vspace{0.6em}

{\fontsize{11}{14}\selectfont SUBTITLE}\par
\vspace{1em}
AUTHORS\par
{\small DATE}
\end{center}
\begin{center}
\begin{minipage}{0.88\textwidth}
\small
\begin{center}\bfseries Abstract\end{center}
\setlength{\parindent}{0pt}\noindent
ABSTRACT
\end{minipage}
\end{center}
'''
    for key,value in [('SUBTITLE',SUBTITLE),('TITLE',TITLE),('AUTHORS',AUTHORS),('DATE',DATE),('ABSTRACT',abstract[0])]:
        preamble=preamble.replace(key,value if key=='ABSTRACT' else escape(value))
    main=preamble+'\n'.join(inputs)+r'''

\bibliographystyle{unsrtnat}
\bibliography{references}
\clearpage
\appendix
\begingroup
\renewcommand{\baselinestretch}{1}
\fontsize{10.5}{12.8}\selectfont
'''+ '\n'.join(appendices)+'\n'+r'\endgroup'+'\n'+r'\end{document}'+'\n'
    (DEST/'main.tex').write_text(main)
    (DEST/'README.txt').write_text(f'''OVERLEAF PROJECT — {TITLE.upper()}

Upload ../options_exposure_overleaf.zip through Overleaf: New Project > Upload Project.
Set main.tex as the main document and use pdfLaTeX (current TeX Live).
The default compile runs BibTeX automatically. No Python, data access or raw
option files are needed. All 16 figures are included as vector PDFs.

Edit main.tex for formatting, sections/*.tex for the main paper,
appendices/*.tex for supporting detail and references.bib for citations.
The source is actual editable LaTeX, not images of manuscript pages.
The primary repository PDF is typeset from this LaTeX project. The Word copy
shares the same manuscript content and may paginate differently.

This export synthesizes the original exposure-learning study with the teammate's
straddle economics and separate management trial. Source metrics and limits are
unchanged. The trial is a risk rule in a separate joint-contract account; it does
not turn the main Q experiment into contract-level management.

Exported by scripts/export_options_overleaf.py from the shared manuscript source.
If the paper is edited in Overleaf, download that version before re-exporting;
there is no automatic bidirectional sync and a fresh export replaces local TeX.
Build files and the compiled PDF are deliberately excluded from the upload ZIP.
Data licensing and the original repository's academic-use conditions still apply.
''')
    # Reject missing figures, unresolved source dependencies or lost text blocks.
    assert len(figures)==len(CAPTIONS)==16 and len(set(figures))==16
    assert len(inputs)==20 and len(appendices)==6
    all_tex='\n'.join(p.read_text() for p in DEST.rglob('*.tex'))
    assert 'paper/unified_learning' not in all_tex
    citations=set(re.findall(r'\\cite\{([^}]+)\}',all_tex))
    assert citations==set(KEYS),citations
    assert len(re.findall(r'\\caption\{',all_tex))==33
    manifest=dict(manuscript_sha256=sha(SOURCE/'manuscript.json'),
        sections=len(inputs),appendices=len(appendices),figures=figures,
        all_citation_keys_resolved=True,source_blocks=block_count,
        figure_hashes={n:sha(DEST/'figures'/f'{n}.pdf') for n in figures},
        scope='Editable publication export; no new simulation or model fitting.')
    (DEST/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files=[DEST/'main.tex',DEST/'references.bib',DEST/'README.txt',DEST/'source_manifest.json',
           *sorted((DEST/'sections').glob('*.tex')),*sorted((DEST/'appendices').glob('*.tex')),
           *[DEST/'figures'/f'{n}.pdf' for n in figures]]
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.relative_to(DEST))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert 'main.tex' in z.namelist() and 'main.pdf' not in z.namelist()
    print(f'Exported {len(files)} files, 26 sections/appendices and 16 vector figures to {ARCHIVE}',flush=True)


def compile_project(executable):
    output=Path('/tmp/options_overleaf_compile');output.mkdir(exist_ok=True)
    command=[str(Path(executable).resolve()),'--keep-logs','--keep-intermediates','--outdir',str(output),'main.tex']
    with (output/'compiler_console.log').open('w') as console:
        result=subprocess.run(command,cwd=DEST,stdout=console,stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join((output/'compiler_console.log').read_text().splitlines()[-40:]),flush=True)
        result.check_returncode()
    with zipfile.ZipFile(ARCHIVE) as archive:
        names=archive.namelist()
        assert all(archive.read(name)==(DEST/name).read_bytes() for name in names)
    record=dict(source_files={name:sha(DEST/name) for name in names},
                archive_sha256=sha(ARCHIVE),pdf_sha256=sha(output/'main.pdf'),
                main_source_sha256=sha(DEST/'main.tex'))
    (output/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Compiled Overleaf project: '+str(output/'main.pdf'),flush=True)


def publish(compiled_pdf,auxiliary):
    """Promote the verified LaTeX PDF to the stable repository paper filenames.

    The JSON pages remain editorial content units, not physical typeset pages.
    Appendix A's LaTeX label determines the actual PDF split. This replaces
    fixed-page plot-based document rendering while retaining the same results.
    """
    import unicodedata
    from pypdf import PdfReader,PdfWriter
    from scripts.build_unified_options_paper import CAPTIONS
    from scripts.options_paper_layout import TITLE,SUBTITLE,AUTHORS
    pdf=Path(compiled_pdf);aux=Path(auxiliary)
    build=json.loads((pdf.parent/'build_manifest.json').read_text())
    assert build['pdf_sha256']==sha(pdf),'Compiled PDF changed'
    assert build['archive_sha256']==sha(ARCHIVE),'Source ZIP changed after compilation'
    for name,expected in build['source_files'].items():assert sha(DEST/name)==expected,name
    meta=json.loads((SOURCE/'provenance.json').read_text())
    for path,expected in {**meta['source_hashes'],**meta['preserved_versions']}.items():assert sha(path)==expected,path
    log=(pdf.parent/'main.log').read_text()
    assert not re.search(r'Overfull|There were undefined|Missing character|Citation .+ undefined',log)
    auxiliary_text=aux.read_text()
    assert all(re.search(r'\\bibcite\{'+re.escape(key)+r'\}',auxiliary_text) for key in KEYS)
    label=re.search(r'\\newlabel\{app:a\}\{\{[^{}]*\}\{(\d+)\}',auxiliary_text)
    assert label,'Missing Appendix A page label'
    first_appendix=int(label[1]);reader=PdfReader(pdf)
    text='\n'.join(p.extract_text() for p in reader.pages)
    compact=lambda value:re.sub(r'\s+','',unicodedata.normalize('NFKC',str(value))).replace('−','-').replace('–','-')
    all_text=compact(text);pages=json.loads((SOURCE/'manuscript.json').read_text())
    tables=[b for p in pages for b in p['blocks'] if b['type']=='table']
    for b in tables:
        for row in b['rows']:
            for value in row:assert compact(value) in all_text,value
    for value in ['AlphaGlass','35186','1.726','2.152','1.263','1.041','1.576','2,599']:
        assert value in text,value
    semantic=re.sub(r'(?<=\w)-\s*\n\s*(?=\w)','',text)
    assert 'withoutvalidating' in compact(semantic) or 'doesnotvalidate' in compact(semantic)
    fonts=set()
    for page in reader.pages:
        resources=page.get('/Resources')
        if resources:
            for font in resources.get_object().get('/Font',{}).get_object().values():
                fonts.add(str(font.get_object().get('/BaseFont','')))
    assert any('LMRoman' in f or 'lmr' in f.lower() for f in fonts),fonts
    assert 1<first_appendix<=len(reader.pages)
    # Append copies the matching navigation outlines as well as page content.
    for filename,bounds in [('options_exposure_learning.pdf',(0,first_appendix-1)),
                            ('technical_appendix.pdf',(first_appendix-1,len(reader.pages)))]:
        writer=PdfWriter();writer.append(str(pdf),pages=bounds)
        writer.add_metadata({'/Title':TITLE,'/Author':AUTHORS})
        with (SOURCE/filename).open('wb') as f:writer.write(f)
    shutil.copy2(pdf,SOURCE/'options_exposure_learning_with_appendix.pdf')
    figures=json.loads((DEST/'source_manifest.json').read_text())['figure_hashes']
    for name,expected in figures.items():assert sha(SOURCE/'figures'/f'{name}.pdf')==expected,name
    record=dict(document_engine='LaTeX / Tectonic 0.17.0 (XeTeX)',
        typography='Latin Modern serif text and matching mathematics; conventional academic article layout',
        main_pages=first_appendix-1,appendix_pages=len(reader.pages)-first_appendix+1,
        total_pages=len(reader.pages),main_editorial_units=meta['main_pages'],appendix_editorial_units=meta['appendix_pages'],
        all_table_values_in_pdf=True,numeric_tables=len(tables),source_hashes_verified=True,
        original_versions_unchanged=True,vector_figures_embedded=len(CAPTIONS),
        bibliography_keys=KEYS,all_citations_and_references_resolved=True,
        no_overfull_boxes_or_missing_characters=True,management_fitting_not_validation_explicit=True,
        font_resources=sorted(fonts),compiled_source_manifest=build,
        figure_hashes={str(SOURCE/'figures'/f'{name}.{ext}'):sha(SOURCE/'figures'/f'{name}.{ext}') for name in CAPTIONS for ext in ('png','pdf','svg')},
        output_hashes={str(SOURCE/name):sha(SOURCE/name) for name in ['options_exposure_learning.pdf','technical_appendix.pdf','options_exposure_learning_with_appendix.pdf']})
    (SOURCE/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    (SOURCE/'formatting_verification.json').write_text(json.dumps(dict(
        style='classic-latex-article-v1',title=TITLE,subtitle=SUBTITLE,
        primary_pdf_typeset_from_overleaf_source=True,all_numeric_tables_verified=True,
        vector_figures_embedded=16,main_pages=record['main_pages'],appendix_pages=record['appendix_pages'],
        pdf_sha256=sha(pdf),body_and_math_fonts='Latin Modern',
        alphaglass_cited=True,all_17_numeric_tables_preserved=True,
        paragraph_first_line_indent_pt=0,paragraph_spacing_em=0.3,
        management_fitting_not_validation_explicit=True),indent=2)+'\n')
    (SOURCE/'overleaf_compilation_verification.json').write_text(json.dumps(dict(
        engine=record['document_engine'],compiled=True,compiled_pages=len(reader.pages),
        compiled_pdf_sha256=sha(pdf),upload_zip_sha256=sha(ARCHIVE),main_source_sha256=sha(DEST/'main.tex'),
        all_17_numeric_tables_present=True,all_citations_and_references_resolved=True,
        no_overfull_boxes_or_missing_characters=True,compilation_outputs_excluded_from_zip=True,
        overleaf_account_upload_performed=False,alphaglass_cited=True),indent=2)+'\n')
    economics_path=SOURCE/'economics_integration_verification.json'
    economics=json.loads(economics_path.read_text())
    economics.update(integrated_pdf_sha256=sha(pdf),main_pages=record['main_pages'],appendix_pages=record['appendix_pages'])
    economics_path.write_text(json.dumps(economics,indent=2)+'\n')
    (SOURCE/'README.txt').write_text(f'''PRIMARY PAPER — LATEX EDITION
{TITLE}
{SUBTITLE}

options_exposure_learning_with_appendix.pdf is the full typeset paper.
options_exposure_learning.pdf contains {record['main_pages']} main/reference pages;
technical_appendix.pdf contains the remaining {record['appendix_pages']} appendix pages.
The primary PDFs are compiled from paper/overleaf_options_exposure/main.tex.
Their editable source, six BibTeX references and 16 vector figures are bundled
in paper/options_exposure_overleaf.zip for Overleaf Upload Project.
The Word edition shares the manuscript content and may paginate differently.

The paper retains the original signal, trade, methodology and learning-layer
explanations, plus the teammate's straddle attribution and separate management
trial. AlphaGlass (Bell et al., 2026) is cited for direct investment-objective
optimization and interpretable characteristic-based portfolio choice.
The cuts model is fitted during 2018–2021 without applying management then;
that period does not validate the trimming benefit. All numerical results and
figures are unchanged by this typography/citation revision.

Edit sections/*.tex, appendices/*.tex and references.bib in the Overleaf project.
No direct Overleaf account connection or automatic synchronization is configured.
The local JSON manuscript is still divided into editorial units; those unit
counts do not describe physical LaTeX pages. The older plot-based per-page
preview files are not the typesetting source of the primary PDF.

Reproduction: prepare the shared manuscript with scripts.build_unified_options_paper;
export, compile and publish with scripts.export_options_overleaf. The guarded
driver scripts/rebuild_options_paper_guarded.py --stage latex runs these three
steps sequentially. Tectonic is needed only for local compilation; Overleaf can
compile the ZIP directly. Run Word export separately if needed.
Preserve one worker, 20% of one CPU core and the 256 MiB RSS/swap-growth limits.
No new fitting, strategy selection or raw-data backtest occurs in this build.

verification.json verifies sources, retained results, vector figures, resolved
citations and the compiled-source hashes. Frozen economics data reproduce the
publication rather than independently refitting the management simulator.
Earlier versions, original strategy data restrictions and academic-use terms
remain applicable. See REVISION_NOTES.txt and the source reports for provenance.
''')
    print(f'Published LaTeX paper: {record["main_pages"]} main/reference + {record["appendix_pages"]} appendix pages; {len(tables)} tables and six references verified',flush=True)


def preview(compiled_pdf,page_number):
    """Rasterize one selected PDF page with the macOS native image converter."""
    from pypdf import PdfReader,PdfWriter
    root=Path('/tmp/options_typography_review');root.mkdir(exist_ok=True)
    reader=PdfReader(compiled_pdf);assert 1<=page_number<=len(reader.pages)
    page=root/f'page_{page_number:02d}.pdf';png=page.with_suffix('.png')
    writer=PdfWriter();writer.add_page(reader.pages[page_number-1])
    with page.open('wb') as f:writer.write(f)
    subprocess.run(['sips','-s','format','png','-s','dpiWidth','144','-s','dpiHeight','144',str(page),'--out',str(png)],check=True,capture_output=True)
    from PIL import Image
    with Image.open(png) as image:
        if 'A' in image.getbands():
            white=Image.new('RGB',image.size,'white')
            white.paste(image,mask=image.getchannel('A'));white.save(png)
    print('Preview: '+str(png),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['export','compile','publish','preview'],required=True)
    parser.add_argument('--tectonic',default='/tmp/options-tectonic/tectonic')
    parser.add_argument('--compiled-pdf',default='/tmp/options_overleaf_compile/main.pdf')
    parser.add_argument('--aux',default='/tmp/options_overleaf_compile/main.aux')
    parser.add_argument('--page',type=int,default=1)
    args=parser.parse_args()
    if args.phase=='export':export()
    elif args.phase=='compile':compile_project(args.tectonic)
    elif args.phase=='publish':publish(args.compiled_pdf,args.aux)
    elif args.page==0:
        import gc
        from pypdf import PdfReader
        count=len(PdfReader(args.compiled_pdf).pages)
        for number in range(1,count+1):
            preview(args.compiled_pdf,number)
            gc.collect()
    else:preview(args.compiled_pdf,args.page)
