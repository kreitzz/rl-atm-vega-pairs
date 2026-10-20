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
KEYS=['ning2024','goyal2009','deboer2005','ernst2005','caner2026']

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
    pattern=r'(\[[1-5]\]|(?:reports|artifacts|paper|scripts|data|config)/[A-Za-z0-9_./-]+)'
    for token in re.split(pattern,str(text)):
        if re.fullmatch(r'\[[1-5]\]',token):chunks.append(r'\cite{'+KEYS[int(token[1])-1]+'}')
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
    parts=[r'\begin{center}\small',r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.18}',
           r'\begin{tabularx}{\textwidth}{@{}'+''.join(spec)+r'@{}}\toprule',
           ' & '.join(escape(x) for x in block['headers'])+r'\\\midrule']
    parts.extend(' & '.join(escape(x) for x in row)+r'\\' for row in block['rows'])
    return '\n'.join(parts+[r'\bottomrule\end{tabularx}\end{center}'])


def export():
    from scripts.build_unified_options_paper import CAPTIONS,MAIN_PAGES
    from scripts.options_paper_layout import TITLE,SUBTITLE,AUTHORS,DATE
    pages=json.loads((SOURCE/'manuscript.json').read_text())
    for folder in ['sections','appendices','figures']:(DEST/folder).mkdir(parents=True,exist_ok=True)
    inputs=[];appendices=[];abstract=[];figures=[];block_count=0
    for index,page in enumerate(pages):
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
        content=[r'\section{'+escape(title)+r'}\label{'+label+'}']
        for b in page['blocks']:
            block_count+=1;kind=b['type']
            if kind=='reference':continue
            if kind=='heading' and b['text'] in ['Abstract','References']:continue
            if index==0 and kind=='paragraph' and not abstract:
                abstract.append(escape(b['text']));continue
            if index==0 and kind=='heading' and b['text'].startswith('1.'):continue
            if kind=='paragraph':content.append(escape(b['text'])+'\n')
            elif kind=='heading':content.append(r'\subsection*{'+escape(b['text'])+'}')
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
        content.append(r'\FloatBarrier')
        (DEST/filename).write_text('\n\n'.join(content)+'\n')
        (appendices if is_appendix else inputs).append(r'\input{'+filename[:-4]+'}')
    (DEST/'references.bib').write_text(BIB)
    preamble=r'''\documentclass[11pt,letterpaper]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype,amsmath,booktabs,tabularx,array,graphicx,placeins,float}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\setlength{\emergencystretch}{2em}
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\intextsep}{12pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{.9}
\renewcommand{\bottomfraction}{.8}
\renewcommand{\textfraction}{.1}
\renewcommand{\floatpagefraction}{.8}
\hypersetup{pdftitle={From Equity Pairs to Options Relative Value},
 pdfauthor={Adam Kreitzman and Soo Xin Yang}}
\begin{document}
\begin{center}
\makebox[\textwidth][c]{{\fontsize{20}{24}\selectfont\bfseries TITLE}}
\vspace{0.6em}

{\large SUBTITLE}\par
\vspace{1em}
AUTHORS\par
\small MFE 230Z: Deep Learning in Finance\par
DATE
\end{center}
\begin{abstract}
ABSTRACT
\end{abstract}
'''
    for key,value in [('SUBTITLE',SUBTITLE),('TITLE',TITLE),('AUTHORS',AUTHORS),('DATE',DATE),('ABSTRACT',abstract[0])]:
        preamble=preamble.replace(key,value if key=='ABSTRACT' else escape(value))
    main=preamble+'\n'.join(inputs)+r'''

\bibliographystyle{unsrtnat}
\bibliography{references}
\clearpage
\appendix
'''+ '\n'.join(appendices)+'\n'+r'\end{document}'+'\n'
    (DEST/'main.tex').write_text(main)
    (DEST/'README.txt').write_text('''OVERLEAF PROJECT — FROM EQUITY PAIRS TO OPTIONS RELATIVE VALUE

Upload ../options_exposure_overleaf.zip through Overleaf: New Project > Upload Project.
Set main.tex as the main document and use pdfLaTeX (current TeX Live).
The default compile runs BibTeX automatically. No Python, data access or raw
option files are needed. All 16 figures are included as vector PDFs.

Edit main.tex for formatting, sections/*.tex for the main paper,
appendices/*.tex for supporting detail and references.bib for citations.
The source is actual editable LaTeX, not images of manuscript pages.
Sections may paginate differently from the fixed-layout PDF/DOCX edition.

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
    assert len(re.findall(r'\\caption\{',all_tex))==16
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
    command=[str(Path(executable).resolve()),'--keep-logs','--outdir',str(output),'main.tex']
    subprocess.run(command,cwd=DEST,check=True)
    print('Compiled Overleaf project: '+str(output/'main.pdf'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['export','compile'],required=True)
    parser.add_argument('--tectonic',default='/tmp/options-tectonic/tectonic')
    args=parser.parse_args()
    if args.phase=='export':export()
    else:compile_project(args.tectonic)
