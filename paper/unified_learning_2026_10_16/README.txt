UNIFIED PAPER: LEARNING EXPOSURE WITHIN AND ACROSS OPTIONS STRATEGIES

options_exposure_learning.pdf is the twenty-page main paper.
options_exposure_learning_with_appendix.pdf contains the same main paper plus six technical appendix pages.
options_exposure_learning.docx is an editable copy of the complete manuscript.
options_exposure_learning.tex and manuscript.json preserve the same text, tables and figures.

The original ten-page paper and original-with-extension version are unchanged. This integrated revision restores the original economic motivation, paper discussion, worked vega example, actual spread illustration, feature exploration, training timeline and detailed tree/CEM figure. It adds the straddle signal and Q allocation as extensions of that foundation. Entry learning and allocation receive three dedicated pages each, with a separate four-way comparison. The previous integrated version is archived locally in ../unified_learning_2026_10_16_before_figure_revision.

To build the figures on another checkout, extract reproduction_inputs.zip into the repository root first. It supplies four derived daily portfolio ledgers and one saved context model, verified against provenance.json; it contains no raw option chains or credentials. Other required reports and the signal illustration are committed alongside the paper. This compact archive supports document reproduction, not a complete refit of the underlying strategies.

Section 18 adds conventional allocation benchmarks adapted from Caner et al. (2026): minimum variance, maximum Sharpe and mean–variance objectives. Sample and linear Ledoit-Wolf covariance estimators are compared using validation only within each objective family. All cases and the zero-penalty exception are retained in reports/paper_allocation_benchmarks. Same-schedule fee diagnostics distinguish allocation performance from incremental resizing charges. Realized volatility is reported alongside profit; policies do not have matched risk exposure.

Sections 6–8 integrate the teammate's straddle economics: full-history option/hedge attribution, IV versus subsequent RV and two actual trade paths. Section 19 and Appendix F retain the separate joint-contract risk-cut trial, including uncertainty, exposure differences and reproduction limits. None replaces the original bandit or Q comparisons. The original 11 figures are preserved. Frozen inputs are in reports/strategy_economics; their hashes are checked during preparation.

An independent editable Overleaf project is in ../overleaf_options_exposure; ../options_exposure_overleaf.zip is ready for Overleaf New Project > Upload Project. It includes local vector figures, sections, appendices and a BibTeX database. No Overleaf account connection or automatic synchronization is configured.

Reproduce using scripts.build_unified_options_paper under scripts.run_signal_discovery_guarded. Run prepare; figure --figure NAME separately for each name in provenance.json; render --page N in a fresh worker for each of 26 pages; assemble; docx. scripts.rebuild_options_paper_guarded automates the figure, page and finalization stages sequentially. Figures are implemented in scripts.options_paper_figures at their printed width with 170-dpi PNGs and vector PDF/SVG exports. Preserve one worker, 20% of one core, 256 MiB worker and swap-growth caps. No model fitting is performed.

The PDF uses traditional academic typography: the original one-line title, centered authors and subtitle, black serif headings, plain ruled tables and centered page numbers. Charts are embedded as vectors; PNGs support previews and the editable Word export. scripts.options_paper_layout defines the matching PDF and Word style. LaTeX and Word may paginate differently in other software. Source report hashes, original-version preservation checks and layout checks are saved alongside the paper. All Sharpe ratios are annualized excess Sharpe; 2025 is partial. Data licensing remains applicable.
