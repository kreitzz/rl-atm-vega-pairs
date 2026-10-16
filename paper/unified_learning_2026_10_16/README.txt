UNIFIED PAPER: LEARNING EXPOSURE WITHIN AND ACROSS OPTIONS STRATEGIES

options_exposure_learning.pdf is the fifteen-page main paper.
options_exposure_learning_with_appendix.pdf contains the same main paper plus five technical appendix pages.
options_exposure_learning.docx is an editable copy of the complete manuscript.
options_exposure_learning.tex and manuscript.json preserve the same text, tables and figures.

The original ten-page paper and original-with-extension version are unchanged. This integrated revision restores the original economic motivation, paper discussion, worked vega example, actual spread illustration, feature exploration, training timeline and detailed tree/CEM figure. It adds the straddle signal and Q allocation as extensions of that foundation. Entry learning and allocation receive three dedicated pages each, with a separate four-way comparison. The previous integrated version is archived locally in ../unified_learning_2026_10_16_before_figure_revision.

To build the figures on another checkout, extract reproduction_inputs.zip into the repository root first. It supplies four derived daily portfolio ledgers and one saved context model, verified against provenance.json; it contains no raw option chains or credentials. Other required reports and the signal illustration are committed alongside the paper. This compact archive supports document reproduction, not a complete refit of the underlying strategies.

Reproduce using scripts.build_unified_options_paper under scripts.run_signal_discovery_guarded. Run prepare; figure --figure NAME separately for each name in provenance.json; render --page N in a fresh worker for each of 20 pages; assemble; docx. scripts.rebuild_options_paper_guarded automates the figure, page and finalization stages sequentially. Figures are implemented in scripts.options_paper_figures at their printed width with 170-dpi PNGs and vector PDF/SVG exports. Preserve one worker, 20% of one core, 256 MiB worker and swap-growth caps. No model fitting is performed.

The PDF is rendered from the shared manuscript using the existing publication renderer. LaTeX and Word are editable exports and may paginate differently in other software. Source report hashes, original-version preservation checks and layout checks are saved alongside the paper. All Sharpe ratios are annualized excess Sharpe; 2025 is partial. Data licensing remains applicable.
