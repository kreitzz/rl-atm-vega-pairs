PRIMARY PAPER — LATEX EDITION
Learning to Allocate Risk in Options Markets
Contextual Bandits for Trade Sizing and Q-Learning for Capital Allocation

options_exposure_learning_with_appendix.pdf is the full typeset paper.
options_exposure_learning.pdf contains 22 main/reference pages;
technical_appendix.pdf contains the remaining 6 appendix pages.
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
