Working manuscript: main.tex
Stable compiled preview: main.pdf
Compile: pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex
Main structure: Introduction -> Related Work -> Data and Backtesting Framework -> Put Pair Strategy -> ATM Straddle Strategy -> Capital Allocation with Fitted Q-Learning -> Discussion and Conclusion
Appendices retain detailed supporting analyses. Earlier manuscripts and research assets remain in their existing paper subdirectories; the export ZIP contains only the active main.tex project.

Manifest roles: source_manifest.json preserves the research-input provenance used by submission CI. latex_source_manifest.json records the current LaTeX project files and hashes.

Appendices A-D: contracts/features/risk and provenance; model estimation/selection; supplementary results/diagnostics; straddle attribution/reconciliation. Performance measures are in main Section 3.5.

Canonical working branch: main. Current source and PDF are paper/main.tex and paper/main.pdf. Research manifests used by submission checks are preserved separately from latex_source_manifest.json.

Sections 4-6 each contain methods and numbered Results subheadings. Joint allocation results are in Section 6.4; Discussion and Conclusion is Section 7.

Research sources and reproduction scope
Paths below are relative to the repository root, not the source-only Overleaf ZIP.
- Put entry results: reports/p25_validation_split/periods.csv
- Straddle entry results: reports/straddle_policy_refinement/comparison.csv
- Exact entry-policy explanations: reports/retained_entry_explanations/
- Allocation results, decisions, and convergence checks: reports/signal_allocation_variance/
- Independently fitted fixed-entry Q policies and four-way comparisons: reports/signal_learning_factorial/
- Attribution snapshots: reports/strategy_economics/data/

The factorial checks reproduce all eight pure-strategy endpoints and 324 monthly checkpoint timings. Research source provenance is recorded in source_manifest.json; current document and figure hashes are recorded in latex_source_manifest.json. The originating research records are on main.

Rebuild the manuscript with the compile sequence at the top of this file; no model refitting is required. The actual-contract ledger and the fractional-index allocator are separate implementations. Model fitting requires the licensed source cache or suitable archived derived inputs. Underlying WRDS data require the appropriate subscriptions and vendor permissions; redistribution restrictions apply.

The strategy-economics snapshots and manifest support reproduction of attribution and associated figures without raw option chains. The associated renderer checks snapshot hashes, cohort identities, aligned account dates, and saved performance arithmetic. Snapshot-based reproduction verifies archived results; it is not an independent regeneration of the original model-fitting and trading experiments. The source-only Overleaf export does not include the complete research pipeline or licensed source data.

Pair-universe roster: paper/pair_universe.csv records prepared pairs and candidate-date ranges, not continuous quote availability.
