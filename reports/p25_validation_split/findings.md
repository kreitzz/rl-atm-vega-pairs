# Validation selection and retrospective holdout

Select within the inherited30D25-delta put setup using2018–2021 only, then evaluate2022–29August2025. These later dates were inspected in earlier research, so this is not an untouched holdout. Instrument, roster and formation calendar were not reselected.

| Proposal pool | Validation choice | Validation Sharpe |2022–2025 selected Sharpe | Fixed-full Sharpe |
|---|---|---:|---:|---:|
| Screened |27 inputs,18months,50/75/100% |0.575 |1.263 |1.055 |
| Broader |33 inputs,18months,25/50/75/100% |1.309 |0.456 |0.567 |

Screened validation uses the archived345m research account; its selected policy is replayed at10m for final evaluation. All final controls use10m. The two pools select separately, so their difference is not a pure gate ablation.

The screened policy makes110 new entries and adds$248,673 above cash, versus75 entries and$200,853 at fixed full. The Sharpe gain is0.208 (paired126-session circular-block bootstrap,1000 replications,95% interval−0.103 to+0.424; no search adjustment). Broader sizing loses to fixed full, with900 entries and$278,895 above cash versus459 and$330,791.

Books continue across2022; carried inventory contributes to holdout P&L. Monthly refits use completed outcomes, including earlier holdout outcomes once completed. Only configuration choice is frozen.2025 is partial.

[Locked selection](selection.json) · [All validation candidates](validation_candidates.csv) · [Period results](periods.csv) · [Yearly results](annual.csv) · [Intervals](holdout_intervals.csv) · [Reproduction](reproduction.txt)
