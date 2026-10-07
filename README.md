# Reinforcement learning for options relative value

Short-term 25-delta put pairs, initially vega neutral and delta hedged daily. A frozen spread signal proposes trades; a contextual learner selects 50%, 75% or 100% vega exposure using monthly fits on completed past outcomes. The retained learner uses direct Sharpe optimization, not Q-learning.

**Validation-selected filtered strategy:** RL achieves **0.573 versus 0.478** excess Sharpe at fixed full sizing in 2018–2021 validation, and **1.263 versus 1.055** in January 2022–August 2025 evaluation. All these comparisons use $10m funded accounts. Evaluation profit above cash is $248,673 versus $200,853. Selection ranks 33 archived screened configurations using only validation scores; the selected 27-input, 18-month, 50/75/100% policy keeps the profitability filter. Archived selection scores use the original $345m research account. This is a retrospective split within an inherited strategy and universe, not an untouched holdout.

[New selection and findings](reports/p25_validation_split/findings.md) · [Both-period results](reports/p25_validation_split/periods.csv) · [Yearly results](reports/p25_validation_split/annual.csv) · [New configuration](config/p25_validation_2018_2021.json) · [Models and selection evidence](data/p25_validation_split_experiment.zip)

The original **1.003 versus 0.769** full-history checkpoint remains preserved. Sizing changes inventory and admission, so portfolios differ; the new evaluation Sharpe-gain interval includes zero. The broader-pool validation winner did not beat fixed full sizing in the later period.

[10-page paper](paper/options_relative_value.pdf) · [Latest saved presentation](slides/230ZB_volatility_spreads%20%281%29.pptx) · [Browser slides](slides/index.html) · [Submission and reproduction guide](SUBMISSION.md)

[Finalized strategy inputs (CSV)](data/p25_variance_premium_inputs.csv): **36,327 rows and 33 model inputs**, plus contract identifiers and entry flags. No future rewards or targets; profitability memory uses completed past outcomes. Saved models, separate training-label paths and selected execution data are in [data/submission](data/submission/manifest.json).

[Checkpoint](config/p25_submission.json) · [Controlled results](reports/p25_capital_sensitivity/comparison.csv) · [Yearly returns and Sharpe](reports/p25_capital_sensitivity/10m_learned_annual.csv) · [Research and extension results](SUBMISSION.md#results-and-research-history)

Training commission is $0.65 per contract; execution uses $0.03 plus a 0.10 vol-point penalty per leg and 1 bp stock hedge cost. Financing, dividends and inventory use a shared cash ledger. Total account return includes cash interest; reported excess Sharpe removes the cash return.

The earlier ATM result and your teammate's equity/Q-learning experiments remain in the history. Newer standalone one-year cheap-vega tests lost money after costs and are reported separately.

[Earnings-state test](reports/p25_earnings/findings.md): adding earnings countdowns to sizing gave 0.977 Sharpe with historical dates assumed known and 0.956 with dates estimated from prior reports, versus 1.003 retained. [Inputs, fitted models and replay archive](data/p25_earnings_experiment.zip).

[Profitability-memory gate test](reports/p25_memory_ablation/findings.md): removing profit requirements gave 0.626 Sharpe; removing those and count minimums gave 0.835, with 1,334 trades and $787k above cash versus 212 trades and $323k retained. The ungated fixed-size control scored 0.713. [Reproduction archive](data/p25_memory_ablation_experiment.zip).

[Ungated sizing and earnings tests](reports/p25_ungated_search/findings.md): ten variants leave 0.835 best; adding a 25% size gives 0.830 with 1,938 trades, while prior-report earnings inputs give 0.768. All four constant-size controls score lower. [Chart](reports/p25_ungated_search/comparison.png) · [Reproduction archive](data/p25_ungated_search_experiment.zip).

**Academic use only.** Third-party data remain subject to vendor and institutional licenses; this notice grants no additional data rights. This repository remains private. [Data-use notice](data/DATA_LICENSE.txt).
