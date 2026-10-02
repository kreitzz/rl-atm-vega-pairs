# Why the $0.65 RL checkpoint loses in 2018 and 2025

Saved original RL-sized policy, the matched 345 pairs, actual $0.03 commission. No fitting, pricing, sizing or policy changes. Calendar-year P&L includes positions carried across year boundaries; shared financing stays separate from leg attribution. 2025 ends August 29.

| Year | Gross hedged P&L | Option commission | Option penalty | Stock execution | Stock borrow | Financing vs cash | Net excess P&L | New entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | $-5,536 | $30 | $533 | $1,069 | $266 | $-481 | $-7,917 | 14 |
| 2025 | $-9,850 | $12 | $209 | $296 | $85 | $3,764 | $-6,687 | 5 |

Option and hedge P&L must be read together; gross excludes execution, stock borrow and shared financing. Net excess dollars are the annual change in portfolio NAV less the annual change in the cash benchmark. Cash earns interest even when the trading sleeve loses. Initial vega matching and daily delta hedging do not eliminate relative gamma/jump risk or later vega drift. Event causes require contract/underlying inspection; this ledger establishes where losses occurred.

2018 had 14 entered cohorts. RRC/SLB lost $4,041 above cash and BAC/C lost $3,601, together accounting for $7,642 of the $7,917 annual shortfall. Before costs and shared financing, short RRC lost $903 and long SLB lost $2,617; long BAC lost $1,097 and short C lost $2,038. HES/CVX generated $565 gross across eight entries, but costs and financing turned that into a $322 shortfall. The loss therefore combines unsuccessful relative-volatility directions in two trades with costs exceeding a small third-pair edge.

2025 had five entered cohorts: four long CENX / short AA entries on March 18–21, plus one long TRV / short CB entry on March 6. The four aluminum entries used the same two option IDs (167479063 and 163651105), so they represent repeated exposure to one setup, not four independent opportunities. CENX/AA lost $8,592 above cash; its long CENX leg lost $2,521 gross after delta hedging and short AA lost $6,162. TRV/CB lost $2,135 above cash; gross long TRV earned $673 while short CB lost $2,276. On April 11 alone, CENX/AA lost $7,488. Quote or market-event causes have not been inferred from these accounting numbers.

2025 gross hedged P&L was −$9,850. Option commissions were just $12 and the vol-point entry penalty was $209, so execution fees do not explain this loss. Financing versus cash was +$3,764 in aggregate and reduced the reported annual shortfall; it includes interest on previous years' trading gains and losses, not solely current-position funding. The weakness is sparse, concentrated trading and an unfavorable realized path for the chosen long/short directions. Negative annual Sharpe describes the loss relative to low daily excess-return volatility; it is not a large percentage portfolio drawdown.

Daily attribution reconciles each saved sleeve to less than $5e-10, and the summed portfolio to less than $0.000001. The current signal set is IV spread, trailing 30-calendar-day realized vol for both legs, earnings proximity, relative RV-minus-IV, a prior-year spread reversion z-score, rolling beta/correlation, completed-outcome profitability/count memory by direction over 91/365 days, prior VIX close, smile/term-structure values and availability flags, and contract/cost characteristics. All these inputs are unchanged by the diagnosis.

[Annual components](atm65_loss_years/annual_components.csv) · [Pairs](atm65_loss_years/pairs.csv) · [Legs](atm65_loss_years/legs.csv) · [Worst days](atm65_loss_years/worst_pair_days.csv) · [Entries](atm65_loss_years/entries.csv) · [Audit](atm65_loss_years/audit.json)
