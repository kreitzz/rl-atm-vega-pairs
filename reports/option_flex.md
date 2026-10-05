# Call/put choice on the 0.226 Sharpe reference

Allowing independent call/put choice changed annualized excess Sharpe from 0.226 to -0.024 on the same 346 pairs.

| Policy | Annualized excess Sharpe | P&L above cash | Entries | Active pairs |
|---|---:|---:|---:|---:|
| Calls only | 0.226 | $105,789 | 280 | 35 |
| Call/put choice | -0.024 | $-10,598 | 277 | 34 |

Actual entries by option type (first letter is the first underlying): CC: 152, CP: 29, PC: 84, PP: 12.

Largest pair-level P&L changes versus calls only: UAL/DAL $-44,644, GES/TJX $-40,027, A/TMO $-24,047. These compare retrained policies, so entries and trade dates can differ.

The full fixed 346-pair VIX reference roster is retained. Each pair retrains independently with 33 actions: CC, CP, PC or PP at each nonzero size in the original ladder, plus flat. Puts match the original selected call’s strike and expiry. Direction determines which underlying is long vega; call/put selection does not reverse that direction.

Market-state features, beta 0.85–1.15, correlation ≥0.5, monthly retraining, trailing training windows, validation, recent refits, hold-to-expiry treatment, costs and per-pair risk limits are retained. Each contract combination has its own completed-outcome profitability-memory gate. The original call-memory features remain the tree inputs; put-specific memory is used as a gate, not added state features. Tree partitions use the CC directional payoff target; Sharpe optimization selects among all 33 actions.

Training requires completed finite trajectories for every available action; an unresolved trajectory excludes that historical training row. Missing or ineligible put entries are zero-reward no-trades, and a put action cannot deploy without a valid signal-date selection. As in the reference, training uses standalone fixed-capital payoff trajectories; validation and evaluation apply the full portfolio risk overlay and NAV-dependent sizing. The wider action set can therefore change training samples, memory eligibility, validation and fitted policies. This is a retrained policy comparison, not a fixed-trade attribution.

Vega is matched initially using integer contracts within the existing 10% mismatch tolerance; daily delta hedging continues. Vega can drift after entry, and expiry mismatch can leave one leg outstanding. The fixed 0.1-vol-point entry penalty per leg, commissions, stock execution, financing and borrow settings are unchanged. The model uses European Black–Scholes call/put fits with declared-dividend adjustment and intrinsic cash settlement; American early exercise/assignment is not modeled. Put results require that qualification.

346/346 pairs completed. Every aggregate uses the same completed roster on both sides; incomplete books remain disclosed. The original full calls-only reference Sharpe was 0.2257. For pairs with any beta/correlation-eligible evaluation date, frozen calls-only actions were replayed through the new engine and required to reproduce each original NAV path within $0.0000001. The other 218 sleeves are provably ineligible regardless of option type and retain their verified cash-only NAV; they remain in the aggregate capital base.


January 2018–August 2025. Sharpe is daily cash-excess return annualized by √252, including inactive sleeves. Each sleeve starts with $1m. Percentage returns include cash interest. Shared-underlying exposures are not subject to an additional global portfolio limit. These are previously inspected research dates, not an untouched holdout. As in the reference, validation tests the pre-refit proposal rather than the exact deployed weights.

[Pair results](option_flex/pair_results.csv) · [Pair changes](option_flex/pair_changes.csv) · [Annual results](option_flex/annual.csv) · [Call/put entry mix](option_flex/option_types.csv) · [P&L components](option_flex/pnl_components.csv) · [Audit](option_flex/audit.csv) · [Matched-mark quality](option_flex/matched_mark_quality.csv)
