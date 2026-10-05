# 90-day-only test — complete

Same 346-pair roster, original state, costs, risk limits, monthly retraining, and sizing ladder. The expanded policy trades only approximately 90-day options (75–105 days at entry), with independent ATM/10/25/50/75/90-delta choices and matched calls/calls or puts/puts. A separate ATM90 calls-only control uses the same training procedure. There are 72 curve combinations; the annual past-training-only shortlist retains ATM90 plus up to seven alternatives.

Vega is matched at entry within the existing integer tolerance; delta is hedged daily; each leg is held to its own expiry. A 110-calendar-day runoff protects both validation and final evaluation. Historical features remain the original reference state; this tests contract choice rather than a new feature model. Earlier 30-day results used a shorter runoff and are contextual, not a matched maturity comparison.

The existing raw cache ends at 60 DTE. Download only the missing 61–110 DTE calls/puts, select 90-day contracts, then combine their new early histories with the existing 0–60 DTE quotes. Do not substitute short-lived contracts or synthetic prices for the missing long-dated history.

The queue prioritizes the AMD/NVDA data partitions, real-data reconciliation, and both policies for that pair before completing the full roster. This is an execution-order change; it does not narrow the experiment. Each history partition requires completed selections for its securities in the current and prior calendar year. Streaming remains bounded to 16,000 downloaded rows or 32,768 preparation rows per batch.

Resource controls: one low-priority process, one numerical thread, 50% wall-clock duty cycle (increased from 20% at the user's request), 3 GiB worker RSS stop threshold, 30-minute job timeout, 1-second pauses between checkpoints, and a 10 GiB disk-space floor. Stop safely by creating `data/curve90/STOP`. Only one pipeline instance can hold the runner lock. The child exits if its supervisor disappears while running.

Completed both policies for all 346 pairs with no failures. See [results](curve90.md). Completion is recorded in `artifacts/contracts/curve90/task_status.json`; the runner has exited. The final label-storage optimization reproduced both AMD/NVDA pilot policies exactly, including daily NAV, trades, actions, decisions, and fits.
