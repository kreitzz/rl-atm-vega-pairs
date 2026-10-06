"""Isolated AMD/NVDA action-value and budget-constrained fitted-Q experiment.

Uses the original 30-calendar-day IV/RV payoff, not actual option prices.
Run with python -m scripts.run_q_sizing; --smoke checks one small fold.
No changes to legacy experiments, no network access, no worker pool.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from scripts.run_first_pass import fit_policy, net_rewards, payoff_sharpe
from scripts.run_rv_vix_pass import fit_tree_policy, leaf_states
from scripts.run_walk_forward import fold_data

FEATURES = ["spread", "amd_earnings_days_capped", "nvda_earnings_days_capped",
            "trailing_rv_amd", "trailing_rv_nvda"]
ACTIONS = np.arange(-1., 1.01, .25)
ZERO = 4
RISK_WEIGHTS = (0., .1, .5)
COST = .5
HISTORICAL_SHARPE = .7353226789575836


class Tape:
    """Cache immutable row observations for repeated counterfactual replay."""
    def __init__(self, frame):
        self.index, self.end, self.long_reward = frame.index, frame.end, frame.long_reward
        self.rows = []
        for _, source in frame.iterrows():
            row = SimpleNamespace(**source.to_dict())
            row._market = np.asarray([getattr(row, f) for f in FEATURES]+[row.iv_amd, row.iv_nvda]) / np.array([50., 31., 31., 50., 50., 50., 50.])
            row._entry = np.asarray([row.iv_amd, row.iv_nvda, row.trailing_rv_amd,
                                      row.trailing_rv_nvda, row.spread]) / 50.
            self.rows.append(row)
        self.iloc = self.rows

    def __len__(self):
        return len(self.rows)

    def iterrows(self):
        return zip(self.index, self.rows)


class Resources:
    """Single-core affinity, cooperative 50% CPU duty, RSS and STOP checks."""
    def __init__(self, root, duty=.5, memory_gib=3.):
        self.process = psutil.Process()
        self.root, self.duty, self.limit = Path(root), duty, memory_gib * 2**30
        self.wall, self.cpu = time.monotonic(), time.process_time()
        self.peak = 0
        self.affinity = False
        try:
            self.process.cpu_affinity([self.process.cpu_affinity()[0]])
            self.affinity = True
            self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        except (AttributeError, psutil.Error):
            pass

    def check(self):
        if (self.root / "STOP").exists():
            raise InterruptedError("STOP file present; saved fold checkpoints retained")
        self.peak = max(self.peak, self.process.memory_info().rss)
        if self.peak > self.limit:
            raise MemoryError("Research process exceeded memory limit")
        delay = (time.process_time() - self.cpu) / self.duty - (time.monotonic() - self.wall)
        # Check interruption between short sleeps; never freeze progress for a long wait.
        while delay > 0:
            time.sleep(min(delay, .25))
            if (self.root / "STOP").exists():
                raise InterruptedError("STOP file present")
            delay = (time.process_time() - self.cpu) / self.duty - (time.monotonic() - self.wall)


def load_panel(path):
    p = pd.read_csv(path, index_col="date", parse_dates=["date", "end"]).sort_index()
    required = FEATURES + ["iv_amd", "iv_nvda", "rv_amd", "rv_nvda", "end"]
    if not p.index.is_unique or p[required].isna().any().any():
        raise ValueError("Expected unique dates and complete benchmark inputs/outcomes")
    if not ((p.end - p.index).dt.days == 30).all():
        raise ValueError("This experiment requires exact 30-calendar-day outcomes")
    p["long_reward"] = (p.rv_amd - p.iv_amd) - (p.rv_nvda - p.iv_nvda)
    return p


def utility(net, risk, scale):
    """Mean/second-moment utility, explicitly different from exact Sharpe."""
    return net - risk * np.square(net) / scale


def action_values(train, tree, ids, risk, scale):
    state = leaf_states(tree, train, FEATURES, ids)
    reward = net_rewards(ACTIONS[None, :], train.long_reward.to_numpy()[:, None], COST)
    values = np.zeros((len(ids), len(ACTIONS)))
    for s in range(len(ids)):
        values[s] = utility(reward[state == s], risk, scale).mean(axis=0)
    # Prefer flat in a tie. All-zero utility should never create a spurious short.
    best = np.argmax(values, axis=1)
    best[values.max(axis=1) <= 0] = ZERO
    return ACTIONS[best], values


def fitted_entry_models(train, val, control, smoke=False):
    """Same six tree configurations and five CEM seeds as the saved reference."""
    grid = [(4, 126)] if smoke else [(n, m) for n in (4, 8, 16) for m in (126, 252)]
    scale = max(float(train.long_reward.std(ddof=1)), 1e-6)
    candidates = []
    for leaves, minimum in grid:
        tree, ids, cem, _, _ = fit_tree_policy(train, FEATURES, leaves, minimum, COST,
                                             action_values=ACTIONS)
        vstate = leaf_states(tree, val, FEATURES, ids)
        config = {"max_leaves": leaves, "min_leaf": minimum}
        candidates.append(dict(model="cem", config=config, tree=tree, ids=ids,
                               policy=cem, validation=payoff_sharpe(net_rewards(cem[vstate], val.long_reward, COST))))
        for risk in RISK_WEIGHTS:
            policy, values = action_values(train, tree, ids, risk, scale)
            candidates.append(dict(model="entry_q", config=dict(config, risk=risk), tree=tree,
                                   ids=ids, policy=policy, q=values,
                                   validation=payoff_sharpe(net_rewards(policy[vstate], val.long_reward, COST))))
        control.check()
    winners = {name: max((c for c in candidates if c["model"] == name),
                         key=lambda c: c["validation"]) for name in ("cem", "entry_q")}
    winners["entry_q_plain"] = max((c for c in candidates if c["model"] == "entry_q" and c["config"]["risk"] == 0.),
                                  key=lambda c: c["validation"])
    return winners, [{k: c[k] for k in ("model", "config", "validation")} for c in candidates], scale


def state_features(row, date, book, budget):
    """Observable current market and complete reservation/entry schedule.

    No forward outcome is read. Slots represent calendar days to maturity.
    Accrued intratrade realized variance is absent from the public panel; this
    is a partially observed market state, not a claim of exact Markov sufficiency.
    """
    market = getattr(row, "_market", None)
    if market is None:
        market = np.asarray([getattr(row, f) for f in FEATURES] + [row.iv_amd, row.iv_nvda], dtype=float)
        market /= np.array([50., 31., 31., 50., 50., 50., 50.])
    schedule = np.zeros((30, 7), dtype=np.float32)
    for e, a in book:
        days = int((e.end - date).days)
        if not 1 <= days <= 30:
            raise ValueError("Book must settle matured tickets before encoding")
        slot = schedule[days - 1]
        entry = getattr(e, "_entry", None)
        if entry is None:
            entry = np.array([e.iv_amd, e.iv_nvda, e.trailing_rv_amd, e.trailing_rv_nvda, e.spread])/50.
        slot += np.r_[a, abs(a), abs(a)*entry] / budget
    used = sum(abs(a) for _, a in book)
    return np.r_[market, budget/10., (budget-used)/budget, schedule.ravel()].astype(np.float32)


def settle(book, date, risks, scale):
    survivors, reward = [], np.zeros(len(risks))
    for e, a in book:
        if e.end <= date:
            net = float(a*e.long_reward-COST*abs(a))
            # Entry already paid the cost. Settlement pays gross and risk penalty.
            reward += a*e.long_reward - np.asarray(risks)*net*net/scale
        else:
            survivors.append((e, a))
    return survivors, reward


def available(book, budget):
    free = max(0., budget-sum(abs(a) for _, a in book))
    return np.abs(ACTIONS) <= free + 1e-6


def transition(panel, i, book, action, budget, risks, scale):
    if not np.isin(action, ACTIONS) or abs(action) > budget-sum(abs(a) for _, a in book)+1e-6:
        raise ValueError("Illegal action or capacity breach")
    row, date = panel.iloc[i], panel.index[i]
    following = list(book)
    reward = np.full(len(risks), -COST*abs(action))
    if action:
        following.append((row, float(action)))
    terminal = i == len(panel)-1
    next_date = panel.end.max() if terminal else panel.index[i+1]
    following, payout = settle(following, next_date, risks, scale)
    reward += payout
    if terminal:
        return following, reward, np.zeros(219, np.float32), np.ones(9, bool), False
    nxt = state_features(panel.iloc[i+1], next_date, following, budget)
    return following, reward, nxt, available(following, budget), True


def counterfactuals(panel, i, book, budget, risks, scale):
    """Encode shared successor inventory once, then insert each feasible ticket."""
    surviving, base_reward, base_state, _, alive = transition(panel, i, book, 0., budget, risks, scale)
    legal = available(book, budget)
    actions = np.where(legal, ACTIONS, 0.)
    magnitude = np.abs(actions)
    rewards = base_reward[:, None]-COST*magnitude[None, :]
    next_states = np.repeat(base_state[None, :], len(ACTIONS), axis=0)
    row = panel.iloc[i]
    next_date = panel.index[i+1] if alive else panel.end.max()
    if row.end <= next_date:
        net = actions*row.long_reward-COST*magnitude
        rewards += actions[None, :]*row.long_reward-np.asarray(risks)[:, None]*net[None, :]**2/scale
        next_masks = np.repeat(available(surviving, budget)[None, :], 9, axis=0)
    else:
        offset = 9+7*(int((row.end-next_date).days)-1)
        entry = getattr(row, "_entry", None)
        if entry is None:
            entry = np.array([row.iv_amd, row.iv_nvda, row.trailing_rv_amd, row.trailing_rv_nvda, row.spread])/50.
        next_states[:, offset:offset+7] += np.c_[actions, magnitude, magnitude[:, None]*entry[None, :]]/budget
        free = budget-sum(abs(a) for _, a in surviving)-magnitude
        next_states[:, 8] = free/budget
        next_masks = np.abs(ACTIONS)[None, :] <= free[:, None]+1e-6
    return rewards.astype(np.float32), next_states, next_masks, np.full(9, alive), legal


def predict_q(model, x, risk_index):
    result = model.predict(np.atleast_2d(x)).reshape(-1, len(RISK_WEIGHTS), len(ACTIONS))
    return result[:, risk_index, :]


def choose_q(model, x, mask, risk_index):
    if model is None:
        return ZERO
    q = predict_q(model, x, risk_index)[0]
    q = np.where(mask, q, -np.inf)
    maximum = q.max()
    return ZERO if np.isclose(q[ZERO], maximum, rtol=0, atol=1e-9) else int(q.argmax())


def simulate(panel, budget, decide, risks=RISK_WEIGHTS, scale=1.):
    panel = Tape(panel) if not isinstance(panel, Tape) else panel
    book, actions, daily, peak = [], [], np.zeros(len(risks)), 0.
    for i, (date, row) in enumerate(panel.iterrows()):
        x, mask = state_features(row, date, book, budget), available(book, budget)
        a = float(decide(i, x, mask, book))
        peak = max(peak, sum(abs(v) for _, v in book)+abs(a))
        book, reward, _, _, _ = transition(panel, i, book, a, budget, risks, scale)
        daily += reward
        actions.append(a)
    actions = np.asarray(actions)
    net = net_rewards(actions, panel.long_reward.to_numpy(), COST)
    np.testing.assert_allclose(daily, [utility(net, risk, scale).sum() for risk in risks], atol=1e-7)
    if book or peak > budget+1e-6:
        raise AssertionError("Unsettled book or budget breach")
    return actions, {"peak_used": peak, "settled": True, "net_payoff": float(net.sum())}


def training_transitions(train, budgets, initial, tree, ids, scale, control, smoke=False):
    """Generate full-action counterfactuals on diverse, feasible training books.

    Sampling controls coverage, not which action outcomes can be learned.
    Counterfactual rows remain dependent observations of one price history.
    """
    rows, rewards, next_rows, next_masks, live, legal = [], [], [], [], [], []
    leaves = leaf_states(tree, train, FEATURES, ids)
    train = Tape(train)
    for budget in budgets:
        for behavior in range(2 if smoke else 3):
            rng = np.random.default_rng(2026+behavior)
            book = []
            for i, (date, row) in enumerate(train.iterrows()):
                x, mask = state_features(row, date, book, budget), available(book, budget)
                rs, xs, ms, ls, mask = counterfactuals(train, i, book, budget, RISK_WEIGHTS, scale)
                rows.append(x); rewards.append(rs)
                next_rows.append(xs); next_masks.append(ms); live.append(ls); legal.append(mask)
                if behavior == 0:
                    wanted = initial[leaves[i]]
                    action = np.sign(wanted)*min(abs(wanted), budget-sum(abs(v) for _, v in book))
                    action = np.sign(action)*np.floor(abs(action)*4+1e-6)/4
                else:
                    options = np.flatnonzero(mask)
                    action = ACTIONS[rng.choice(options)] if rng.random()>.2 else 0.
                book, _, _, _, _ = transition(train, i, book, action, budget, RISK_WEIGHTS, scale)
                if i % 250 == 0:
                    control.check()
    return (np.asarray(rows, np.float32), np.asarray(rewards, np.float32),
            np.asarray(next_rows, np.float32), np.asarray(next_masks, bool),
            np.asarray(live, bool), np.asarray(legal, bool))


def fit_sequential(data, control, iterations=32, gamma=.98, trees=12, min_leaf=64):
    x, rewards, nx, next_masks, live, legal = data
    values = np.zeros((len(x), len(RISK_WEIGHTS), len(ACTIONS)), np.float32)
    model = None
    for iteration in range(iterations):
        continuation = np.zeros_like(rewards)
        if model is not None:
            # Prediction in chunks avoids a second large copy of the transition tensor.
            flattened = nx.reshape(-1, nx.shape[-1])
            masks = next_masks.reshape(-1, len(ACTIONS))
            best = np.empty((len(flattened), len(RISK_WEIGHTS)), np.float32)
            for start in range(0, len(flattened), 4096):
                stop = min(start+4096, len(flattened))
                p = model.predict(flattened[start:stop]).reshape(-1, len(RISK_WEIGHTS), len(ACTIONS))
                p = np.where(masks[start:stop, None, :], p, -np.inf)
                best[start:stop] = p.max(axis=2)
            continuation = best.reshape(len(x), len(ACTIONS), len(RISK_WEIGHTS)).transpose(0, 2, 1)
        target = rewards + gamma*live[:, None, :]*continuation
        target = np.where(legal[:, None, :], target, 0.)
        model = ExtraTreesRegressor(n_estimators=trees, max_depth=8, min_samples_leaf=min_leaf,
                                    max_features=.5, n_jobs=1, random_state=2026)
        model.fit(x, target.reshape(len(x), -1))
        residual = float(np.mean(np.abs(target-values)))
        values = target
        control.check()
        if iteration % 4 == 3 or iteration == iterations-1:
            print(f"  FQI sweep {iteration+1}/{iterations}, target change {residual:.4f}", flush=True)
    return model, {"sweeps": iterations, "last_target_change": residual, "observed_books": len(x),
                   "counterfactual_rows": int(legal.sum()), "gamma": gamma}


def constrained_population(panel, leaves, policies, budget):
    """Exact capacity replay of market-leaf policies, vectorized across candidates."""
    n, population = len(panel), len(policies)
    release = np.zeros((n+1, population))
    ends = panel.index.searchsorted(pd.DatetimeIndex(panel.end), side="left")
    used, total, square = np.zeros(population), np.zeros(population), np.zeros(population)
    gross = panel.long_reward.to_numpy()
    for i in range(n):
        used -= release[i]
        desired = policies[:, leaves[i]]
        magnitude = np.minimum(np.abs(desired), np.maximum(0., budget-used))
        magnitude = np.floor(magnitude*4+1e-6)/4
        a = np.sign(desired)*magnitude
        used += magnitude
        release[min(ends[i], n)] += magnitude
        net = a*gross[i]-COST*magnitude
        total += net; square += net*net
    mean = total/n
    variance = np.maximum((square-n*mean*mean)/(n-1), 0.)
    return np.sqrt(365/30)*np.divide(mean, np.sqrt(variance), out=np.zeros_like(mean), where=variance>1e-20)


def fit_constrained_cem(train, tree, ids, initial, budget, control, smoke=False):
    """Refit the existing market-state CEM policy under the identical capacity rule.

    Smaller fixed search budget than historical CEM; retains its incumbent.
    It does not receive the Q learner's added inventory/entry-history features.
    """
    states = leaf_states(tree, train, FEATURES, ids)
    probabilities = np.full((len(ids), len(ACTIONS)), 1/len(ACTIONS))
    rng = np.random.default_rng(2026)
    best = initial.copy()
    score = float(constrained_population(train, states, best[None, :], budget)[0])
    if score < 0:
        best, score = np.zeros_like(best), 0.
    population, iterations = (16, 3) if smoke else (64, 24)
    for _ in range(iterations):
        draws = rng.random((population, len(ids), 1))
        indices = (draws > probabilities.cumsum(axis=1)[None, :, :]).sum(axis=2).clip(max=8)
        policies = ACTIONS[indices]; policies[0] = best
        scores = constrained_population(train, states, policies, budget)
        k = scores.argmax()
        if scores[k] > score:
            best, score = policies[k].copy(), float(scores[k])
        elite = policies[np.argsort(scores)[-max(2, population//10):]]
        probabilities = .3*probabilities+.7*np.stack([(elite==a).mean(axis=0) for a in ACTIONS], axis=1)
        probabilities = .97*probabilities+.03/len(ACTIONS)
        control.check()
    return best, {"training_sharpe": score, "population": population, "iterations": iterations}


def entry_policy(tree, ids, policy, panel, budget):
    leaves = leaf_states(tree, panel, FEATURES, ids)
    def decide(i, x, mask, book):
        desired = policy[leaves[i]]
        free = max(0., budget-sum(abs(a) for _, a in book))
        return np.sign(desired)*np.floor(min(abs(desired), free)*4+1e-6)/4
    return decide


def summarize(name, actions, panel):
    r = net_rewards(actions, panel.long_reward.to_numpy(), COST)
    return {"model": name, "sharpe": payoff_sharpe(r), "entries": int(np.count_nonzero(actions)),
            "mean_size": float(np.abs(actions).mean()), "net_payoff": float(r.sum())}


def paired_interval(a, b, repeats=1000):
    rng = np.random.default_rng(2026)
    differences = []
    for _ in range(repeats):
        ix = ((rng.integers(len(a), size=int(np.ceil(len(a)/63)))[:, None]+np.arange(63))%len(a)).ravel()[:len(a)]
        differences.append(payoff_sharpe(a[ix])-payoff_sharpe(b[ix]))
    return np.quantile(differences, [.025, .975]).tolist()


def write_report(root, predictions, selections, settings, complete):
    panel = predictions
    models = [c[:-7] for c in panel if c.endswith("_action")]
    scores = [summarize(name, panel[name+"_action"].to_numpy(), panel) for name in models]
    summary = {"complete": complete, "settings": settings, "scores": scores,
               "first": str(panel.index.min().date()), "last": str(panel.index.max().date()),
               "entry_dates": len(panel), "folds_finished": sorted(panel.fold.unique().tolist()),
               "paired_difference_ci": {}}
    cem = net_rewards(panel.cem_action, panel.long_reward, COST).to_numpy()
    entry = net_rewards(panel.entry_q_action, panel.long_reward, COST).to_numpy()
    seq = net_rewards(panel.sequential_q_action, panel.long_reward, COST).to_numpy()
    constrained = net_rewards(panel.constrained_cem_action, panel.long_reward, COST).to_numpy()
    summary["paired_difference_ci"]["entry_q_minus_cem"] = paired_interval(entry, cem)
    summary["paired_difference_ci"]["sequential_q_minus_constrained_cem"] = paired_interval(seq, constrained)
    same = net_rewards(panel.same_budget_cem_action, panel.long_reward, COST).to_numpy()
    summary["paired_difference_ci"]["sequential_q_minus_same_budget_cem"] = paired_interval(seq, same)
    reference = next(s for s in scores if s["model"] == "cem")["sharpe"]
    summary["historical_reference_difference"] = reference-HISTORICAL_SHARPE
    annual = []
    for year, frame in predictions.groupby(predictions.index.year):
        for name in models:
            annual.append(dict(year=int(year), **summarize(name, frame[name+"_action"].to_numpy(), frame)))
    pd.DataFrame(annual).to_csv(root/"annual.csv", index=False)
    predictions.to_csv(root/"predictions.csv", index_label="date")
    pd.DataFrame(scores).to_csv(root/"scores.csv", index=False)
    (root/"selections.json").write_text(json.dumps(selections, indent=2))
    (root/"summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Q sizing on the original 30-day volatility proxy", "",
             f"Status: {'complete' if complete else 'partial; fold checkpoints retained'}. Run: `{root.as_posix()}`.", "",
             f"{len(panel):,} entry dates, {summary['first']}–{summary['last']}. AMD/NVDA only. "
             "All Sharpes use sqrt(365/30) on net entry-cohort proxy payoffs, including flat dates. "
             "These are not actual option-contract or daily NAV returns.", ""]
    if complete and not settings["smoke"]:
        by_name = {s["model"]: s for s in scores}
        ci = summary["paired_difference_ci"]["entry_q_minus_cem"]
        lines += [f"**Result:** original CEM {reference:.3f}; plain entry action values {by_name['entry_q_plain']['sharpe']:.3f}; "
                  f"validation-selected entry utility {by_name['entry_q']['sharpe']:.3f}; sequential Q "
                  f"{by_name['sequential_q']['sharpe']:.3f} versus constrained CEM "
                  f"{by_name['constrained_cem']['sharpe']:.3f}.", "",
                  f"Entry improvement is {by_name['entry_q']['sharpe']-reference:+.3f}, with paired 95% interval "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}]. "
                  +( "The interval includes zero, so a reliable advantage is not established. " if ci[0] <= 0 <= ci[1] else
                     "This interval does not correct for repeated research or validation/model-selection uncertainty. ")
                  +"The better entry point estimate comes from a different action-value/utility rule; it remains a one-decision "
                  "bandit control. The genuinely sequential capacity-control version did not improve performance.", ""]
    lines += [
             "| Model | Payoff Sharpe | Nonzero entries | Average size |", "|---|---:|---:|---:|"]
    for s in scores:
        lines.append(f"| {s['model']} | {s['sharpe']:.3f} | {s['entries']} | {s['mean_size']:.3f} |")
    lines += ["", "## Settings and interpretation", "",
              "The CEM reference reproduces the original five inputs, six tree configurations, five search seeds, "
              "nine signed size choices, outcome purging, two-year validation/test blocks, and fixed 0.5-point cost. "
              f"Historical payoff Sharpe: {HISTORICAL_SHARPE:.6f}; rerun difference: {reference-HISTORICAL_SHARPE:+.6f}.", "",
              "Entry Q is a terminal action-value control, not sequential learning. It uses the same fitted tree "
              "partitions and estimates mean/second-moment utility per action. Risk weights 0, 0.1, 0.5 are selected "
              "on preceding validation Sharpe; the training-only payoff SD supplies their scale. "
              "CEM has six validation candidates; entry Q has eighteen. Plain Q and risk-adjusted Q are reported separately.", "",
              f"Sequential Q uses capacities {settings['budgets']} full-size equivalents; values are selected on validation only. "
              "Existing positions reserve absolute initial vega until their exact 30-calendar-day maturity. "
              "Costs are paid at entry, gross rewards at maturity, and final tickets settle during runoff. "
              "Entry-cohort reporting exactly reconciles to these settlement rewards. No early exits or synthetic option marks.", "",
              "Tree-based fitted Q iteration learns nine action values for each of the three risk weights together. "
              "The inputs include the five current signals, current IV levels, available capacity, and a 30-slot "
              "reservation schedule with signed allocations and observable entry-IV/RV context. "
              "Forward RV, terminal rewards, raw dates and fold IDs are excluded. "
              "The source CSV lacks the complete intratrade return path, so this is partially observed, "
              "not an exactly sufficient Markov state. Counterfactual training rows add coverage, not independent market days.", "",
              f"FQI: {settings['q_iterations']} sweeps, gamma {settings['gamma']}, 12 single-worker extra trees, "
              "depth 8, minimum leaf 64. It optimizes discounted expected utility, not exact Sharpe. "
              "The smaller-budget constrained CEM control is refitted on the reference market partition, "
              "using 64 policies and 24 search iterations plus the original policy as incumbent. "
              "It sees capacity through execution clipping, but does not use Q's inventory features. "
              "The sequential comparison therefore mixes learning method, representation, objective and search budget; "
              "it is not a pure algorithm ablation.", "",
              "The 32-sweep FQI budget is fixed rather than a convergence claim; final mean target changes are "
              "saved per fold. Episode books restart flat at each split, with separate maturity runoff. "
              "Earnings dates retain the original known-at-entry assumption rather than a new point-in-time audit.", "",
              "## Validation selection and uncertainty", ""]
    for selection in selections:
        lines.append(f"- {selection['fold']}: entry Q {selection['entry_q']}; sequential Q {selection['sequential_q']}; "
                     f"constrained CEM {selection['constrained_cem']}.")
    lines += ["", "63-entry paired circular-block bootstrap, 1,000 draws:"]
    for name, ci in summary["paired_difference_ci"].items():
        lines.append(f"- {name}: 95% interval [{ci[0]:.3f}, {ci[1]:.3f}].")
    lines += ["", "All historical years have already been inspected. Intervals describe fixed realized policies; "
              "they do not account for repeated research, refitting or all selection uncertainty. "
              "Risk-shaped utility is never substituted for unshaped payoff in reported Sharpe.", "",
              "## Yearly results", "",
              "Payoffs are attributed to entry year; 2025 ends in August. The same frozen two-year-fold models are used.", "",
              "| Entry year | Original CEM | Entry utility | Sequential Q | Constrained CEM |",
              "|---|---:|---:|---:|---:|"]
    for year in sorted(predictions.index.year.unique()):
        values = {r["model"]: r["sharpe"] for r in annual if r["year"] == year}
        lines.append(f"| {year} | {values['cem']:.3f} | {values['entry_q']:.3f} | "
                     f"{values['sequential_q']:.3f} | {values['constrained_cem']:.3f} |")
    audit = settings.get("post_run_audit")
    if audit:
        lines += ["", "## Verification", "",
                  f"{audit['focused_tests_passed']} focused tests passed. Saved models reproduced all "
                  f"{audit['dates_replayed']:,} test-date decisions across eight folds. "
                  "Replacing unseen test outcomes left every selected sequential-Q decision unchanged. "
                  "Ticket settlement, proportional costs, split purging and capacity checks passed.", "",
                  f"Training took {settings['elapsed_seconds']/60:.1f} minutes at the configured one-core, "
                  f"50% duty limit; observed peak RSS was {settings['peak_rss_gib']:.3f} GiB. "
                  "The frozen training source is kept separately from the reporting source snapshot."]
    lines += ["",
              "## Files and reproduction", "",
              "`scripts/run_q_sizing.py` contains the full isolated experiment; `tests/test_q_sizing.py` checks "
              "payoff reconciliation, causal features, action masks, terminal settlement and deterministic learning.", "",
              "```powershell", "python -m unittest tests.test_q_sizing tests.test_sizing tests.test_walk_forward",
              "python -m scripts.run_q_sizing --smoke", "python -m scripts.run_q_sizing", "```", "",
              "Each timestamped ignored run directory keeps settings, source/version hashes, dated actions, "
              "validation grids, fold checkpoints and trusted-local model pickles. Only load pickles generated "
              "by this experiment. A STOP file in that run directory requests cooperative termination. "
              "One CPU core, 50% cooperative duty, one numerical thread and 3 GiB RSS stop. "
              "The memory guard runs between operations rather than enforcing a hard allocation ceiling."]
    report = "\n".join(lines)+"\n"
    (root/"report.md").write_text(report, encoding="utf-8")
    if complete and not settings["smoke"]:
        Path("reports/q_sizing.md").write_text(report, encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=Path("data/current_dataset.csv"))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--budgets", type=float, nargs="+", default=[1., 5., 10.])
    ap.add_argument("--q-iterations", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=.98)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if any(b < 1 or b*4 != round(b*4) for b in args.budgets):
        ap.error("Budgets must be at least one and multiples of 0.25")
    if not 0 <= args.gamma < 1 or args.q_iterations < 1:
        ap.error("Require 0 <= gamma < 1 and positive Q iterations")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output or Path("artifacts/q_sizing")/(stamp+("_smoke" if args.smoke else ""))
    root.mkdir(parents=True, exist_ok=False)
    control = Resources(root)
    panel = load_panel(args.panel)
    iterations = 3 if args.smoke else args.q_iterations
    settings = {"panel": str(args.panel), "source_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cost": COST,
                "features": FEATURES, "actions": ACTIONS.tolist(), "budgets": args.budgets,
                "risk_weights": RISK_WEIGHTS, "q_iterations": iterations, "gamma": args.gamma,
                "smoke": args.smoke, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__}, "cpu_duty": .5,
                "cpu_affinity": control.affinity, "memory_stop_gib": 3., "workers": 1}
    (root/"settings.json").write_text(json.dumps(settings, indent=2))
    (root/"experiment_source.py").write_bytes(Path(__file__).read_bytes())
    print(f"Run directory: {root}", flush=True)
    pieces, selections = [], []
    years = [2010] if args.smoke else list(range(2010, 2026, 2))
    for year in years:
        tr, va, te = fold_data(panel, year)
        if args.smoke:
            tr, va, te = tr.iloc[-320:], va.iloc[:70], te.iloc[:70]
        if not len(te):
            continue
        assert tr.end.max() < va.index.min() and va.end.max() < te.index.min()
        print(f"Fold {year}: {len(tr)} train / {len(va)} validation / {len(te)} test", flush=True)
        winners, grid, scale = fitted_entry_models(tr, va, control, args.smoke)
        cem = winners["cem"]
        dataset = training_transitions(tr, args.budgets, cem["policy"], cem["tree"], cem["ids"],
                                       scale, control, args.smoke)
        q, diagnostic = fit_sequential(dataset, control, iterations, args.gamma,
                                       trees=4 if args.smoke else 12, min_leaf=32 if args.smoke else 64)
        del dataset
        qgrid, cgrid, constrained_models = [], [], {}
        for budget in args.budgets:
            for ri, risk in enumerate(RISK_WEIGHTS):
                actions, audit = simulate(va, budget, lambda i, x, mask, book, ri=ri:
                                          ACTIONS[choose_q(q, x, mask, ri)], scale=scale)
                qgrid.append({"budget": budget, "risk": risk, "risk_index": ri,
                              "validation": payoff_sharpe(net_rewards(actions, va.long_reward, COST))})
            policy, cdiag = fit_constrained_cem(tr, cem["tree"], cem["ids"], cem["policy"], budget,
                                               control, args.smoke)
            constrained_models[budget] = policy
            actions, _ = simulate(va, budget, entry_policy(cem["tree"], cem["ids"], policy, va, budget), scale=scale)
            cgrid.append({"budget": budget, "validation": payoff_sharpe(net_rewards(actions, va.long_reward, COST)), **cdiag})
            control.check()
        qs = max(qgrid, key=lambda v: v["validation"])
        cs = max(cgrid, key=lambda v: v["validation"])
        out = te[["long_reward", "end"]].copy(); out["fold"] = year
        for name, model in winners.items():
            out[name+"_action"] = model["policy"][leaf_states(model["tree"], te, FEATURES, model["ids"])]
        out["cash_action"] = 0.
        out["short_higher_iv_action"] = -np.sign(te.spread)
        out["always_long_action"] = 1.
        out["always_short_action"] = -1.
        qa, audit = simulate(te, qs["budget"], lambda i, x, mask, book: ACTIONS[choose_q(q, x, mask, qs["risk_index"])], scale=scale)
        out["sequential_q_action"] = qa
        # Plain-reward Q gets its own validation-only capacity selection.
        plain = max((v for v in qgrid if v["risk"] == 0.), key=lambda v: v["validation"])
        out["sequential_q_plain_action"] = simulate(te, plain["budget"],
             lambda i, x, mask, book: ACTIONS[choose_q(q, x, mask, 0)], scale=scale)[0]
        # Same-capacity comparator prevents separate capacity selection explaining the gap.
        same = constrained_models[qs["budget"]]
        out["same_budget_cem_action"] = simulate(te, qs["budget"],
             entry_policy(cem["tree"], cem["ids"], same, te, qs["budget"]), scale=scale)[0]
        chosen = constrained_models[cs["budget"]]
        out["constrained_cem_action"] = simulate(te, cs["budget"],
             entry_policy(cem["tree"], cem["ids"], chosen, te, cs["budget"]), scale=scale)[0]
        pieces.append(out)
        selection = {"fold": year, "entry_q": winners["entry_q"]["config"], "cem": cem["config"],
                     "sequential_q": qs, "sequential_q_plain": plain, "constrained_cem": cs,
                     "audit": audit, "diagnostic": diagnostic,
                     "train_last_outcome": str(tr.end.max()), "validation_first": str(va.index.min()),
                     "validation_last_outcome": str(va.end.max()), "test_first": str(te.index.min())}
        selections.append(selection)
        (root/f"fold_{year}_validation.json").write_text(json.dumps({"entry": grid, "q": qgrid, "cem": cgrid}, indent=2))
        with (root/f"fold_{year}_models.pkl").open("wb") as f:
            pickle.dump({"entry": winners, "sequential": q, "scale": scale, "constrained": constrained_models}, f)
        out.to_csv(root/f"fold_{year}_predictions.csv", index_label="date")
        summary = write_report(root, pd.concat(pieces), selections, settings, False)
        print("  Finished", year, {s["model"]: round(s["sharpe"], 3) for s in summary["scores"]}, flush=True)
        control.check()
    settings["peak_rss_gib"] = control.peak/2**30
    settings["elapsed_seconds"] = time.monotonic()-control.wall
    (root/"settings.json").write_text(json.dumps(settings, indent=2))
    summary = write_report(root, pd.concat(pieces), selections, settings, True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
