"""Small, pooled tabular fitted-Q controller for actual option cohorts.

All reward transitions are dated. Flat is a zero-exposure episodic action;
the continuation model covers holding an existing contract, not a hypothetical
investment in a continuously tradable IV index. No oracle equilibrium reward.
"""
import numpy as np
import pandas as pd

PATHS = 257  # 256 four-move patterns plus explicit unavailable history.
BASE_STATES = PATHS * 3 * 3
HELD_STATES = BASE_STATES * 3 * 6
BACKOFF_BASE = 5 * 3 * 3
BACKOFF_HELD = BACKOFF_BASE * 3 * 6
LEVELS = np.array([2, 3, 4, 2, 3, 4])
DIRECTIONS = np.array([-1, -1, -1, 1, 1, 1])
LABEL_CODES = np.array([3, 2, 1, 6, 7, 8])
GAMMA = .99


def spread_states(spread):
    """Every row is computed from that close and its available history only."""
    x = pd.Series(spread, dtype=float)
    prior = x.shift(1).rolling(126, min_periods=63)
    mu, sd = prior.mean(), prior.std()
    z = ((x-mu)/sd).replace([np.inf, -np.inf], np.nan)
    changes = x.diff()
    scale = changes.shift(1).rolling(126, min_periods=20).std()
    scaled = changes/scale
    move = np.where(scaled > 1, 3, np.where(scaled > 0, 2, np.where(scaled < -1, 0, 1)))
    path = np.zeros(len(x), np.int32)
    available = np.ones(len(x), bool)
    for lag in range(4):
        path += np.roll(move, lag)*4**lag
        available &= np.isfinite(scaled.shift(lag).to_numpy())
    path[~available] = 256
    zbin = np.where(z < -1, 0, np.where(z > 1, 2, 1))
    values, means, zs = x.to_numpy(), mu.to_numpy(), z.to_numpy()
    durations = np.full(len(x), np.nan)
    failures = np.full(len(x), np.nan)
    counts = np.zeros(len(x), np.int32)
    age = np.zeros(len(x), np.int32)
    events = []
    active = None
    for t in range(len(x)):
        if active is not None:
            start, equilibrium, sign = active
            age[t] = t-start
            crossed = np.isfinite(values[t]) and sign*(values[t]-equilibrium) <= 0
            if crossed or t-start >= 63:
                events.append((start, t, bool(crossed), t-start))
                active = None
        if active is None and np.isfinite(zs[t]) and abs(zs[t]) >= 2:
            active = (t, means[t], np.sign(zs[t]))
        events = [e for e in events if e[0] >= t-252]
        finished = [e[3] for e in events if e[2]]
        censored = int(active is not None)
        counts[t] = len(events)+censored
        if finished:
            durations[t] = np.mean(finished)
        if counts[t]:
            failures[t] = (sum(not e[2] for e in events)+censored)/counts[t]
    # A historical excursion diagnostic, not the paper's retrospective local
    # extreme estimator. Unfinished excursions stay explicit and cannot make
    # an unreliable spread appear reliably mean reverting.
    reliable = np.isfinite(durations)&(failures < .5)&(counts >= 3)
    reversion = np.where(reliable, np.where(durations < 10, 0, 1), 2)
    base = (path*3+zbin)*3+reversion
    return pd.DataFrame(dict(base_state=base.astype(np.int32), path_code=path,
        spread_z=z, reversion_days=durations, failed_or_censored_fraction=failures,
        excursion_count=counts, active_excursion_age=age), index=x.index)


def held_state(base, age, action):
    agebin = np.where(np.asarray(age) < 5, 0, np.where(np.asarray(age) < 15, 1, 2))
    return (np.asarray(base)*3+agebin)*6+np.asarray(action)


def base_backoff(states):
    states = np.asarray(states, dtype=np.int64)
    path, rest = states//9, states % 9
    last_move = np.where(path == 256, 4, path % 4)
    return last_move*9+rest


def held_backoff(states):
    states = np.asarray(states, dtype=np.int64)
    return base_backoff(states//18)*18+states % 18


def pooled_mean(states, values, count, backoff, backoff_count, prior=12.):
    fine_n = np.bincount(states, minlength=count).astype(float)
    fine_sum = np.bincount(states, weights=values, minlength=count)
    group = backoff(states)
    coarse_n = np.bincount(group, minlength=backoff_count).astype(float)
    coarse_sum = np.bincount(group, weights=values, minlength=backoff_count)
    coarse_mean = np.divide(coarse_sum, coarse_n, out=np.zeros_like(coarse_sum), where=coarse_n > 0)
    fallback = coarse_mean[backoff(np.arange(count))]
    estimate = (fine_sum+prior*fallback)/(fine_n+prior)
    return estimate, coarse_n[backoff(np.arange(count))]


def causal_rows(rows, cutoff, months=18):
    start = pd.Timestamp(cutoff)-pd.DateOffset(months=months)
    return np.flatnonzero((rows.date >= start)&(rows.date < cutoff)&(rows.end < cutoff)&rows.finite_label)


def fit_q(entry_base, held, following, reward, close_reward, executable_close,
          terminal, entry_reward, entry_next, entry_valid, risk_penalty=.25,
          iterations=40):
    """Batch Bellman updates on completed historical cohort transitions.

    The risk utility is a declared mean/variance proxy, not exact portfolio
    Sharpe optimization. Its scale is estimated solely from this fit's rewards.
    """
    scale = max(float(np.std(reward, ddof=1)), 1e-6)
    utility = lambda r: r-risk_penalty*r*r/(2*scale)
    attempted_reward = np.where(executable_close,close_reward,reward)
    close_q, close_n = pooled_mean(held, utility(attempted_reward),
        HELD_STATES, held_backoff, BACKOFF_HELD)
    hold_q, hold_n = pooled_mean(held, utility(reward), HELD_STATES, held_backoff, BACKOFF_HELD)
    for _ in range(iterations):
        continuation = np.maximum(hold_q[following], np.where(close_n[following] > 0, close_q[following], -np.inf))
        continuation[terminal] = 0
        hold_q, hold_n = pooled_mean(held, utility(reward)+GAMMA*continuation,
            HELD_STATES, held_backoff, BACKOFF_HELD)
        # An unavailable next-session quote leaves inventory alive. The close
        # attempt gets the held PnL and continuation, not an omitted sample or
        # an imaginary successful liquidation.
        close_q, close_n = pooled_mean(held, utility(attempted_reward)+
            GAMMA*continuation*(~executable_close&~terminal),
            HELD_STATES, held_backoff, BACKOFF_HELD)
    model=dict(hold_q=hold_q, close_q=close_q, hold_n=hold_n, close_n=close_n,
        reward_scale=scale, risk_penalty=risk_penalty,gamma=GAMMA, bellman_iterations=iterations)
    fit_entry_head(model,entry_base,entry_reward,entry_next,entry_valid)
    return model


def fit_entry_head(model,entry_base,entry_reward,entry_next,entry_valid):
    next_value=np.maximum(model['hold_q'][entry_next],np.where(model['close_n'][entry_next]>0,model['close_q'][entry_next],-np.inf))
    utility=entry_reward-model['risk_penalty']*entry_reward**2/(2*model['reward_scale'])
    # An unfilled entry is an observed zero-exposure outcome, not a sample to
    # discard after discovering next-session eligibility.
    entry_target=np.where(entry_valid,utility+GAMMA*next_value,0.)
    entry_q = np.zeros((BASE_STATES, 6))
    entry_n = np.zeros_like(entry_q)
    for action in range(6):
        entry_q[:, action], entry_n[:, action] = pooled_mean(entry_base, entry_target[:, action],
            BASE_STATES, base_backoff, BACKOFF_BASE)
    model.update(entry_q=entry_q,entry_n=entry_n)


def entry_actions(model, states, sized=True):
    menu = np.arange(6) if sized else np.array([2, 5])
    values = model['entry_q'][np.asarray(states)[:, None], menu[None, :]].copy()
    # Fine patterns borrow the last-move pooled estimate even without a fine
    # observation. No pair is disabled for lacking a specific pattern.
    best = np.argmax(values, axis=1)
    chosen = menu[best]
    positive = values[np.arange(len(states)), best] > 0
    return np.where(positive, DIRECTIONS[chosen], 1), np.where(positive, LEVELS[chosen], 0)


def should_close(model, states):
    states = np.asarray(states)
    # Unseen pooled inventory/path states retain rather than liquidate by a
    # default zero prediction. Close actions need observed transition support.
    return (model['close_n'][states] > 0)&(model['close_q'][states] > model['hold_q'][states]+1e-12)
