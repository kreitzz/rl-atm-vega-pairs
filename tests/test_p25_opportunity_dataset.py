import numpy as np
import pandas as pd

from scripts.p25_opportunity_dataset import evaluation_mask, orient_candidates


def sample():
    row = dict(date=pd.Timestamp('2020-01-03'), pair='a__b', observed=True,
               eligible=True, context=True, rule_direction=0, signal_eligible=False,
               iv_mean=.3, iv_gap_relative=.1, iv_rank=.6, vrp_gap_relative=.2,
               rv_mean=.2, beta=1., correlation=.7, dte_mean=30., dte_gap=4.,
               commission_vol_points=.2, vix=.2, finite_label=True,
               target_long=2., target_short=-3., end=pd.Timestamp('2020-02-03'))
    for days in [91, 365]:
        for prefix, value in [('profit', 5.), ('count', 10.)]:
            row[f'{prefix}_long_{days}d'] = value
            row[f'{prefix}_short_{days}d'] = value + 1
    return pd.DataFrame([row])


def test_middle_rank_direction_features_are_reconstructed():
    rows = orient_candidates(sample()).set_index('direction')
    assert rows.loc[1, 'oriented_iv_gap'] == .1
    assert rows.loc[-1, 'oriented_iv_gap'] == -.1
    assert rows.loc[1, 'profit_91d'] == 5.
    assert rows.loc[-1, 'profit_91d'] == 6.
    assert rows.loc[1, 'target_net_per_reference_vega'] == 2.
    assert rows.loc[-1, 'target_net_per_reference_vega'] == -3.
    assert not rows.legacy_selected.any()


def test_future_completion_cannot_change_signal_eligibility():
    meta = sample()
    cal = meta[['pair', 'date']].copy()
    before = evaluation_mask(meta, cal)
    meta['finite_label'] = False
    meta['target_long'] = np.nan
    meta['target_short'] = np.nan
    assert evaluation_mask(meta, cal).equals(before)
    rows = orient_candidates(meta)
    assert rows.target_net_per_reference_vega.isna().all()
    assert not rows.train_eligible.any()
