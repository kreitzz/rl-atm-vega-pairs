"""Contract mapping for an unchanged, frozen P25 decision stream."""
import numpy as np
import pandas as pd

WINGS = {'ATM': ('C', 'atm'), 'P25': ('P', '25'), 'P75': ('P', '75')}


def choose_existing_first(existing, additional, wing):
    """No replacement of an existing choice; fallback uses only signal quotes.

    Additional quotes retain OI100, the original experiment's requirement.
    This is a bounded fallback from the already selected cache, not a claim
    to find every possible OI100 expiry in the raw chain.
    """
    cp, bucket = WINGS[wing]
    old = existing.loc[existing.cp_flag.eq(cp) & existing.bucket.eq(bucket), ['date', 'optionid']].copy()
    old['contract_source'] = 'existing'
    new = additional.loc[additional.cp_flag.eq(cp) & additional.bucket.eq(bucket) &
                         additional.open_interest.ge(100), ['date', 'optionid']].copy()
    new['contract_source'] = 'additional'
    assert not old.date.duplicated().any() and not new.date.duplicated().any()
    return pd.concat([old, new], ignore_index=True).drop_duplicates('date', keep='first').set_index('date')


def transfer_rows(anchor, maps, wing, existing_only=False):
    """Same pair/date/direction/size decisions, with contract IDs remapped.

    Missing quotes remain explicit missing IDs; no execution or forward-path
    filter determines which observations reach the replay.
    """
    rows = anchor.copy()
    rows['wing'] = wing
    if wing == 'P25':
        rows['contract_source_a'] = rows['contract_source_b'] = 'frozen_P25'
        return rows
    for side in ['a', 'b']:
        rows['optionid_' + side] = np.nan
        rows['contract_source_' + side] = 'missing'
        for sid, idx in rows.groupby('secid_' + side).groups.items():
            chosen = maps[(int(sid), wing)]
            if existing_only:
                chosen = chosen.loc[chosen.contract_source.eq('existing')]
            mapped = chosen.reindex(pd.DatetimeIndex(rows.loc[idx, 'date']))
            rows.loc[idx, 'optionid_' + side] = mapped.optionid.to_numpy()
            rows.loc[idx, 'contract_source_' + side] = mapped.contract_source.fillna('missing').to_numpy()
    return rows
