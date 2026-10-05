"""Column-backed lookups avoid creating a dictionary for every cached quote."""
from functools import lru_cache
import pandas as pd
from scripts.option_flex.contract_engine import MarketRow

class CompactIndex:
    def __init__(self,frame,keys):
        self.index=pd.MultiIndex.from_frame(frame[keys])
        if self.index.has_duplicates:raise ValueError('Duplicate market keys')
        self.arrays={c:frame[c].array for c in frame};self.loc=self
        self._cached=lru_cache(maxsize=4096)(self._row)
    def __getitem__(self,key):return self._cached(key)
    def _row(self,key):
        row=self.index.get_loc(key)
        return MarketRow({c:a[row] for c,a in self.arrays.items()})

class CompactPreparedMarket:
    def __init__(self,quotes,stocks):
        self.sources=(quotes,stocks,None)
        self.quotes=CompactIndex(quotes,['date','optionid'])
        self.stocks=CompactIndex(stocks,['date','secid'])
        self.selection_ids=None
