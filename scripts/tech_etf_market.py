"""Combine isolated SMH history with the unchanged cached constituent history."""
from functools import lru_cache
from pathlib import Path
import pandas as pd
from scripts.run_broad_contract_replay import Market
from scripts.run_contract_surface_features import augment as stock_augment
from scripts.build_contract_surface import VALUES

TECH=Path('data/tech_etf')
SID=151720


class TechMarket(Market):
    def __init__(self,calendar):
        super().__init__(calendar)
        extra=pd.concat([pd.read_parquet(p,columns=['date','secid','close','cfadj'])
                         for p in sorted((TECH/'contracts/underlying').glob('*.parquet'))],ignore_index=True)
        extra['date']=pd.to_datetime(extra.date)
        assert not self.stocks.secid.eq(SID).any()
        self.stocks=pd.concat([self.stocks,extra],ignore_index=True)
        d=pd.read_parquet(TECH/'contracts/distributions.parquet')
        for col in ['ex_date','declare_date']:d[col]=pd.to_datetime(d[col])
        d=d[d.cancel_flag.eq('0')&d.distr_type.eq('1')&d.currency.eq('USD')&d.adj_factor.eq(1)&d.approx_flag.eq('0')&d.liquid_flag.eq('0')]
        self.divs=pd.concat([self.divs,d],ignore_index=True)

    @lru_cache(maxsize=4)
    def asset(self,sid):
        if sid!=SID:return super().asset(sid)
        qpaths=[TECH/'history'/str(y)/f'{sid}.parquet' for y in sorted(set(self.calendar.year))]
        spaths=[TECH/'history'/str(y)/f'{sid}_selection.parquet' for y in sorted(set(self.calendar.year))]
        q=pd.concat([pd.read_parquet(p) for p in qpaths if p.exists()],ignore_index=True)
        selection=pd.concat([pd.read_parquet(p) for p in spaths if p.exists()],ignore_index=True)
        stocks=self.stocks[self.stocks.secid.eq(sid)].set_index('date').reindex(self.calendar)
        stocks['secid']=sid;stocks['rate']=self.rates;stocks.index.name='date'
        return q,selection,stocks.reset_index()


def augment(definition,panel,calendar):
    result=stock_augment(definition,panel,calendar)
    assert definition['secids'][0]==SID
    surface=pd.read_parquet(TECH/'surface/assets'/f'{SID}.parquet').reindex(panel.index)
    for value in VALUES:
        observed=surface[value]
        result[f'surface_{value}_a']=observed.fillna(0.)
        result[f'surface_{value}_a_available']=observed.notna().astype(float)
        both=result[f'surface_{value}_a_available'].eq(1)&result[f'surface_{value}_b_available'].eq(1)
        result[f'surface_{value}_difference']=(result[f'surface_{value}_a']-result[f'surface_{value}_b']).where(both,0.)
        result[f'surface_{value}_difference_available']=both.astype(float)
    return result
