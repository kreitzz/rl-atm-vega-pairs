import copy
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.curve_management.config import TENORS,BUCKETS,COMBOS,ACTIONS,LADDER,REFIT_MONTHS
from scripts.curve_management.unit_paths import UnitPaths,pair_labels
from scripts.curve_management.compact_market import CompactPreparedMarket
from scripts.curve_management.engine import simulate
from scripts.curve_management.learning import exit_stops,apply_stops
from scripts.curve_trading.policy import memory_gates
from scripts.curve_management.label_store import CachedLabelStore

def load_pair(market,secids):
    qs=[];ss=[];sels=[];paths=[]
    for sid in secids:
        for root,tenors in [(Path('data/curve_trading'),[14,30]),(Path('data/curve90'),[90])]:
            parts=[]
            for year in sorted(set(market.calendar.year)):
                p=root/'selection/C'/str(year)/f'{sid}.parquet'
                if p.exists():
                    selection=pd.read_parquet(p);paths.append(p)
                    parts.append(selection[selection.tenor.isin(tenors)&selection.bucket.astype(str).isin(['75','atm','25'])&selection.cp_flag.eq('C')])
            if not parts:continue
            selected=pd.concat(parts,ignore_index=True);sels.append(selected);ids=selected.optionid.unique().tolist()
            for year in sorted(set(market.calendar.year)):
                p=root/'history/C'/str(year)/f'{sid}.parquet'
                if p.exists():qs.append(pd.read_parquet(p,filters=[('optionid','in',ids)]));paths.append(p)
        s=market.stocks[market.stocks.secid.eq(sid)].set_index('date').reindex(market.calendar)
        s['secid']=sid;s['rate']=market.rates;s.index.name='date';ss.append(s.reset_index())
    if not qs or not sels:raise ValueError('No cached contract selections')
    q=pd.concat(qs,ignore_index=True)
    duplicate=q[q.duplicated(['date','optionid'],keep=False)]
    if len(duplicate):
        for field in ['secid','expiry','cp_flag','strike','cfadj','bid','ask','delta','vega','bs_iv','bs_rate','known_dividend_pv']:
            groups=duplicate.groupby(['date','optionid'])[field]
            if pd.api.types.is_numeric_dtype(duplicate[field]):
                difference=(groups.max()-groups.min()).max()
                if difference>1e-8:raise ValueError(f'Conflicting overlapping quote caches: {field}, {difference}')
            elif groups.nunique(dropna=False).gt(1).any():raise ValueError(f'Conflicting overlapping quote caches: {field}')
    q=q.drop_duplicates(['date','optionid']).reset_index(drop=True)
    selection=pd.concat(sels,ignore_index=True)
    if selection.duplicated(['date','secid','tenor','cp_flag','bucket']).any():raise ValueError('Duplicate contract choices')
    inventory=dict(quote_rows=len(q),selection_rows=len(selection),duplicate_quote_rows=len(duplicate),
      source_files={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths})
    return q,pd.concat(ss,ignore_index=True),selection,market.divs[market.divs.secid.isin(secids)],inventory

def entry_context(panel,q,s,selection,secids):
    out=panel.copy();columns=[]
    for tenor in TENORS:
        legs=[]
        for sid in secids:
            selected=selection[selection.secid.eq(sid)&selection.tenor.eq(tenor)].copy()
            selected['bucket']=selected.bucket.astype(str)
            merged=selected[['date','optionid','bucket']].merge(q[q.secid.eq(sid)],on=['date','optionid'],validate='many_to_one')
            iv=merged.pivot(index='date',columns='bucket',values='bs_iv')*100
            atm=merged[merged.bucket.eq('atm')].set_index('date')
            spot=s[s.secid.eq(sid)].set_index('date').close
            time=(atm.expiry-atm.index.to_series()).dt.days/365
            with np.errstate(all='ignore'):
                ratio=100/(atm.bs_iv*time)*(spot.reindex(atm.index)/(spot.reindex(atm.index)-atm.known_dividend_pv))**2
            legs.append((iv,ratio))
        a,b=legs
        values={'spread':a[0].get('atm',pd.Series(dtype=float))-b[0].get('atm',pd.Series(dtype=float)),
            'wing_skew':(a[0].get('75',pd.Series(dtype=float))-a[0].get('25',pd.Series(dtype=float)))-(b[0].get('75',pd.Series(dtype=float))-b[0].get('25',pd.Series(dtype=float))),
            'gamma_gap':(a[1]-b[1])/1000}
        for name,series in values.items():
            column=f'curve_{tenor}_{name}';observed=series.reindex(panel.index).replace([np.inf,-np.inf],np.nan)
            out[column]=observed.fillna(0);out[column+'_available']=observed.notna().astype(float)
            columns.extend([column,column+'_available'])
    pd.testing.assert_frame_equal(out[panel.columns],panel)
    return out,columns

class Paths:
    def __init__(self,panel,q,s,selection,divs,cal,secids,cfg,path):
        self.panel,self.cal,self.cfg=panel,cal,cfg
        self.stop_cost=lru_cache(maxsize=4)(self._stop_cost)
        self.held_features=lru_cache(maxsize=4)(self._held_features)
        self.positions=cal.get_indexer(panel.index);self.rates=s[s.secid.eq(secids[0])].set_index('date').rate.reindex(cal).to_numpy()
        started=time.monotonic()
        self.units=[UnitPaths(q[q.secid.eq(sid)],s[s.secid.eq(sid)],selection[selection.secid.eq(sid)],cal,panel.index,
                             divs[divs.secid.eq(sid)],cfg) for sid in secids]
        print(f'  held contract paths: {time.monotonic()-started:.0f}s',flush=True)
        self.labels=CachedLabelStore(path,shape=(len(COMBOS),len(panel),8,cfg.max_dte+2))
        self.ends=np.zeros((len(COMBOS),len(panel)),int);self.gates=np.zeros((len(COMBOS),len(panel),2),bool)
        self.available=np.zeros((len(COMBOS),len(panel)),bool);self.entered=np.zeros_like(self.available)
        self.status=[]
        for i,combo in enumerate(COMBOS):
            daily,ends,vegas,status=pair_labels(*self.units,combo,cfg);self.labels[i]=daily;self.ends[i]=ends
            reward=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
                out=np.full((len(panel),2),np.nan),where=vegas[:,[0,-1]]>0)
            self.gates[i]=memory_gates(panel.index,cal[ends],reward,vegas[:,[0,-1]]>0)
            ia,ib=self.ids(i);self.available[i]=(ia>=0)&(ib>=0);self.entered[i]=(vegas[:,[0,-1]]>0).any(axis=1)
            self.status.append(dict(combo=i,**{str(k):int(v) for k,v in pd.Series(status).value_counts().items()}))
        print(f'  27 combination labels: {time.monotonic()-started:.0f}s cumulative',flush=True)
        self.rv=[]
        for sid in secids:
            stock=s[s.secid.eq(sid)].set_index('date').reindex(cal);adjusted=stock.close*stock.cfadj
            with np.errstate(all='ignore'):ret=np.log(adjusted/adjusted.shift(1))
            self.rv.append((100*np.sqrt(252*ret.pow(2).rolling(21,min_periods=21).mean())).to_numpy())
        self.context=panel[['vix_previous_close','beta','correlation']].reindex(cal).ffill(limit=31)

    def ids(self,combo):
        tenor,cp,ba,bb=COMBOS[combo]
        return self.units[0].ids(tenor,cp,ba),self.units[1].ids(tenor,cp,bb)

    def _stop_cost(self,combo):
        a,b=self.units;ia,ib=self.ids(combo)
        target=self.cfg.entry_vega_per_leg*abs(np.asarray(LADDER))
        with np.errstate(all='ignore'):
            qa=np.floor(target[None,:]/a.entry_vega[ia,None]);qb=np.floor(target[None,:]/b.entry_vega[ib,None])
            va=qa*a.entry_vega[ia,None];vb=qb*b.entry_vega[ib,None]
            accepted=(ia[:,None]>=0)&(ib[:,None]>=0)&(qa>=1)&(qb>=1)&(abs(va-vb)/np.maximum(va,vb)<=self.cfg.max_entry_vega_imbalance)
        qa=np.where(accepted,qa,0);qb=np.where(accepted,qb,0)
        return ((qa[:,:,None]*a.held['close_extra'][ia,None,:]+qb[:,:,None]*b.held['close_extra'][ib,None,:])/self.cfg.entry_vega_per_leg).astype(np.float32)

    def _held_features(self,combo):
        a,b=self.units;ia,ib=self.ids(combo);aa={k:v[ia] for k,v in a.held.items()};bb={k:v[ib] for k,v in b.held.items()}
        w=self.labels.shape[-1];now=np.minimum(self.positions[:,None]+1+np.arange(w),len(self.cal)-1)
        spread=aa['iv']-bb['iv'];start=spread[:,0,None]
        with np.errstate(all='ignore'):
            value=(self.rv[0][now]**2-aa['iv']**2)/(2*aa['iv'])-(self.rv[1][now]**2-bb['iv']**2)/(2*bb['iv'])
        columns=[np.full(spread.shape,COMBOS[combo][0]/90),aa['dte']/365,bb['dte']/365,
            aa['alive'],bb['alive'],spread,spread-start,value,(aa['gamma']-bb['gamma'])/1000,
            aa['moneyness']-bb['moneyness'],aa['vega_ratio']-bb['vega_ratio'],aa['width_vol']-bb['width_vol'],
            self.context.vix_previous_close.to_numpy()[now],self.context.beta.to_numpy()[now],self.context.correlation.to_numpy()[now]]
        raw=np.stack(columns,axis=-1).astype(np.float32);present=np.isfinite(raw)
        x=np.concatenate([np.where(present,raw,0),present.astype(np.float32)],axis=2)
        valid=(now<self.ends[combo,:,None])&(now<len(self.cal)-1)&(aa['alive'].astype(bool)|bb['alive'].astype(bool))
        valid &= self.entered[combo,:,None]
        return x,valid

    @staticmethod
    def direction_features(features,side):
        x=features.copy();direction=1. if side else -1.
        x[:,:,[5,6,7,8,9,10,11]]*=direction
        return np.concatenate([x,np.full((*x.shape[:2],1),direction,np.float32)],axis=2)

    def subset(self,combos,exit_model=None):
        n=len(self.panel);w=self.labels.shape[-1];daily=np.zeros((n,1+len(combos)*8,w),np.float32);codes=[0]
        for j,combo in enumerate(combos):
            block=self.labels[combo].copy()
            if exit_model is not None:
                held,valid=self.held_features(combo);cost=self.stop_cost(combo)
                # Only beta/correlation eligible, completed paths can reach the
                # entry fit. Other rows are zeroed by the fit and cannot deploy.
                eligible=self.panel.eligible.to_numpy(bool)&self.entered[combo]&np.isfinite(block).all(axis=(1,2))
                for side in range(2):
                    stops=exit_stops(exit_model,self.direction_features(held[eligible],side),valid[eligible],self.positions[eligible],self.ends[combo,eligible])
                    for action in (range(4) if side==0 else range(4,8)):
                        block[eligible,action]=apply_stops(block[eligible,action],cost[eligible,action],stops,self.positions[eligible],self.ends[combo,eligible],self.cal,self.rates)
            daily[:,1+j*8:1+(j+1)*8]=block;codes.extend(range(1+combo*8,1+(combo+1)*8))
        return daily,self.ends[combos].max(axis=0),codes

    def schedule(self,actions,models):
        schedule={};nonzero=actions[actions.ne(0)]
        for cutoff,model in sorted(models.items()):
            if model is None:continue
            end=(pd.Timestamp(cutoff).to_period('M')+REFIT_MONTHS).start_time
            for signal,code in nonzero.items():
                if signal>=end or signal<pd.Timestamp(cutoff)-pd.Timedelta(days=110):continue
                combo,size=ACTIONS[int(code)];row=self.panel.index.get_loc(signal)
                held,valid=self.held_features(combo);x=self.direction_features(held[row:row+1],int(size>0))[0]
                dates=self.cal[np.minimum(self.positions[row]+1+np.arange(len(x)),len(self.cal)-1)]
                use=valid[row]&(dates>=pd.Timestamp(cutoff))&(dates<end)
                if not use.any():continue
                tree,leaves,close=model;chosen=close[np.searchsorted(leaves,tree.apply(x[use]))]
                for date in dates[use][chosen]:schedule.setdefault(date,set()).add(str(signal.date()))
        return schedule

class Replay:
    def __init__(self,q,s,selection,secids,divs,cfg):
        self.q,self.s,self.secids,self.divs,self.cfg=q,s,secids,divs,cfg
        self.prepared=CompactPreparedMarket(q,s)
        self.lookup={(r.date,r.secid,int(r.tenor),r.cp_flag,str(r.bucket)):r.optionid for r in selection.itertuples()}
    def run(self,codes,calendar,schedule=None):
        assert codes.isin(range(len(ACTIONS))).all()
        sizes=pd.Series([ACTIONS[int(c)][1] if c else 0. for c in codes],index=codes.index,dtype=float)
        selected={}
        for date,code in codes[codes.ne(0)].items():
            tenor,cp,ba,bb=COMBOS[ACTIONS[int(code)][0]]
            for sid,bucket in zip(self.secids,[ba,bb]):
                oid=self.lookup.get((date,sid,tenor,cp,str(bucket)))
                if oid is not None:selected[(date,sid)]=oid
        prepared=copy.copy(self.prepared);prepared.selection_ids=selected
        selection=pd.DataFrame(columns=['date','secid','optionid']);prepared.sources=(self.q,self.s,selection)
        return simulate(self.q,self.s,sizes,calendar,self.secids,self.cfg,self.divs,selection,prepared,
                        cohort_exit_schedule=schedule)
