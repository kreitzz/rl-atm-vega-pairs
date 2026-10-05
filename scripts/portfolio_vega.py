"""Small integer portfolio allocator and daily cash-ledger helpers."""
import numpy as np
from scipy.optimize import linprog
from scripts.black_scholes import call_values

CAPITAL=1_000_000.
GROSS=20_000.
NAME=2_000.
OPTION_COMMISSION=.03

def exposures(q,spot,date,sign):
    t=(q['expiry']-date).days/365
    moves=np.repeat([-.2,0.,.2],3); vols=np.tile([-.1,0.,.1],3)
    effective=spot-q['known_dividend_pv']
    base=call_values(effective,q['strike'],t,q['bs_rate'],q['bs_iv'])[0]
    shock=call_values(np.maximum(spot*(1+moves)-q['known_dividend_pv'],1e-8),q['strike'],t,q['bs_rate'],np.maximum(.0001,q['bs_iv']+vols))[0]
    pnl=sign*100*(shock-base-q['delta']*spot*moves)
    hedge=100*q['delta']*spot
    return [100*q['vega'], hedge, 100*(q['bid']+q['ask'])/2+.5*hedge,
            max(0.,-pnl.min()), .5*100*abs(q['gamma'])*spot**2*.01**2]


def constraint_matrix(m,sign,sectors,event,nav):
    n=len(sign);scale=nav/CAPITAL
    a=[m[0],m[1],m[2],m[3],m[4]];b=[GROSS*scale,2*nav,.8*nav,.05*nav,.01*nav]
    for sector in set(sectors):
        sel=(np.asarray(sectors)==sector).astype(float)
        a.extend([m[0]*sel,m[0]*sign*sel,-m[0]*sign*sel]);b.extend([.35*GROSS*scale,.10*GROSS*scale,.10*GROSS*scale])
    a.append(m[0]*event);b.append(.25*GROSS*scale)
    return np.asarray(a),np.asarray(b)


def allocate(m,sign,sectors,event,nav,score,mode,target=None,max_contracts=None):
    """Nonnegative contract magnitudes, fixed direction; no fractional fills."""
    sign=np.asarray(sign); n=len(sign)
    if not n or nav<=0:return np.zeros(n,dtype=int)
    a,b=constraint_matrix(m,sign,sectors,event,nav)
    cap=NAME*nav/CAPITAL/m[0]
    if max_contracts is not None:cap=np.minimum(cap,np.asarray(max_contracts))
    if target is not None or mode=='equal_rank':
        desired=(np.asarray(target) if target is not None else np.full(n,GROSS*nav/CAPITAL/n))/m[0]
        # Project desired vegas to exact continuous neutrality before rounding.
        for direction in [-1,1]:
            mask=sign==direction
            if not mask.any():return np.zeros(n,dtype=int)
        actual=m[0]*desired
        smaller=min(actual[sign<0].sum(),actual[sign>0].sum())
        if smaller<=0:return np.zeros(n,dtype=int)
        for direction in [-1,1]:desired[sign==direction]*=smaller/actual[sign==direction].sum()
        fractions=np.r_[1.,cap/np.maximum(desired,1e-12),b/np.maximum(a@desired,1e-12)]
        x=desired*max(0.,min(fractions))
        if target is None and np.dot(m[0]*sign*score,x)<=np.dot(.2*m[0]+2*OPTION_COMMISSION,x):x*=0
    else:
        # Gross forecast edge less two modeled option fills/commissions.
        objective=-(sign*score*m[0]-.2*m[0]-2*OPTION_COMMISSION)
        fit=linprog(objective,A_ub=a,b_ub=b,A_eq=(m[0]*sign)[None,:],b_eq=[0.],bounds=list(zip(np.zeros(n),cap)),method='highs',options={'threads':1})
        if not fit.success:raise RuntimeError(f'Allocation failed: {fit.message}')
        x=fit.x
    qty=np.floor(x+1e-9).astype(int)
    # Only reduce exposure to remove rounding imbalance or risk breaches.
    for _ in range(int(qty.sum())+1):
        gross=m[0]@qty;net=(m[0]*sign)@qty
        breaches=a@qty>b+1e-7
        if abs(net)<=max(10.,.01*gross) and not breaches.any():return qty
        if breaches.any():
            risk=(a[breaches]/b[breaches,None]).sum(axis=0)
            candidates=np.where(qty>0)[0];k=candidates[np.argmax(risk[candidates])]
        else:
            candidates=np.where((qty>0)&(sign==np.sign(net)))[0]
            if not len(candidates):break
            k=candidates[np.argmin(abs(net-sign[candidates]*m[0,candidates]))]
        qty[k]-=1
    return np.zeros(n,dtype=int)
