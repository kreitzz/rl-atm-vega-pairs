from dataclasses import replace
from scripts.curve_trading.config import config as reference
TENORS=[14,30,90]
BUCKETS=[75,'atm',25]  # Calls: lower strike, ATM, higher strike.
SLOTS=[(t,'C',b) for t in TENORS for b in BUCKETS]
COMBOS=[(t,'C',a,b) for t in TENORS for a in BUCKETS for b in BUCKETS]
LADDER=[-1.,-.75,-.5,-.25,.25,.5,.75,1.]
ACTIONS=[None]+[(combo,size) for combo in range(len(COMBOS)) for size in LADDER]
ATM=COMBOS.index((30,'C','atm','atm'))
UNIVERSES={'atm30':[ATM],
 'tenors':[i for i,c in enumerate(COMBOS) if c[2:]==('atm','atm')],
 'smile':[i for i,c in enumerate(COMBOS) if c[0]==30],
 'both':list(range(len(COMBOS)))}
RUNOFF=110
REFIT_MONTHS=3
SHORTLIST=4
MAX_EXIT_COHORTS_PER_COMBO=32
def config(commission=.65):
    return replace(reference(),min_dte=7,max_dte=105,option_commission=commission)
