from dataclasses import replace
from pathlib import Path
from scripts.train_expiry_contracts import settings
DATA=Path('data/curve_trading')
OUTPUT=Path('artifacts/contracts/curve_trading')
REPORT=Path('reports/curve_trading')
TENORS=[14,30,45]
BUCKETS=['atm',10,25,50,75,90]
SLOTS=[(t,cp,b) for t in TENORS for cp in ['C','P'] for b in BUCKETS]
COMBOS=[(t,cp,a,b) for t in TENORS for cp in ['C','P'] for a in BUCKETS for b in BUCKETS]
LADDER=[-1.,-.75,-.5,-.25,.25,.5,.75,1.]
ACTIONS=[None]+[(combo,size) for combo in range(len(COMBOS)) for size in LADDER]
FLAT=0
DELTA_TOLERANCE=.075
RUNOFF=60

def slot_name(t,cp,b):return f'{t}_{cp}_{b}'
def config():
    # Explicit delta-bucket selection replaces the old ATM-only 10% moneyness screen.
    return replace(settings(),min_dte=7,max_dte=55,max_moneyness_distance=1_000_000.)
