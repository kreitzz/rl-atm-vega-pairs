from dataclasses import replace
from pathlib import Path
from scripts.train_expiry_contracts import settings
DATA=Path('data/curve90')
OUTPUT=Path('artifacts/contracts/curve90')
REPORT=Path('reports/curve90')
TENORS=[90]
BUCKETS=['atm',10,25,50,75,90]
SLOTS=[(t,cp,b) for t in TENORS for cp in ['C','P'] for b in BUCKETS]
COMBOS=[(t,cp,a,b) for t in TENORS for cp in ['C','P'] for a in BUCKETS for b in BUCKETS]
LADDER=[-1.,-.75,-.5,-.25,.25,.5,.75,1.]
ACTIONS=[None]+[(combo,size) for combo in range(len(COMBOS)) for size in LADDER]
FLAT=0
DELTA_TOLERANCE=.075
RUNOFF=110

def slot_name(t,cp,b):return f'{t}_{cp}_{b}'
def config():
    # Explicit delta-bucket selection replaces the old ATM-only 10% moneyness screen.
    return replace(settings(),target_dte=90,min_dte=75,max_dte=105,max_moneyness_distance=1_000_000.)
