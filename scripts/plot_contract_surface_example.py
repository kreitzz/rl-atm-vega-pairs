"""Deterministic AMD/NVDA surface snapshot from observed selected points."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.build_contract_surface import OUTPUT as SURFACE, VALUES


def main():
    definitions=json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    d=next(d for d in definitions if d['label'].replace('_','/')=='AMD/NVDA')
    panels=[pd.read_parquet(SURFACE/'assets'/f'{sid}.parquet') for sid in d['secids']]
    dates=panels[0].dropna(subset=VALUES).index.intersection(panels[1].dropna(subset=VALUES).index)
    date=dates[dates<='2025-07-25'].max()
    if pd.isna(date):raise ValueError('No common complete snapshot')
    fig,axes=plt.subplots(1,2,figsize=(10,4));rows=[]
    for name,panel in zip(d['label'].replace('_','/').split('/'),panels):
        r=panel.loc[date]
        x=[r.lower_strike/r.atm_strike,r.atm_strike/r.atm_strike,r.upper_strike/r.atm_strike]
        y=[r.lower_iv,r.atm_iv,r.upper_iv]
        axes[0].plot(x,y,'o-',label=name)
        axes[1].plot([r.front_dte,r.atm_dte,r.back_dte],[r.front_iv,r.atm_iv,r.back_iv],'o-',label=name)
        rows.append(dict(name=name,date=str(date.date()),**{v:float(r[v]) for v in VALUES}))
    axes[0].set(xlabel='Strike / traded ATM strike',ylabel='Implied volatility (vol points)',title='Same-expiry smile points')
    axes[1].set(xlabel='Days to expiry',ylabel='ATM implied volatility (vol points)',title='Nearby-expiry ATM points')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f"AMD / NVDA — {date.date()}\nLines connect observed points; they are not a fitted surface",fontsize=11)
    fig.tight_layout()
    destination=Path('reports/contract_surface_features')
    fig.savefig(destination/'example_surface.png',dpi=180)
    pd.DataFrame(rows).to_csv(destination/'example_surface.csv',index=False)
    document=Path('reports/contract_surface_features.md')
    text='\n![Observed AMD/NVDA smile and term-structure points](contract_surface_features/example_surface.png)\n'
    if document.exists() and text not in document.read_text():document.write_text(document.read_text()+text)


if __name__=='__main__':main()
