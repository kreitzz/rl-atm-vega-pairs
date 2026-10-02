"""Annual observed gross and cash-excess P&L at unchanged simulated sizing."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('reports/current_contract_losses');data=pd.read_csv(root/'annual_components.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,policy,label in zip(axes,['surface_filtered_rl','surface_filtered_nn'],['Filtered surface RL','Filtered surface NN']):
        g=data[data.policy.eq(policy)].sort_values('year');x=np.arange(len(g))
        gross=(g.option_mid_pnl+g.stock_hedge_gross_pnl+g.dividends)/1000
        ax.bar(x-.18,gross,width=.36,label='Gross option + hedge',color='#3876a8')
        ax.bar(x+.18,g.net_excess_pnl/1000,width=.36,label='Net versus cash',color='#d17d41')
        ax.axhline(0,color='#333333',lw=.8);ax.set_xticks(x,[str(y)+('*' if y==2025 else '') for y in g.year],rotation=35)
        ax.set_title(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('P&L, thousands of dollars');axes[1].legend(fontsize=8)
    fig.suptitle('Annual attribution at existing simulated sizes\n348 equal-budget pairs; *2025 through August',fontsize=11)
    fig.tight_layout();fig.savefig(root/'annual_attribution.png',dpi=180)
    document=Path('reports/current_contract_losses.md')
    addition='\n![Annual gross versus net P&L](current_contract_losses/annual_attribution.png)\n'
    if addition not in document.read_text():document.write_text(document.read_text()+addition)


if __name__=='__main__':main()
