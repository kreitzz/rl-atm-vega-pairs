"""Explicit spread boundaries and independently switchable entry caps."""
import numpy as np
from scripts.same_structure_smile import oriented_states
from scripts.option_flex.contract_risk import risk_breaches

BOUNDARIES=[20,30,40]
RISK_MODES=['current','no_stress','no_caps']


def states(raw, boundary):
    if boundary not in BOUNDARIES:raise ValueError('Undeclared spread boundary')
    x=raw.copy();tail=boundary/100
    x['rule_direction']=np.where(x.iv_rank.le(tail),1,np.where(x.iv_rank.gt(1-tail),-1,0))
    return oriented_states(x)


def breaches(metrics,nav,cfg,mode):
    if mode not in RISK_MODES:raise ValueError('Undeclared entry-cap configuration')
    reasons=risk_breaches(metrics,nav,cfg)
    if mode=='no_stress':return [r for r in reasons if r!='stress_loss']
    if mode=='no_caps':return [r for r in reasons if r=='nonpositive_nav']
    return reasons
