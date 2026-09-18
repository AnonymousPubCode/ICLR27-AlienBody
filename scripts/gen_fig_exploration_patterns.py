#!/usr/bin/env python3
"""Figure (appendix): three Phase-1 exploration patterns on Family 1.

Cumulative unique actions discovered vs. Phase-1 steps, reconstructed from
the measured (P1, EC) of each agent:
  Premature  (Qwen3.5-4B,  image): P1 ~ 0.9, EC ~ 0.24 -> 1 step, 1 action
  Redundant  (Qwen3.5-397B, image): P1 ~ 16.6, EC ~ 0.85 -> all 4 after ~4 re-tests each
  Structured (Systematic):          P1 = 4, EC = 1.00     -> one new action per step

Output: fig/exploration_patterns.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from figstyle import *  # noqa: F401,F403

PREMATURE = ([0, 1, 2, 3], [0, 1, 1, 1])
REDUNDANT = (list(range(0, 18)),
             [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4])
STRUCTURED = ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
BUDGET = 20


def build():
    W,H=5.5,1.76
    fig=plt.figure(figsize=(W,H),facecolor='white')
    ax=fig.add_axes([.34/W,.36/H,3.21/W,1.06/H])
    ax.set(xlim=(-.25,20.6),ylim=(-.12,4.32),xticks=[0,4,8,12,16,20],yticks=range(5))
    ax.yaxis.grid(True,color='#D8DDDF',lw=.45,zorder=0)
    for edge in ('top','right','left'):
        ax.spines[edge].set_visible(False)
    ax.spines['bottom'].set(color='#7D878C',linewidth=.55)
    ax.tick_params(axis='both',length=0,pad=3,labelsize=6.3)
    ax.set_xlabel('Phase-1 step',fontsize=6.6,labelpad=3)
    fig.text(.05/W,1.61/H,'Unique actions tested',fontsize=7.5,weight='bold',va='center')
    fig.text(3.80/W,1.61/H,'Illustrative patterns',fontsize=7.5,weight='bold',va='center')
    ax.axvline(4,color='#B3B9BD',ls='--',lw=.65)
    ax.axvline(BUDGET,color='#B3B9BD',ls=':',lw=.65)
    entries=[(PREMATURE,RED,'o',':','Premature','Qwen3.5-4B'),
             (REDUNDANT,C_GPT4O,'s','--','Redundant','Qwen3.5-397B'),
             (STRUCTURED,GREEN,'^','-','Structured','Systematic')]
    for k,((xs,ys),c,m,ls,name,model) in enumerate(entries):
        ax.step(xs,ys,where='post',color=c,lw=1.2,ls=ls,zorder=3)
        ax.plot(xs[-1],ys[-1],marker=m,color=c,ms=3.7,mfc='white',mew=1,zorder=4)
        y=1.28-k*.38
        fig.add_artist(plt.Line2D([3.83/W,3.96/W,4.09/W],[y/H]*3,
            transform=fig.transFigure,color=c,ls=ls,lw=1.2,marker=m,markevery=[1],
            ms=3.7,mfc='white',mew=1))
        fig.text(4.18/W,y/H,name,fontsize=6.8,color=INK,weight='bold',va='center')
        fig.text(4.18/W,(y-.14)/H,model,fontsize=6.2,color=MUTED,va='center')
    fig.text(3.80/W,.13/H,'4 actions | 20-step budget',fontsize=6.2,color=MUTED,va='center')
    out=figure_out_dir();out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'exploration_patterns.pdf')
    fig.savefig(out/'exploration_patterns.png',dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    build()
