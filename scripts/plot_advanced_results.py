"""Figures for hierarchical model optimization and engineering screening."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def save(fig,name):
    out=ROOT/'results/advanced/figures';out.mkdir(exist_ok=True,parents=True)
    fig.savefig(out/(name+'.png'),dpi=210,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def plot_training():
    p=ROOT/'logs/advanced/full_dit_training.csv'
    if not p.exists():return
    df=pd.read_csv(p)
    fig,ax=plt.subplots(figsize=(8,5))
    for (seed,phase),rows in df.groupby(['seed','phase']):
        ax.plot(rows.epoch,rows.loss,marker='o',label=f'{seed} / {phase}')
    ax.set(xlabel='Epoch',ylabel='Training objective',title='Hierarchical 12-block diffusion: independent initializations')
    ax.grid(alpha=.2);ax.legend(ncol=2);save(fig,'hierarchical_diffusion_training')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for axis,key in zip(axes,['continuous','categorical','consistency']):
        for seed,grp in df[df.phase=='val'].groupby('seed'):
            axis.plot(grp.epoch,grp[key],marker='o',label=str(seed))
        axis.set(xlabel='Epoch',ylabel=key,title=f'Validation: {key}')
        axis.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();save(fig,'diffusion_loss_components')


def plot_catalog_and_joints():
    df=pd.read_csv(ROOT/'results/advanced/reference_scwb_screens.csv')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4))
    axes[0].hist(df.min_scwb_plastic_ratio,bins=32,edgecolor='white')
    axes[0].axvline(1,color='red',linestyle='--',label='Plastic-moment ratio = 1.0')
    axes[0].set(xlabel='Minimum joint plastic-modulus ratio',ylabel='Reference buildings',title='Published design: simplified joint screening')
    axes[0].legend()
    df.boxplot(column='min_scwb_plastic_ratio',by='stories',ax=axes[1])
    axes[1].set(xlabel='Stories',ylabel='Minimum joint ratio',title='Ratio by building height')
    fig.suptitle('');fig.tight_layout();save(fig,'reference_joint_screening')
    paths=list((ROOT/'results/advanced').glob('generated_seed_*.json'))
    if not paths:return
    rows=[]
    for p in paths:
        for item in json.loads(p.read_text()):
            rows.append({'seed':item['seed'],'mass_change':100*(item['mass_repaired']/item['mass_reference']-1),
                         'raw':item['screen_raw'],'repaired':item['screen_repaired'],'reference':item['screen_reference']})
    pd.DataFrame(rows).to_csv(ROOT/'results/advanced/engineering_screening_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4))
    x=np.arange(len(rows))
    axes[0].plot(x,[z['raw'] for z in rows],'o',label='Raw diffusion')
    axes[0].plot(x,[z['repaired'] for z in rows],'s',label='After column adjustment')
    axes[0].axhline(1,color='gray',ls='--')
    axes[0].set(xlabel='Generated candidate',ylabel='Minimum joint ratio',title='Simplified strong-column screen')
    axes[0].legend()
    axes[1].bar(x,[z['mass_change'] for z in rows])
    axes[1].axhline(0,color='gray',ls='--')
    axes[1].set(xlabel='Generated candidate',ylabel='Mass relative to reference (%)',title='Mass after section adjustment')
    fig.tight_layout();save(fig,'candidate_joint_and_mass_screening')


def plot_active():
    p=ROOT/'results/advanced/archival_active_learning.csv'
    if not p.exists():return
    df=pd.read_csv(p)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method,g in df.groupby('strategy'):
        axes[0].plot(g.training_buildings,g.mean_brier,marker='o',label=method)
        axes[1].plot(g.training_buildings,g.mean_ap,marker='o',label=method)
    axes[0].set(xlabel='Published training buildings labeled',ylabel='Validation mean Brier',title='Risk probability calibration')
    axes[1].set(xlabel='Published training buildings labeled',ylabel='Validation mean AP',title='Rare-event ranking')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();save(fig,'archival_active_learning')


def plot_risk():
    p=ROOT/'results/advanced/risk_guided_designs.csv'
    if not p.exists():return
    df=pd.read_csv(p)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,method in enumerate(sorted(df.method.unique())):
        rows=df[df.method==method]
        ax.scatter([i]*len(rows),rows.smooth_proxy_tail_risk,s=60,label=method)
    ax.set_xticks(range(len(df.method.unique())),sorted(df.method.unique()))
    ax.set(ylabel='Original regression proxy CVaR',title='Risk-guided sampling diagnostic; no nonlinear validation')
    ax.grid(alpha=.2);save(fig,'risk_guidance_diagnostic')

if __name__=='__main__':
    plot_training();plot_catalog_and_joints();plot_active();plot_risk()
