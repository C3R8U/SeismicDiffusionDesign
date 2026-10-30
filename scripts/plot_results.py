"""Produce print-ready figures from actual observations and saved model outputs."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
F=R/'results'/'figures';F.mkdir(exist_ok=True)
D=dict(np.load(R/'data/processed/designsafe_processed.npz'))
L=pd.read_csv(R/'logs/training_log.csv')
P=dict(np.load(R/'results/surrogate_test_predictions.npz'))
C=pd.read_csv(R/'results/generated_candidates.csv')

def export(name):
    plt.tight_layout();plt.savefig(F/(name+'.png'),dpi=220,bbox_inches='tight');plt.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close()

plt.figure(figsize=(7,4))
for stage,df in L.groupby('stage'):
    plt.plot(df['epoch'],df['train_loss'],label=stage+' train')
    plt.plot(df['epoch'],df['val_loss'],linestyle='--',label=stage+' validation')
plt.xlabel('Epoch');plt.ylabel('Loss');plt.legend();plt.title('Recorded training and validation losses');export('training_losses')

labels=['Peak drift','Peak floor acceleration (m/s²)','Residual drift']
for j,name in enumerate(('peak_drift','floor_acceleration','residual_drift')):
    a=P['observed'][:,j];b=P['predicted'][:,j]
    plt.figure(figsize=(5.2,5.1));plt.hexbin(a,b,gridsize=48,bins='log',mincnt=1)
    hi=max(np.quantile(a,.995),np.quantile(b,.995));plt.plot([0,hi],[0,hi],color='black',linewidth=1)
    plt.xlim(0,hi);plt.ylim(0,hi);plt.xlabel('Observed '+labels[j]);plt.ylabel('Predicted '+labels[j]);plt.title('Held-out structure and record pairs');
    plt.colorbar(label='log10 count');export('surrogate_'+name)

plt.figure(figsize=(7,4))
plt.hist(C['proxy_cvar'],bins=20,edgecolor='black');plt.xlabel('Proxy empirical CVaR');plt.ylabel('Generated candidate count');plt.title('Surrogate-screened generated designs');export('proxy_risk_distribution')

examples=json.loads((R/'results/generated_designs.json').read_text()); item=max((e for e in examples if e['candidate']==0),key=lambda e:e['num_stories'])
n=item['num_stories'];reference=np.array(item['reference_sections']).reshape(n,3);generated=np.array(item['sections']).reshape(n,3)
fig,ax=plt.subplots(1,2,figsize=(8,5),sharey=True)
for pane,arr,title in zip(ax,[reference,generated],['Reference design','Diffusion sample']):
    image=pane.imshow(arr,interpolation='nearest',aspect='auto',origin='lower',cmap='viridis')
    pane.set_title(title);pane.set_xticks(range(3),['Exterior','Interior','Beam']);pane.set_xlabel('Design group')
    pane.set_ylabel('Story index (bottom up)')
fig.colorbar(image,ax=ax.tolist(),label='W-section catalog index',fraction=.035,pad=.05)
fig.savefig(F/'section_layout_comparison.png',dpi=220,bbox_inches='tight');fig.savefig(F/'section_layout_comparison.pdf',bbox_inches='tight');plt.close(fig)

plt.figure(figsize=(7,4))
source=pd.read_csv(R/'data/processed/design_summary.csv')
for stories,group in source.groupby('num_stories'):
    plt.scatter(group['span_m'],group['p90_drift'],s=12,alpha=.45,label=f'{stories} stories')
plt.xlabel('Span width (m)');plt.ylabel('90th percentile peak drift');plt.legend(ncol=2);plt.title('Observed DesignSafe design response distribution');export('observed_response_distribution')

plt.figure(figsize=(7,4))
plt.scatter(C['frame_mass_kg'],C['proxy_cvar'],c=C['building_id'],alpha=.75)
plt.xlabel('Estimated 2D frame steel mass (kg)');plt.ylabel('Predicted CVaR');plt.title('Generated designs: proxy risk versus section mass');export('generated_proxy_risk_mass')
print('Saved figures to',F)

B=pd.read_csv(R/'results/baseline_generated_candidates.csv')
D=pd.read_csv(R/'logs/cvae_training_log.csv')
plt.figure(figsize=(7,4))
plt.plot(D['epoch'],D['train_loss'],label='CVAE train')
plt.plot(D['epoch'],D['validation_loss'],label='CVAE validation')
plt.xlabel('Epoch');plt.ylabel('Conditional VAE objective');plt.legend();plt.title('Conditional VAE training behavior');export('cvae_training')

methods={'Diffusion':C['frame_mass_kg'].to_numpy(),
         'Conditional VAE':B[B.method=='conditional_vae']['frame_mass_kg'].to_numpy(),
         'Training library':B[B.method=='library_retrieval']['frame_mass_kg'].to_numpy()}
plt.figure(figsize=(7.5,4.2))
plt.boxplot(list(methods.values()),tick_labels=list(methods),showfliers=False)
plt.ylabel('Estimated 2D frame steel mass (kg)');plt.title('Candidate design masses by generator');export('generator_mass_comparison')

G=pd.read_csv(R/'results/risk_guidance_trajectory.csv')
plt.figure(figsize=(7,4))
plt.plot([0]+G['step'].tolist(),[json.loads((R/'results/risk_guidance_metrics.json').read_text())['initial_proxy_cvar']]+G['proxy_cvar'].tolist(),marker='o')
plt.xlabel('Guidance step');plt.ylabel('Surrogate CVaR');plt.title('Local differentiable risk sensitivity');export('risk_guidance_trace')

raw=np.load(R/'data/processed/designsafe_processed.npz')
metadata=json.loads((R/'data/splits/split.json').read_text())
test_ids=np.array(metadata['buildings']['test'])
test_pair_idx=P['paired_indices']
story_counts=raw['geom_raw'][test_pair_idx[:,0],0]
plt.figure(figsize=(7,4))
for stories in np.unique(story_counts):
    subset=story_counts==stories
    error=np.abs(P['predicted'][subset,0]-P['observed'][subset,0])
    plt.scatter([stories],[np.mean(error)],s=65)
plt.xlabel('Story count');plt.ylabel('Mean absolute peak-drift error');plt.title('Surrogate error by building height');export('surrogate_error_by_stories')

A=pd.read_csv(R/'logs/ablation_training_log.csv')
plt.figure(figsize=(7,4.2))
for variant,part in A.groupby('variant'):
    plt.plot(part['epoch'],part['heldout_validation_full_objective'],marker='o',label=variant.replace('_',' '))
plt.xlabel('Diagnostic training epoch');plt.ylabel('Shared validation objective');plt.legend(fontsize=8)
plt.title('Short-budget architecture diagnostics');export('ablation_short_budget')
