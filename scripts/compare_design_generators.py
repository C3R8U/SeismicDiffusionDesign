"""Screen matched generated candidates with one frozen response-risk model."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baselines import ConditionalSectionVAE
from risk_surrogate import ExceedanceNetwork


def mass(labels,geom,section_properties):
    n,bays=int(geom[0]),int(geom[1]); span=float(geom[2]);height=np.array([geom[3]]+[geom[4]]*(n-1))
    groups=section_properties[np.asarray(labels).reshape(n,3),0]*7850
    return float(np.sum(2*groups[:,0]*height+(bays-1)*groups[:,1]*height+bays*groups[:,2]*span))


def main():
    torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026)
    d=dict(np.load(ROOT/'data/processed/designsafe_processed.npz'))
    splits=json.loads((ROOT/'data/splits/split.json').read_text())
    vae_state=torch.load(ROOT/'checkpoints/cvae_baseline.pt',map_location='cpu',weights_only=True)
    vae=ConditionalSectionVAE(vae_state['classes']);vae.load_state_dict(vae_state['model']);vae.eval()
    risk_state=torch.load(ROOT/'checkpoints/risk_aware_surrogate.pt',map_location='cpu',weights_only=True)
    risk=ExceedanceNetwork(risk_state['width']);risk.load_state_dict(risk_state['model']);risk.eval()
    saved=json.loads((ROOT/'results/generated_designs.json').read_text())
    catalog=(np.log(d['section_props'])-d['cont_mu'])/d['cont_std']
    train=np.asarray(splits['buildings']['train']);events=np.asarray(splits['ground_motions']['test'])
    rows=[]
    for building_id in sorted(set(int(x['building_id']) for x in saved)):
        index=int(np.where(d['ids']==building_id)[0][0]); geometry=d['geom_raw'][index]
        n=int(geometry[0]); active=3*n; mask=d['mask'][index].reshape(57)
        training_candidates = train[(d['geom_raw'][train,0]==n)&(d['geom_raw'][train,1]==geometry[1])]
        similarity=np.sum((d['geom'][training_candidates]-d['geom'][index])**2,axis=1)
        nearest=training_candidates[np.argsort(similarity)[:4]]
        geom=torch.from_numpy(np.repeat(d['geom'][index:index+1],4,axis=0))
        target=(np.log1p(np.array([.02,14.71,.01],dtype=np.float32))-d['perf_mu'])/d['perf_std']
        perf=torch.from_numpy(np.tile(target,(4,1)).astype('float32'))
        valid_mask=torch.from_numpy(np.repeat(mask[None],4,axis=0))
        with torch.no_grad():
            vae_labels=vae.generate(geom,perf,valid_mask).numpy()
        candidates=[]
        for item in saved:
            if int(item['building_id'])==building_id:
                candidates.append(('graph_diffusion',int(item['candidate']),np.asarray(item['sections'],dtype=int)))
        for c in range(4):
            candidates.append(('conditional_vae',c,vae_labels[c,:active]))
        for c,i in enumerate(nearest):
            candidates.append(('design_retrieval',c,d['section_idx'][i].reshape(-1)[:active]))
        reference=d['section_idx'][index].reshape(-1)[:active]
        reference_mass=mass(reference,geometry,d['section_props'])
        for method,candidate,labels in candidates:
            features=np.zeros((57,2),dtype=np.float32);features[:active]=catalog[labels]
            batch=len(events)
            with torch.inference_mode():
                _,_,logit=risk(torch.from_numpy(np.repeat(features[None],batch,axis=0)),
                    torch.from_numpy(np.repeat(mask[None],batch,axis=0)),
                    torch.from_numpy(np.repeat(d['geom'][index:index+1],batch,axis=0)),
                    torch.from_numpy(d['gm'][events]))
            probs=torch.sigmoid(logit).numpy()
            m=mass(labels,geometry,d['section_props'])
            rows.append({'building_id':building_id,'method':method,'candidate':candidate,
                 'nominal_mass_kg':m,'reference_nominal_mass_kg':reference_mass,
                 'mass_change_percent':100*(m/reference_mass-1),
                 'peak_drift_exceedance_score':float(probs[:,0].mean()),
                 'floor_acceleration_exceedance_score':float(probs[:,1].mean()),
                 'residual_drift_exceedance_score':float(probs[:,2].mean()),
                 'mean_exceedance_score':float(probs.mean()),
                 'role_label_seen_in_training_fraction':float(np.mean([labels[j] in
                  d['section_idx'][train,:,j%3][d['mask'][train,:,j%3]] for j in range(active)])),
                 'verified_structural_compliance':False})
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/'results/matched_generator_comparison.csv',index=False)
    summary=out.groupby('method').agg(candidates=('candidate','size'),
              mean_mass_change_percent=('mass_change_percent','mean'),
              mean_exceedance_score=('mean_exceedance_score','mean'),
              median_exceedance_score=('mean_exceedance_score','median'),
              mean_role_catalog_coverage=('role_label_seen_in_training_fraction','mean')).reset_index()
    summary.to_csv(ROOT/'results/matched_generator_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5.2))
    for method,group in out.groupby('method'):
        ax.scatter(group['mass_change_percent'],group['mean_exceedance_score'],alpha=.6,s=40,label=method)
    ax.set_xlabel('Steel mass change relative to published design (%)')
    ax.set_ylabel('Rare-event predictor exceedance score')
    ax.set_title('Matched design screening: five buildings, four candidates per method')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(ROOT/'results/audit_figures/matched_generator_tradeoff.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.5))
    for method,group in out.groupby('method'):
        axs[0].hist(group['mean_exceedance_score'],bins=10,alpha=.45,label=method)
        axs[1].hist(group['mass_change_percent'],bins=10,alpha=.45,label=method)
    axs[0].set_xlabel('Predicted exceedance score');axs[1].set_xlabel('Mass change (%)')
    axs[0].set_ylabel('Candidates');axs[0].legend()
    fig.tight_layout();fig.savefig(ROOT/'results/audit_figures/matched_generator_distributions.png',dpi=200);plt.close(fig)
    print(summary.to_string(index=False),flush=True)
    (ROOT/'results/matched_generator_status.json').write_text(json.dumps({'test_buildings':5,'candidates_per_method':20,
       'test_ground_motion_records_per_candidate':len(events),
       'generated_response_source':'frozen multitask response-risk predictor',
       'nonlinear_seismic_compliance':'not evaluated',
       'catalog_constraint':'known section IDs only; detailed design constraints not certified',
       'risk_score':'uncalibrated classifier output; not an estimate of CVaR'},indent=2))

if __name__=='__main__': main()
