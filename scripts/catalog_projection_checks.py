"""Apply a role-aware training catalog to candidate section arrays."""
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
from risk_surrogate import ExceedanceNetwork


def main():
    torch.set_num_threads(4)
    d=dict(np.load(ROOT/'data/processed/designsafe_processed.npz'))
    s=json.loads((ROOT/'data/splits/split.json').read_text())
    designs=json.loads((ROOT/'results/generated_designs.json').read_text())
    train=np.asarray(s['buildings']['train']);gm=torch.from_numpy(d['gm'][s['ground_motions']['test']])
    section_properties=d['section_props']
    catalog=(np.log(section_properties)-d['cont_mu'])/d['cont_std']
    role_allowed=np.zeros((3,len(d['section_names'])),dtype=bool)
    for role in range(3):
        ids=d['section_idx'][train,:,role][d['mask'][train,:,role]]
        role_allowed[role,np.unique(ids)]=True
    state=torch.load(ROOT/'checkpoints/risk_aware_surrogate.pt',map_location='cpu',weights_only=True)
    predictor=ExceedanceNetwork(state['width']);predictor.load_state_dict(state['model']);predictor.eval()
    out=[]
    for entry in designs:
        bid=int(entry['building_id']);i=int(np.where(d['ids']==bid)[0][0])
        labels=np.asarray(entry['sections'],dtype=int)
        projected=labels.copy()
        for slot in range(len(labels)):
            role=slot%3
            if role_allowed[role,labels[slot]]:continue
            legal=np.flatnonzero(role_allowed[role])
            projected[slot]=legal[np.sum((catalog[legal]-catalog[labels[slot]])**2,axis=1).argmin()]
        n=len(labels);mask=d['mask'][i].reshape(57)
        g=d['geom_raw'][i];bays=int(g[1]);stories=int(g[0]);heights=np.array([g[3]]+[g[4]]*(stories-1))
        lengths=np.column_stack([2*heights,(bays-1)*heights,np.full(stories,bays*g[2])]).reshape(-1)
        result={'building_id':bid,'candidate':int(entry['candidate']),
                'roles_modified':int(np.sum(projected!=labels)),
                'role_fraction_before':float(np.mean([role_allowed[j%3,label] for j,label in enumerate(labels)])),
                'role_fraction_after':float(np.mean([role_allowed[j%3,label] for j,label in enumerate(projected)]))}
        for key,chosen in [('before',labels),('after',projected)]:
            feat=np.zeros((57,2),np.float32);feat[:n]=catalog[chosen]
            batch=len(gm)
            with torch.inference_mode():
                _,_,logits=predictor(torch.from_numpy(np.repeat(feat[None],batch,axis=0)),
                    torch.from_numpy(np.repeat(mask[None],batch,axis=0)),
                    torch.from_numpy(np.repeat(d['geom'][i:i+1],batch,axis=0)),gm)
            result['risk_score_'+key]=float(torch.sigmoid(logits).mean().item())
            result['mass_kg_'+key]=float(np.sum(section_properties[chosen,0]*7850*lengths))
        out.append(result)
    records=pd.DataFrame(out)
    records.to_csv(ROOT/'results/role_catalog_projection.csv',index=False)
    report={'candidate_count':len(records),
            'changed_candidates':int((records['roles_modified']>0).sum()),
            'mean_modified_roles':float(records['roles_modified'].mean()),
            'mean_training_role_coverage_before':float(records['role_fraction_before'].mean()),
            'mean_training_role_coverage_after':float(records['role_fraction_after'].mean()),
            'mean_mass_change_due_to_projection_percent':float((100*(records['mass_kg_after']/records['mass_kg_before']-1)).mean()),
            'mean_proxy_risk_score_change':float((records['risk_score_after']-records['risk_score_before']).mean()),
            'disclaimer':'Role labels are restricted to sections seen in training for that role; full code compliance is not guaranteed.'}
    (ROOT/'results/role_projection_metrics.json').write_text(json.dumps(report,indent=2))
    plt.figure(figsize=(7.5,4.8))
    plt.scatter(records['risk_score_before'],records['risk_score_after'],c=records['roles_modified'],s=45)
    lim=float(max(records[['risk_score_before','risk_score_after']].max()))
    plt.plot([0,lim],[0,lim],color='gray',ls='--')
    plt.xlabel('Before role-aware projection');plt.ylabel('After role-aware projection')
    plt.title('Effect of section-role screening on surrogate risk score')
    plt.colorbar(label='Modified section groups');plt.tight_layout()
    plt.savefig(ROOT/'results/audit_figures/role_projection_risk.png',dpi=200);plt.close()
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
