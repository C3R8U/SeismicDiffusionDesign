"""Compare unconstrained and CVaR-gradient-guided sampling on identical test geometries."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from advanced_models import HierarchicalDualDiT,PairedDiffusion
from hierarchy import TopologyBank
from risk_surrogate import ExceedanceNetwork
from engineering import catalog_arrays,section_role_mask,scwb_screen,repair_scwb,steel_mass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=42);ap.add_argument('--examples',type=int,default=2)
    ap.add_argument('--steps',type=int,default=12);args=ap.parse_args()
    torch.set_num_threads(3)
    data=dict(np.load(ROOT/'data/processed/designsafe_processed.npz'))
    split=json.loads((ROOT/'data/splits/split.json').read_text())
    props=catalog_arrays(ROOT/'data/catalog/section_properties_source.txt',data['section_names'])
    train=np.array(split['buildings']['train'],dtype=int)
    geom_test=np.asarray(split['buildings']['test'],dtype=int)[:args.examples]
    role=section_role_mask(data['section_idx'],train,data['mask'],len(props))
    old_mu=torch.as_tensor(data['cont_mu'],dtype=torch.float32);old_sd=torch.as_tensor(data['cont_std'],dtype=torch.float32)
    values=np.log(props)[data['section_idx']]
    mu=torch.tensor(values[train][data['mask'][train]].mean(0),dtype=torch.float32)
    sd=torch.tensor(np.maximum(values[train][data['mask'][train]].std(0),1e-5),dtype=torch.float32)
    catalog=((torch.tensor(np.log(props),dtype=torch.float32)-mu)/sd)
    role_prior=np.ones((3,len(props)),np.float32)*.001
    for r in range(3):
        idx=data['section_idx'][train,:,r][data['mask'][train,:,r]]
        role_prior[r]+=np.bincount(idx,minlength=len(props))
        role_prior[r]/=role_prior[r].sum()
    checkpoint=torch.load(ROOT/f'checkpoints/advanced_seed_{args.seed}.pt',map_location='cpu',weights_only=False)
    architecture=checkpoint['architecture']
    gen=HierarchicalDualDiT(len(props),property_dim=6,width=architecture['width'],depth=architecture['depth'],heads=8,graph_layers=3)
    gen.load_state_dict(checkpoint['network']);gen.eval()
    process=PairedDiffusion(gen,catalog,role_prior)
    risk_file=torch.load(ROOT/'checkpoints/risk_aware_surrogate.pt',map_location='cpu',weights_only=True)
    risk=ExceedanceNetwork(width=risk_file['width']);risk.load_state_dict(risk_file['model']);risk.eval()
    for param in risk.parameters():param.requires_grad_(False)
    gm=torch.tensor(data['gm'][split['ground_motions']['train'][:16]],dtype=torch.float32)
    response_mean=torch.tensor(data['response_mu'],dtype=torch.float32)
    response_std=torch.tensor(data['response_std'],dtype=torch.float32)
    thresholds=torch.tensor([.02,14.71,.01],dtype=torch.float32)
    bank=TopologyBank(data['geom_raw'])
    rows=[]
    for sample_id in geom_test:
        top=bank.take([sample_id],'cpu')
        geom=torch.tensor(data['geom'][sample_id:sample_id+1],dtype=torch.float32)
        perf_targets=np.array([.02,14.71,.01],dtype=np.float32)
        performance=torch.tensor(((np.log1p(perf_targets)-data['perf_mu'])/data['perf_std'])[None],dtype=torch.float32)
        mask=torch.tensor(data['mask'][sample_id:sample_id+1].reshape(1,57),dtype=torch.bool)
        def proxy_cvar(z):
            old=(z[...,:2]*sd[:2]+mu[:2]-old_mu)/old_sd
            x=old.expand(len(gm),-1,-1)
            mean,logvar,logits=risk(x,mask.expand(len(gm),-1),geom.expand(len(gm),-1),gm)
            response=torch.expm1(mean*response_std+response_mean).clamp(min=0)
            # A smooth performance loss retains gradients near the constraint boundary.
            excess=F.softplus(3*(response/thresholds-1))/3
            severity=(excess.square()+.1*torch.sigmoid(logits)).sum(-1)
            q=max(1,int(np.ceil(len(gm)*.10)))
            return severity.topk(q).values.mean()
        for method in ('ordinary','cvar_guided'):
            z,s=process.sample(geom,performance,mask,top,n_steps=args.steps,seed=args.seed+int(sample_id),guidance=1.5,
                               risk_fn=proxy_cvar if method=='cvar_guided' else None,risk_step=.25)
            assigned=s[0].reshape(19,3).numpy()
            repaired,checks=repair_scwb(assigned,props,data['geom_raw'][sample_id],role)
            with torch.no_grad():
                screening_score=float(proxy_cvar(z))
            rows.append({'seed':args.seed,'building_id':int(data['ids'][sample_id]),'method':method,
                         'smooth_proxy_tail_risk':screening_score,'joint_ratio_raw':scwb_screen(assigned,props,data['geom_raw'][sample_id])['min_ratio'],
                         'joint_ratio_projected':checks['min_ratio'],'joint_screen_pass':checks['screen_pass'],
                         'mass_raw_kg':steel_mass(assigned,props,data['geom_raw'][sample_id]),
                         'mass_projected_kg':steel_mass(repaired,props,data['geom_raw'][sample_id]),
                         'reference_mass_kg':steel_mass(data['section_idx'][sample_id],props,data['geom_raw'][sample_id])})
            print(json.dumps(rows[-1]),flush=True)
    out=ROOT/'results/advanced/risk_guided_designs.csv';pd.DataFrame(rows).to_csv(out,index=False)
    print('RISK_GUIDANCE_COMPLETE',str(out))

if __name__=='__main__':main()
