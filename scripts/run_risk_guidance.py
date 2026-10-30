"""Test seismic tail-risk sensitivity on held-out frame layouts."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from models import SeismicSurrogate
from risk import guide_continuous,response_in_physical_units,seismic_cvar


def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    D=dict(np.load(R/'data/processed/designsafe_processed.npz'))
    splits=json.loads((R/'data/splits/split.json').read_text())
    net=SeismicSurrogate(width=128)
    net.load_state_dict(torch.load(R/'checkpoints'/'surrogate_0.pt',weights_only=True)['model']);net.eval()
    gm=torch.from_numpy(D['gm'][splits['ground_motions']['train'][:32]])
    mu=torch.from_numpy(D['response_mu']);std=torch.from_numpy(D['response_std'])
    risk_values=[]
    for bid in splits['buildings']['test']:
        x=torch.from_numpy(D['cont'][bid].reshape(1,57,2)); mask=torch.from_numpy(D['mask'][bid].reshape(1,57))
        geom=torch.from_numpy(D['geom'][bid].reshape(1,8))
        with torch.no_grad():
            prediction,_=net(x.expand(len(gm),-1,-1),mask.expand(len(gm),-1),geom.expand(len(gm),-1),gm)
            risk=seismic_cvar(response_in_physical_units(prediction,mu,std)[None,:,:]).item()
        risk_values.append(risk)
    bid=int(splits['buildings']['test'][int(np.argmax(risk_values))])
    original=torch.from_numpy(D['cont'][bid].reshape(1,57,2))
    mask=torch.from_numpy(D['mask'][bid].reshape(1,57))
    geom=torch.from_numpy(D['geom'][bid].reshape(1,8))
    def risk_score(z):
        with torch.no_grad():
            mean,_=net(z.expand(len(gm),-1,-1),mask.expand(len(gm),-1),geom.expand(len(gm),-1),gm)
            return float(seismic_cvar(response_in_physical_units(mean,mu,std)[None,:,:]).item())
    before=risk_score(original);z=original.clone();rows=[]
    for step in range(12):
        z,risk=guide_continuous(z,mask,geom,net,gm,mu,std,eta=.05)
        rows.append({'step':step+1,'proxy_cvar':risk_score(z),'normalized_attribute_mse_to_original':float(((z-original).square()*mask[:,:,None]).sum()/mask.sum())})
    pd.DataFrame(rows).to_csv(R/'results'/'risk_guidance_trajectory.csv',index=False)
    result={'building_id':int(D['ids'][bid]),'initial_proxy_cvar':before,'final_proxy_cvar':risk_score(z),
            'gradient_steps':len(rows),'analysis_type':'differentiable surrogate risk sensitivity only',
            'discrete_sections_updated':False,'openSees_validation':False}
    (R/'results'/'risk_guidance_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
