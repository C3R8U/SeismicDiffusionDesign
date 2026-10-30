"""Generate designs with the response predictor inside the reverse diffusion loop."""
import json,sys
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from models import DualDiffusion,DualDiffusionProcess,SeismicSurrogate
from risk import project_sections


def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    d=dict(np.load(R/'data/processed/designsafe_processed.npz'))
    splits=json.loads((R/'data/splits/split.json').read_text())
    state=torch.load(R/'checkpoints'/'diffusion_best.pt',weights_only=False,map_location='cpu')
    model=DualDiffusion(state['classes'],state['dim'],state['layers'],state['heads'],prior=state['prior'],allowed=state['allowed'])
    model.load_state_dict(state['model']);model.eval()
    sampler=DualDiffusionProcess(model)
    catalog=(np.log(d['section_props'])-d['cont_mu'])/d['cont_std']
    sampler.set_catalog(catalog)
    predictor=SeismicSurrogate(width=128)
    predictor.load_state_dict(torch.load(R/'checkpoints'/'surrogate_0.pt',weights_only=True,map_location='cpu')['model'])
    predictor.eval()
    idx=int(splits['buildings']['test'][35]);num=2
    geom=torch.from_numpy(d['geom'][[idx]*num]);mask=torch.from_numpy(d['mask'][[idx]*num].reshape(num,57))
    performance=(np.log1p(np.array([.02,14.71,.01],dtype=np.float32))-d['perf_mu'])/d['perf_std']
    condition=torch.from_numpy(np.tile(performance,(num,1)))
    earthquake=torch.from_numpy(d['gm'][splits['ground_motions']['train'][:8]])
    mu=torch.from_numpy(d['response_mu']);std=torch.from_numpy(d['response_std'])
    x,sections=sampler.generate(geom,condition,mask,num_steps=8,seed=1024,surrogate=predictor,
                                ground_motions=earthquake,response_mean=mu,response_std=std,
                                guidance_step=.05)
    projected=project_sections(x,model.allowed,sampler.catalog,mask)
    output={'building_id':int(d['ids'][idx]),'samples':projected.numpy().tolist(),
            'sampling_steps':8,'ground_motion_scenarios':8,
            'guidance':'differentiable proxy CVaR gradient and continuous-attribute mass term',
            'validation':'section catalog projection only; no nonlinear finite element analysis'}
    (R/'results'/'guided_samples.json').write_text(json.dumps(output,indent=2))
    print('Saved guided samples for building',output['building_id'])

if __name__=='__main__':main()
