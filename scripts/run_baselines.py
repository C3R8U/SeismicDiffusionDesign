"""Compare a conditional variational generator with structural library retrieval."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from baselines import ConditionalSectionVAE
from models import SeismicSurrogate
from risk import response_in_physical_units,seismic_cvar


def compute_mass(design,bid,data):
    g=data['geom_raw'][bid];stories,bays=int(g[0]),int(g[1]);span=g[2]
    h=np.array([g[3]]+[g[4]]*(stories-1))
    mass=data['section_props'][design.reshape(stories,3),0]*7850
    return float((mass[:,0]*2*h+mass[:,1]*(bays-1)*h+mass[:,2]*bays*span).sum())


def main():
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    data=dict(np.load(R/'data/processed/designsafe_processed.npz'))
    splits=json.loads((R/'data/splits/split.json').read_text())
    train=np.asarray(splits['buildings']['train']);test=np.asarray(splits['buildings']['test'])
    k=len(data['section_names']);vae=ConditionalSectionVAE(k)
    opt=torch.optim.AdamW(vae.parameters(),lr=.0005)
    history=[];best=float('inf');best_epoch=0
    for epoch in range(60):
        vae.train();losses=[]
        for chunk in np.array_split(np.random.permutation(train),14):
            if not len(chunk):continue
            ids=torch.from_numpy(data['section_idx'][chunk].reshape(-1,57)).long()
            mask=torch.from_numpy(data['mask'][chunk].reshape(-1,57))
            geom=torch.from_numpy(data['geom'][chunk]);perf=torch.from_numpy(data['perf'][chunk]);
            loss,_=vae(ids,mask,geom,perf)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(vae.parameters(),1.);opt.step();losses.append(loss.item())
        vae.eval();v=[]
        with torch.no_grad():
            for chunk in np.array_split(np.asarray(splits['buildings']['val']),4):
                ids=torch.from_numpy(data['section_idx'][chunk].reshape(-1,57)).long()
                mask=torch.from_numpy(data['mask'][chunk].reshape(-1,57))
                loss,_=vae(ids,mask,torch.from_numpy(data['geom'][chunk]),torch.from_numpy(data['perf'][chunk]))
                v.append(loss.item())
        value=float(np.mean(v))
        history.append({'epoch':epoch+1,'train_loss':np.mean(losses),'validation_loss':value})
        if value<best:
            best=value;best_epoch=epoch+1
            torch.save({'model':vae.state_dict(),'classes':k,'best_epoch':best_epoch},R/'checkpoints'/'cvae_baseline.pt')
        if (epoch+1)%10==0:print('cvae',epoch+1,round(np.mean(losses),3),round(np.mean(v),3),flush=True)
    vae.load_state_dict(torch.load(R/'checkpoints'/'cvae_baseline.pt',weights_only=True,map_location='cpu')['model'])
    vae.eval()
    pd.DataFrame(history).to_csv(R/'logs'/'cvae_training_log.csv',index=False)
    su=SeismicSurrogate(width=128)
    su.load_state_dict(torch.load(R/'checkpoints'/'surrogate_0.pt',weights_only=True,map_location='cpu')['model']);su.eval()
    scenario=splits['ground_motions']['train'][:16]
    gm=torch.from_numpy(data['gm'][scenario]);response_mean=torch.from_numpy(data['response_mu']);response_std=torch.from_numpy(data['response_std'])
    targets=np.array([.02,14.71,.01],dtype=np.float32)
    perf=(np.log1p(targets)-data['perf_mu'])/data['perf_std']
    cases=test[[0,15,35,55,75]];rows=[]
    with torch.no_grad():
        for bid in cases:
            gm_n=data['geom'][bid];geom=torch.from_numpy(gm_n.reshape(1,-1));condition=torch.from_numpy(perf.reshape(1,-1))
            mask=torch.from_numpy(data['mask'][bid].reshape(1,57));stories=int(data['geom_raw'][bid,0]);active=3*stories
            # Retrieval uses only training labels and measured training-event performance summaries.
            distances=((data['geom'][train]-gm_n)**2).sum(-1)+.25*((data['perf'][train]-perf)**2).sum(-1)
            compatible=(data['geom_raw'][train,0]==data['geom_raw'][bid,0])&(data['geom_raw'][train,1]==data['geom_raw'][bid,1])
            distances[~compatible]=np.inf
            matched=int(train[np.argmin(distances)])
            plans={'library_retrieval':data['section_idx'][matched].reshape(-1)[:active]}
            for variant in range(4):
                plans[f'conditional_vae_{variant}']=vae.generate(geom,condition,mask).numpy().reshape(-1)[:active]
            for name,sections in plans.items():
                labels=sections.astype(int)
                cont=np.log(data['section_props'][labels]);cont=(cont-data['cont_mu'])/data['cont_std']
                padded=np.zeros((57,2),dtype=np.float32);padded[:active]=cont
                pred,_=su(torch.from_numpy(padded[None,:,:]).expand(len(gm),-1,-1),
                         mask.expand(len(gm),-1),geom.expand(len(gm),-1),gm)
                physical=response_in_physical_units(pred,response_mean,response_std)
                risk=float(seismic_cvar(physical[None,:,:]).item())
                rows.append({'building_id':int(data['ids'][bid]),'method':name.split('_0')[0] if False else ('conditional_vae' if name.startswith('conditional_vae') else name),
                             'candidate':name,'proxy_cvar':risk,'frame_mass_kg':compute_mass(labels,bid,data),
                             'reference_label_match':float(np.mean(labels==data['section_idx'][bid].reshape(-1)[:active]))})
    pd.DataFrame(rows).to_csv(R/'results'/'baseline_generated_candidates.csv',index=False)
    result={'baseline_models':['conditional_vae','library_retrieval'],
            'vae_epochs':len(history),'vae_best_epoch':best_epoch,'vae_best_validation_loss':best,
            'evaluated_buildings':len(cases),
            'validation':'surrogate_screening_only; no nonlinear finite element analysis'}
    (R/'results'/'baseline_metrics.json').write_text(json.dumps(result,indent=2))
    print('Baseline evaluation saved',flush=True)

if __name__=='__main__':main()
