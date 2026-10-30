"""Compare hierarchical message passing, property coupling, and performance conditions."""
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from models import DualDiffusion,DualDiffusionProcess

class FlatContext(nn.Module):
    def forward(self,features,mask):
        return features*mask[:,:,None]


def train_one(data,groups,variant,epochs=4):
    torch.manual_seed(1234);np.random.seed(1234)
    model=DualDiffusion(130,dim=96,layers=3,heads=4,prior=data['prior'],allowed=data['observed_roles'])
    if variant=='flat_features':model.graph=FlatContext()
    proc=DualDiffusionProcess(model)
    proc.set_catalog((np.log(data['section_props'])-data['cont_mu'])/data['cont_std'])
    opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
    hist=[]
    for epoch in range(epochs):
        model.train();losses=[]
        for ids in np.array_split(np.random.permutation(groups['train']),14):
            if not len(ids):continue
            design=torch.from_numpy(data['cont'][ids].reshape(-1,57,2))
            classes=torch.from_numpy(data['section_idx'][ids].reshape(-1,57))
            mask=torch.from_numpy(data['mask'][ids].reshape(-1,57))
            geom=torch.from_numpy(data['geom'][ids]);perf=torch.from_numpy(data['perf'][ids]).clone()
            if variant=='no_performance_condition':perf*=0
            loss,_=proc.objective(design,classes,geom,perf,mask,
                                 lambda_physics=0 if variant=='no_property_coupling' else .2)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(loss.item())
        model.eval();valid=[]
        # Identical random schedules keep each validation estimate comparable.
        torch.manual_seed(404+epoch)
        with torch.no_grad():
            for ids in np.array_split(groups['val'],4):
                loss,_=proc.objective(torch.from_numpy(data['cont'][ids].reshape(-1,57,2)),
                      torch.from_numpy(data['section_idx'][ids].reshape(-1,57)),torch.from_numpy(data['geom'][ids]),
                      torch.from_numpy(data['perf'][ids]),torch.from_numpy(data['mask'][ids].reshape(-1,57)),
                      lambda_physics=.2)
                valid.append(loss.item())
        hist.append({'variant':variant,'epoch':epoch+1,'train_objective':float(np.mean(losses)),
                     'heldout_validation_full_objective':float(np.mean(valid))})
    return hist


def main():
    torch.set_num_threads(4)
    data=dict(np.load(R/'data/processed/designsafe_processed.npz'))
    spl=json.loads((R/'data/splits/split.json').read_text())['buildings']
    result=[]
    for v in ('complete','flat_features','no_property_coupling','no_performance_condition'):
        print('Running',v,flush=True)
        result+=train_one(data,spl,v)
    frame=pd.DataFrame(result)
    frame.to_csv(R/'logs'/'ablation_training_log.csv',index=False)
    summary=frame.groupby('variant',as_index=False).tail(1)[['variant','heldout_validation_full_objective']]
    summary.to_csv(R/'results'/'ablation_validation.csv',index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
