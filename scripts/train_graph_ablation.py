"""Train a matched transformer without physical graph communication."""
from __future__ import annotations
import json,sys,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_advanced_training import load_inputs
from hierarchy import TopologyBank
from advanced_models import HierarchicalDualDiT,PairedDiffusion

class WithoutPhysicalGraph(nn.Module):
    """Keep token embeddings but remove the member–node–floor–building exchanges."""
    def forward(self,features,topology,mask):
        return features*mask.unsqueeze(-1)


def main():
    torch.set_num_threads(4)
    seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    data,split,props,cont,catalog,allowed,prior=load_inputs()
    bank=TopologyBank(data['geom_raw'])
    net=HierarchicalDualDiT(len(props),property_dim=6,width=256,depth=12,heads=8,graph_layers=3,role_mask=np.ones_like(allowed))
    net.physical=WithoutPhysicalGraph()
    process=PairedDiffusion(net,catalog,prior)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.01)
    logs=[];best=1e12
    for epoch in range(1,7):
        for phase in ('train','val'):
            net.train(phase=='train')
            ids=np.asarray(split['buildings'][phase],dtype=int).copy()
            if phase=='train':np.random.shuffle(ids)
            values=[]
            for start in range(0,len(ids),32):
                bids=ids[start:start+32];top=bank.take(bids,'cpu')
                x=torch.from_numpy(cont[bids].reshape(-1,57,6));s=torch.from_numpy(data['section_idx'][bids].reshape(-1,57))
                geom=torch.from_numpy(data['geom'][bids]);perf=torch.from_numpy(data['perf'][bids]);mask=torch.from_numpy(data['mask'][bids].reshape(-1,57))
                with torch.set_grad_enabled(phase=='train'):
                    loss,details=process.objective(x,s,geom,perf,mask,top)
                    if phase=='train':
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
                values.append(float(loss.detach()))
            score=float(np.mean(values));logs.append({'epoch':epoch,'phase':phase,'loss':score})
            print(epoch,phase,round(score,6),flush=True)
            if phase=='val' and score<best:
                best=score
                torch.save({'network':net.state_dict(),'epoch':epoch,'best_validation_loss':best,'seed':seed,'width':256,'depth':12},
                           ROOT/'checkpoints/no_physical_graph_seed_42.pt')
    out=ROOT/'results/advanced';pd.DataFrame(logs).to_csv(out/'no_physical_graph_training.csv',index=False)
    (out/'no_physical_graph_metrics.json').write_text(json.dumps({'seed':seed,'epochs':6,'best_validation_loss':best,
        'parameter_count':sum(p.numel() for p in net.parameters()),'comparison':'same building split, training budget, transformer size and seed; physical graph communication removed'},indent=2))
    print('GRAPH_ABLATION_COMPLETE')
if __name__=='__main__':main()
