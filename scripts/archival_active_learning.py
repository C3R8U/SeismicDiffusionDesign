"""Select informative published structural designs and refresh a response-risk model."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import brier_score_loss,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from risk_surrogate import ExceedanceNetwork
from train_risk_surrogate import batch_pairs,make_tensors,THRESHOLDS,infer


def fit(net,optimizer,data,building_ids,gm_ids,epochs=3,seed=42):
    pairs=batch_pairs(building_ids,gm_ids)
    rng=np.random.default_rng(seed)
    labels=(data['response'][pairs[:,0],pairs[:,1]]>THRESHOLDS).astype(float)
    pos=labels.sum(0);weights=torch.tensor(np.minimum((len(labels)-pos)/np.maximum(pos,1),30),dtype=torch.float32)
    history=[]
    for epoch in range(epochs):
        net.train();perm=rng.permutation(len(pairs));losses=[]
        for start in range(0,len(pairs),256):
            batch=pairs[perm[start:start+256]]
            x,m,g,gm,y,exceed=make_tensors(data,batch)
            a,b,c=net(x,m,g,gm)
            loss,reg,cls=net.objective(a,b,c,y,exceed,weights,lambda_exceedance=.75)
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


def predict_probabilities(net,data,bids,rids):
    net.eval(); probs=[]
    with torch.no_grad():
        for bid in bids:
            pairs=batch_pairs([bid],rids)
            x,m,g,gm,y,events=make_tensors(data,pairs)
            _,_,scores=net(x,m,g,gm)
            p=torch.sigmoid(scores)
            probs.append(p.numpy())
    return np.stack(probs)


def ensemble_selection(net,data,candidates,rids,seed):
    """Use stochastic dropout to score uncertainty near a performance boundary."""
    rng=np.random.default_rng(seed);scores=[]
    net.train()
    with torch.no_grad():
        for bid in candidates:
            pairs=batch_pairs([bid],rids)
            x,m,g,gm,y,events=make_tensors(data,pairs)
            predictions=[]
            for k in range(6):
                _,_,logits=net(x,m,g,gm)
                predictions.append(torch.sigmoid(logits).numpy())
            stacked=np.stack(predictions)
            mean=stacked.mean(0)
            variance=stacked.var(0).mean()
            boundary=np.mean(np.exp(-np.abs(mean-.5)/.15))
            scores.append(float(variance+.02*boundary))
    net.eval()
    return np.asarray(scores)


def validate(net,data,split):
    pairs=batch_pairs(split['buildings']['val'],split['ground_motions']['val'])
    _,p,y,_=infer(net,data,pairs,batch=768)
    return {'mean_brier':float(np.mean([brier_score_loss(y[:,j],p[:,j]) for j in range(3)])),
            'brier': [float(brier_score_loss(y[:,j],p[:,j])) for j in range(3)],
            'mean_ap':float(np.mean([average_precision_score(y[:,j],p[:,j]) for j in range(3)]))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rounds',type=int,default=5);ap.add_argument('--initial',type=int,default=60)
    ap.add_argument('--per-round',type=int,default=20);ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
    torch.set_num_threads(3)
    data=dict(np.load(ROOT/'data/processed/designsafe_processed.npz'))
    split=json.loads((ROOT/'data/splits/split.json').read_text())
    rng=np.random.default_rng(args.seed);order=rng.permutation(split['buildings']['train']);initial=order[:args.initial]
    output=ROOT/'results/advanced';output.mkdir(exist_ok=True,parents=True)
    records=[];history=[]
    for strategy in ['uncertainty','random']:
        torch.manual_seed(args.seed);net=ExceedanceNetwork(width=192)
        opt=torch.optim.AdamW(net.parameters(),lr=0.0005,weight_decay=.01)
        used=list(map(int,initial));remaining=list(map(int,order[args.initial:]))
        for round_number in range(args.rounds+1):
            loops=5 if round_number==0 else 2
            losses=fit(net,opt,data,used,split['ground_motions']['train'][:60],epochs=loops,seed=args.seed+round_number)
            stats=validate(net,data,split)
            records.append({'strategy':strategy,'round':round_number,'training_buildings':len(used),
                            'training_pairs':len(used)*60,'last_train_loss':losses[-1],**stats})
            print(json.dumps(records[-1]),flush=True)
            if round_number==args.rounds:break
            if strategy=='uncertainty':
                allscores=ensemble_selection(net,data,remaining,split['ground_motions']['train'][:12],args.seed+round_number)
                chosen=np.argsort(-allscores)[:args.per_round]
            else:chosen=rng.choice(len(remaining),size=args.per_round,replace=False)
            picked=[remaining[i] for i in chosen]
            for id in picked:remaining.remove(id)
            used.extend(picked)
            history.extend([{'strategy':strategy,'round':round_number+1,'building_index':idx,'building_id':int(data['ids'][idx])} for idx in picked])
        torch.save({'network':net.state_dict(),'strategy':strategy,'buildings':used,'epochs':args.rounds},ROOT/f'checkpoints/archival_active_{strategy}.pt')
    pd.DataFrame(records).to_csv(output/'archival_active_learning.csv',index=False)
    pd.DataFrame(history).to_csv(output/'archival_active_selections.csv',index=False)
    print('ARCHIVAL_ACTIVE_COMPLETE',flush=True)

if __name__=='__main__':main()
