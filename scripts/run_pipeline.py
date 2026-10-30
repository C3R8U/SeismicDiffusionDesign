"""Train structural generators and response models, then export quantitative checks."""
from __future__ import annotations
import argparse, csv, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from preprocessing import process_archive
from models import DualDiffusion, DualDiffusionProcess, SeismicSurrogate, parameters_count
from risk import guide_continuous, project_sections, response_in_physical_units, seismic_cvar


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(4,os.cpu_count() or 4))


def read_data():
    return dict(np.load(ROOT/'data/processed/designsafe_processed.npz',allow_pickle=False)),json.loads((ROOT/'data/splits/split.json').read_text())


def batches(indices,batch,shuffle=True):
    if shuffle: indices=np.random.permutation(indices)
    for i in range(0,len(indices),batch):yield indices[i:i+batch]


def tensor(x,device='cpu',dtype=None):
    return torch.as_tensor(x,dtype=dtype,device=device)


def pair_indices(bids,rids):
    return np.stack(np.meshgrid(np.asarray(bids),np.asarray(rids),indexing='ij'),axis=-1).reshape(-1,2)


def train_surrogate(data,split,quick,device,run_dir,seed):
    g=split['ground_motions'];b=split['buildings']
    pairs={name:pair_indices(b[name],g[name]) for name in ('train','val','test')}
    # A small pilot samples only training pairs; validation and test partitions remain fixed.
    if quick and len(pairs['train'])>18000:
        pairs['train']=pairs['train'][np.random.permutation(len(pairs['train']))[:18000]]
    models=[];hist=[]
    members=1 if quick else 5
    epochs=9 if quick else 200
    batch=512 if quick else 256
    for member in range(members):
        net=SeismicSurrogate(width=128 if quick else 256).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        best=float('inf');bad=0
        for epoch in range(epochs):
            tick=time.time();net.train();total=0.; n=0
            for rows in batches(np.arange(len(pairs['train'])),batch):
                indices=pairs['train'][rows];bi=indices[:,0];gi=indices[:,1]
                x=tensor(data['cont'][bi].reshape(-1,57,2),device).float()
                mask=tensor(data['mask'][bi].reshape(-1,57),device).bool()
                geom=tensor(data['geom'][bi],device).float();gm=tensor(data['gm'][gi],device).float()
                y=tensor(data['response_norm'][bi,gi],device).float()
                mean,logvar=net(x,mask,geom,gm)
                loss=net.loss(mean,logvar,y)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                total+=loss.item()*len(rows);n+=len(rows)
            net.eval(); val_sum=0.;val_n=0
            with torch.no_grad():
                for rows in batches(np.arange(len(pairs['val'])),1024,False):
                    idx=pairs['val'][rows];bi=idx[:,0];gi=idx[:,1]
                    mean,logvar=net(tensor(data['cont'][bi].reshape(-1,57,2),device).float(),tensor(data['mask'][bi].reshape(-1,57),device).bool(),
                                    tensor(data['geom'][bi],device).float(),tensor(data['gm'][gi],device).float())
                    truth=tensor(data['response_norm'][bi,gi],device).float()
                    val_sum+=net.loss(mean,logvar,truth).item()*len(rows);val_n+=len(rows)
            val=val_sum/val_n
            hist.append({'stage':'surrogate','member':member,'epoch':epoch+1,'train_loss':total/n,'val_loss':val,'seconds':round(time.time()-tick,2)})
            print('surrogate',member,epoch+1,'train',round(total/n,4),'val',round(val,4),flush=True)
            if val<best-1e-5:
                best=val;bad=0
                torch.save({'model':net.state_dict(),'width':128 if quick else 256,'member':member},ROOT/'checkpoints'/f'surrogate_{member}.pt')
            else:bad+=1
            if bad>=30:break
        net.load_state_dict(torch.load(ROOT/'checkpoints'/f'surrogate_{member}.pt',map_location=device,weights_only=True)['model'])
        net.eval();models.append(net)
    preds=[];trues=[]
    with torch.no_grad():
        for rows in batches(np.arange(len(pairs['test'])),1024,False):
            idx=pairs['test'][rows];bi=idx[:,0];gi=idx[:,1]
            xs=tensor(data['cont'][bi].reshape(-1,57,2),device).float()
            masks=tensor(data['mask'][bi].reshape(-1,57),device).bool()
            geoms=tensor(data['geom'][bi],device).float();gm=tensor(data['gm'][gi],device).float()
            out=torch.stack([net(xs,masks,geoms,gm)[0] for net in models]).mean(0)
            pred=response_in_physical_units(out,tensor(data['response_mu'],device),tensor(data['response_std'],device))
            preds.append(pred.cpu().numpy());trues.append(data['response'][bi,gi])
    pred=np.concatenate(preds);true=np.concatenate(trues)
    metrics={}
    for j,name in enumerate(('peak_drift','peak_floor_acceleration_mps2','residual_drift')):
        residual=pred[:,j]-true[:,j]
        metrics[name]={'mae':float(np.mean(np.abs(residual))),'rmse':float(np.sqrt(np.mean(residual**2))),
                       'r2':float(1-np.sum(residual**2)/np.sum((true[:,j]-true[:,j].mean())**2))}
    np.savez_compressed(run_dir/'surrogate_test_predictions.npz',predicted=pred,observed=true,paired_indices=pairs['test'])
    return models,metrics,hist


def train_generator(data,split,quick,device,run_dir,seed):
    c=data['section_props'].shape[0]
    dim=96 if quick else 256;layers=3 if quick else 12;heads=4 if quick else 8
    net=DualDiffusion(c,dim,layers,heads,prior=data['prior'],allowed=data['observed_roles']).to(device)
    diffuser=DualDiffusionProcess(net,steps=500)
    catalog=(np.log(data['section_props'])-data['cont_mu'])/data['cont_std']
    diffuser.set_catalog(catalog)
    opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.01)
    train_ids=np.array(split['buildings']['train']);val_ids=np.array(split['buildings']['val'])
    epochs=18 if quick else 300
    batch=32 if quick else 16
    best=float('inf');bad=0;hist=[]
    for epoch in range(epochs):
        tick=time.time();net.train();running=0.;counter=0; components=[]
        for idx in batches(train_ids,batch):
            x=tensor(data['cont'][idx].reshape(-1,57,2),device).float()
            s=tensor(data['section_idx'][idx].reshape(-1,57),device).long()
            m=tensor(data['mask'][idx].reshape(-1,57),device).bool()
            g=tensor(data['geom'][idx],device).float()
            p=tensor(data['perf'][idx],device).float()
            p=p*(torch.rand((len(idx),1),device=device)>=.15)
            loss,parts=diffuser.objective(x,s,g,p,m)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            running+=loss.item()*len(idx);counter+=len(idx);components.append(parts)
        net.eval();val_sum=0.;val_n=0
        with torch.no_grad():
            for idx in batches(val_ids,batch,False):
                loss,_=diffuser.objective(tensor(data['cont'][idx].reshape(-1,57,2),device).float(),
                         tensor(data['section_idx'][idx].reshape(-1,57),device).long(),
                         tensor(data['geom'][idx],device).float(),tensor(data['perf'][idx],device).float(),
                         tensor(data['mask'][idx].reshape(-1,57),device).bool())
                val_sum+=loss.item()*len(idx);val_n+=len(idx)
        val=val_sum/val_n
        hist.append({'stage':'diffusion','epoch':epoch+1,'train_loss':running/counter,'val_loss':val,
                     'noise_loss':float(np.mean([x['noise'] for x in components])),
                     'category_loss':float(np.mean([x['category'] for x in components])),
                     'consistency_loss':float(np.mean([x['consistency'] for x in components])),
                     'seconds':round(time.time()-tick,2)})
        print('diffusion',epoch+1,'train',round(running/counter,4),'val',round(val,4),flush=True)
        if val<best-1e-5:
            best=val;bad=0
            torch.save({'model':net.state_dict(),'dim':dim,'layers':layers,'heads':heads,
                        'classes':c,'prior':data['prior'],'allowed':data['observed_roles']},ROOT/'checkpoints'/'diffusion_best.pt')
        else:bad+=1
        if bad>=30:break
    state=torch.load(ROOT/'checkpoints'/'diffusion_best.pt',map_location=device,weights_only=False)
    net.load_state_dict(state['model']);net.eval()
    return net,diffuser,hist,{'parameter_count':parameters_count(net),'best_validation_loss':best,'epochs':len(hist)}


def sample_designs(data,split,diffuser,surrogates,quick,device,run_dir):
    test_ids=np.array(split['buildings']['test'])
    selected=test_ids[[0,15,35,55,75]]
    count=4 if quick else 32
    gm_ids=split['ground_motions']['train'][:16]
    gm=tensor(data['gm'][gm_ids],device).float()
    mean=tensor(data['response_mu'],device).float();std=tensor(data['response_std'],device).float()
    net=diffuser.net
    records=[];all_generated=[]
    catalog=diffuser.catalog
    for idx in selected:
        b=np.full(count,idx,dtype=int)
        geo=tensor(data['geom'][b],device).float()
        mask=tensor(data['mask'][b].reshape(count,57),device).bool()
        thresholds=np.array([.02,14.71,.01],dtype=np.float32)
        targets=(np.log1p(thresholds)-data['perf_mu'])/data['perf_std']
        perf=tensor(np.tile(targets,(count,1)),device).float()
        with torch.no_grad():
            z,sections=diffuser.generate(geo,perf,mask,num_steps=12 if quick else 50,seed=42+int(idx))
        # Project the continuous estimate onto the observed role-compatible catalog.
        sections=project_sections(z,net.allowed,catalog,mask)
        cont=catalog[sections]
        with torch.no_grad():
            losses=[]
            for j in range(count):
                geomj=geo[j:j+1].expand(len(gm),-1)
                maskj=mask[j:j+1].expand(len(gm),-1)
                zj=cont[j:j+1].expand(len(gm),-1,-1)
                pred,logvar=surrogates[0](zj,maskj,geomj,gm)
                actual=response_in_physical_units(pred,mean,std)
                risk=seismic_cvar(actual[None,:,:])
                losses.append(float(risk.item()))
        original=data['section_idx'][idx].reshape(-1)
        original_count=int(mask[0].sum())
        num_bays=int(data['geom_raw'][idx,1]);span=float(data['geom_raw'][idx,2])
        first_h=float(data['geom_raw'][idx,3]);other_h=float(data['geom_raw'][idx,4])
        heights=np.array([first_h]+[other_h]*(original_count//3-1),dtype=np.float32)
        for j in range(count):
            cand=sections[j].cpu().numpy()
            only=cand[:original_count]
            section_masses=data['section_props'][only,0].reshape(-1,3)*7850.0
            frame_mass=float(np.sum(section_masses[:,0]*2*heights+section_masses[:,1]*(num_bays-1)*heights+section_masses[:,2]*num_bays*span))
            reference_masses=data['section_props'][original[:original_count],0].reshape(-1,3)*7850.0
            reference_frame_mass=float(np.sum(reference_masses[:,0]*2*heights+reference_masses[:,1]*(num_bays-1)*heights+reference_masses[:,2]*num_bays*span))
            records.append({'building_id':int(data['ids'][idx]),'candidate':j,'proxy_cvar':losses[j],
                            'valid_catalog_fraction':float(np.mean(net.allowed[net.roles[:original_count],sections[j,:original_count]].cpu().numpy())),
                            'unique_sections':int(np.unique(only).size),'frame_mass_kg':frame_mass,'reference_frame_mass_kg':reference_frame_mass,
                            'reference_label_match':float(np.mean(only==original[:original_count]))})
            all_generated.append({'building_id':int(data['ids'][idx]),'candidate':j,'sections':only.tolist(),
                                  'reference_sections':original[:original_count].tolist(),'num_stories':int(data['geom_raw'][idx,0])})
    import pandas as pd
    pd.DataFrame(records).to_csv(run_dir/'generated_candidates.csv',index=False)
    (run_dir/'generated_designs.json').write_text(json.dumps(all_generated,indent=2))
    return {'generated_candidates':len(records),'valid_catalog_fraction':float(np.mean([r['valid_catalog_fraction'] for r in records])),
            'mean_proxy_cvar':float(np.mean([r['proxy_cvar'] for r in records])),
            'mean_frame_mass_kg':float(np.mean([r['frame_mass_kg'] for r in records])),
            'mean_section_match_to_reference':float(np.mean([r['reference_label_match'] for r in records]))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',default='DesignSafe.zip')
    ap.add_argument('--profile',choices=['pilot','research'],default='pilot')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--skip-preprocess',action='store_true')
    args=ap.parse_args();quick=args.profile=='pilot'
    seed_everything(args.seed)
    for path in ('results','checkpoints','logs','data/processed','data/splits'):(ROOT/path).mkdir(parents=True,exist_ok=True)
    run_dir=ROOT/'results'
    if not args.skip_preprocess:
        report=process_archive(args.archive,ROOT/'data/processed',seed=args.seed)
        print('audit',report,flush=True)
    else:report=json.loads((ROOT/'data/processed/data_audit.json').read_text())
    data,split=read_data();device='cuda' if torch.cuda.is_available() else 'cpu'
    t0=time.time()
    surrogates,sur_metrics,sur_log=train_surrogate(data,split,quick,device,run_dir,args.seed)
    network,diffuser,diff_log,train_info=train_generator(data,split,quick,device,run_dir,args.seed)
    gen_metrics=sample_designs(data,split,diffuser,surrogates,quick,device,run_dir)
    logs=sur_log+diff_log
    with (ROOT/'logs'/'training_log.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in logs))))
        writer.writeheader();writer.writerows(logs)
    metrics={'run_profile':args.profile,'hardware':device,'seed':args.seed,'seconds_total':round(time.time()-t0,2),
             'data_audit':report,'surrogate_heldout_test':sur_metrics,
             'generator_training':train_info,'generated_designs_proxy_screening':gen_metrics,
             'validation_status':{'opensees_new_generated_structures':'not_executed',
                'reported_proxy_risk_is_not_finite_element_validated':True,
                'diffusion_full_training_complete':not quick,'risk_guided_sampler_execution':'not_executed',
                'deep_learning_baselines_trained':'not_executed',
                'event_level_independent_test':'not_supported_by_metadata'}}
    (run_dir/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print('FINISHED',json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
