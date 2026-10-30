"""Fit the 12-block hierarchical categorical/continuous generator with independent seeds."""
from __future__ import annotations
import argparse,csv,json,os,random,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hierarchy import TopologyBank
from advanced_models import HierarchicalDualDiT,PairedDiffusion
from engineering import catalog_arrays,section_role_mask,scwb_screen,repair_scwb,steel_mass


def load_inputs():
    d=dict(np.load(ROOT/'data/processed/designsafe_processed.npz'))
    split=json.loads((ROOT/'data/splits/split.json').read_text())
    props=catalog_arrays(ROOT/'data/catalog/section_properties_source.txt',d['section_names'])
    train=np.asarray(split['buildings']['train'],dtype=int)
    logged=np.log(props)
    values=logged[d['section_idx']]
    mu=values[train][d['mask'][train]].mean(axis=0)
    sd=np.maximum(values[train][d['mask'][train]].std(axis=0),1e-5)
    cont=(values-mu)/sd
    catalog=(logged-mu)/sd
    allowed=section_role_mask(d['section_idx'],train,d['mask'],len(props))
    prior=np.ones((3,len(props)),dtype=np.float32)*.001
    for role in range(3):
        idx=d['section_idx'][train,:,role][d['mask'][train,:,role]]
        prior[role]+=np.bincount(idx,minlength=len(props))
        prior[role]/=prior[role].sum()
    return d,split,props,cont.astype('float32'),catalog.astype('float32'),allowed,prior


def run(args):
    torch.set_num_threads(args.threads)
    d,split,props,cont,catalog,allowed,prior=load_inputs()
    device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    bank=TopologyBank(d['geom_raw'])
    indices={key:np.asarray(value,dtype=int) for key,value in split['buildings'].items()}
    out=ROOT/'results/advanced';out.mkdir(parents=True,exist_ok=True)
    (ROOT/'logs/advanced').mkdir(exist_ok=True,parents=True)
    np.savez_compressed(ROOT/'data/processed/verified_section_properties.npz',section_names=d['section_names'],properties=props,fields=np.array(['A_m2','depth_m','Ix_m4','Zx_m3','Iy_m4','Zy_m3']),continuous_mu=np.log(props).mean(0),cont_train_mu=cont[indices['train']].mean((0,1)))
    catalog_summary={'sections':len(props),'all_positive':bool((props>0).all()),'source':'GUAN-XINGQUAN/AutoSDAPlatform AllSectionDatabase.csv',
        'units':{'A':'in2 -> m2','Ix/Iy':'in4 -> m4','Zx/Zy':'in3 -> m3'},'note':'Selected columns of the published engineering catalog; dimensions tracked in data/catalog.'}
    (out/'catalog_audit.json').write_text(json.dumps(catalog_summary,indent=2))
    record=[];completed=[]
    for seed in args.seeds:
        t0=time.monotonic(); random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        net=HierarchicalDualDiT(len(props),property_dim=6,width=args.width,depth=args.depth,heads=8,graph_layers=3,role_mask=np.ones_like(allowed)).to(device)
        process=PairedDiffusion(net,catalog,prior)
        optimizer=torch.optim.AdamW(net.parameters(),lr=args.learning_rate,weight_decay=.01)
        filename=ROOT/f'checkpoints/advanced_seed_{seed}.pt'
        best=float('inf')
        for epoch in range(1,args.epochs+1):
            epoch_data=[]
            for phase in ('train','val'):
                net.train(phase=='train')
                ids=indices[phase].copy()
                if phase=='train':np.random.shuffle(ids)
                cumulative=[0.,0.,0.,0.];batches=0
                for j in range(0,len(ids),args.batch_size):
                    ids_b=ids[j:j+args.batch_size]
                    topology=bank.take(ids_b,device)
                    x=torch.from_numpy(cont[ids_b].reshape(-1,57,6)).to(device)
                    s=torch.from_numpy(d['section_idx'][ids_b].reshape(-1,57)).long().to(device)
                    gm=torch.from_numpy(d['geom'][ids_b]).float().to(device)
                    perf=torch.from_numpy(d['perf'][ids_b]).float().to(device)
                    mask=torch.from_numpy(d['mask'][ids_b].reshape(-1,57)).to(device)
                    with torch.set_grad_enabled(phase=='train'):
                        loss,parts=process.objective(x,s,gm,perf,mask,topology)
                        if phase=='train':
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                            optimizer.step()
                    components=[float(loss.detach()),parts['continuous'],parts['categorical'],parts['consistency']]
                    cumulative=[a+b for a,b in zip(cumulative,components)];batches+=1
                means=[round(v/batches,7) for v in cumulative]
                row=dict(seed=seed,epoch=epoch,phase=phase,loss=means[0],continuous=means[1],categorical=means[2],consistency=means[3],
                         parameter_count=sum(p.numel() for p in net.parameters()),elapsed_seconds=round(time.monotonic()-t0,2))
                record.append(row);epoch_data.append(row)
                print(json.dumps(row),flush=True)
            if epoch_data[1]['loss']<best:
                best=epoch_data[1]['loss']
                torch.save({'network':net.state_dict(),'seed':seed,'epoch':epoch,'val_loss':best,
                            'architecture':{'width':args.width,'depth':args.depth,'heads':8,'property_dim':6},
                            'feature_mu':np.log(props).mean(axis=0),'catalog_mu':np.log(props),
                            'allowed':allowed,'prior':prior,'feature_train_normalized_catalog':catalog},filename)
        completed.append({'seed':seed,'epochs':args.epochs,'best_validation_loss':best,'checkpoint':str(filename.relative_to(ROOT))})
        # Generate only geometries supported by the available model and topology bank.
        if args.sample:
            state=torch.load(filename,map_location=device,weights_only=False);net.load_state_dict(state['network']);net.eval()
            selected=indices['test'][:args.sample]
            generated=[]
            for sample_id in selected:
                topology=bank.take([sample_id],device)
                geom=torch.from_numpy(d['geom'][sample_id:sample_id+1]).float().to(device)
                performance_targets=np.array([.02,14.71,.01],dtype=np.float32)
                normalized=(np.log1p(performance_targets)-d['perf_mu'])/d['perf_std']
                performance=torch.from_numpy(normalized[None].astype('float32')).to(device)
                mask=torch.from_numpy(d['mask'][sample_id:sample_id+1].reshape(1,57)).to(device)
                z,sections=process.sample(geom,performance,mask,topology,n_steps=args.sampling_steps,seed=seed+int(sample_id),guidance=1.5)
                category=sections[0].reshape(19,3).cpu().numpy()
                true=d['section_idx'][sample_id]
                geometry=d['geom_raw'][sample_id]
                # A physical projection keeps only sections observed in the appropriate member role.
                selected_sections=category.copy()
                for role in range(3):
                    candidates=np.where(allowed[role])[0]
                    invalid=~allowed[role,selected_sections[:,role]]
                    estimated=z[0].reshape(19,3,6).cpu().numpy()
                    distances=((catalog[candidates][None,:,:]-estimated[invalid,role,None,:])**2).sum(-1)
                    selected_sections[invalid,role]=candidates[np.argmin(distances,axis=-1)]
                repaired,checks=repair_scwb(selected_sections,props,geometry,allowed)
                generated.append({'seed':seed,'building_id':int(d['ids'][sample_id]),'story_count':int(geometry[0]),'bays':int(geometry[1]),
                                  'screen_reference':scwb_screen(true,props,geometry)['min_ratio'],
                                  'screen_raw':scwb_screen(selected_sections,props,geometry)['min_ratio'],
                                  'screen_repaired':checks['min_ratio'],'screen_pass':checks['screen_pass'],
                                  'mass_reference':steel_mass(true,props,geometry),
                                  'mass_generated':steel_mass(selected_sections,props,geometry),
                                  'mass_repaired':steel_mass(repaired,props,geometry),
                                  'sections_raw':selected_sections[:int(geometry[0])].tolist(),
                                  'sections_repaired':repaired[:int(geometry[0])].tolist()})
            (out/f'generated_seed_{seed}.json').write_text(json.dumps(generated,indent=2))
    pd.DataFrame(record).to_csv(ROOT/'logs/advanced/full_dit_training.csv',index=False)
    summary={'status':'completed_run','hardware':device,'model':{'depth':args.depth,'width':args.width},'seeds':completed,
             'full_300_epoch_design':False,'opensees_nonlinear_analysis':'not_run',
             'sample_count':args.sample,'training_epochs_per_seed':args.epochs,
             'independent_structural_compliance':'not established'}
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2))
    print('TRAINING_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--epochs',type=int,default=3);a.add_argument('--seeds',type=int,nargs='+',default=[42,2026,3407]);
    a.add_argument('--batch-size',type=int,default=24);a.add_argument('--width',type=int,default=256);a.add_argument('--depth',type=int,default=12);
    a.add_argument('--threads',type=int,default=4);a.add_argument('--sample',type=int,default=2);a.add_argument('--sampling-steps',type=int,default=12)
    a.add_argument('--learning-rate',type=float,default=2e-4);a.add_argument('--cpu',action='store_true');
    run(a.parse_args())
