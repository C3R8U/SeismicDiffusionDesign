"""Sample independent structural designs from stored hierarchical checkpoints."""
from pathlib import Path
import json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from advanced_models import HierarchicalDualDiT,PairedDiffusion
from hierarchy import TopologyBank
from engineering import catalog_arrays,section_role_mask,scwb_screen,repair_scwb,steel_mass

def main():
    torch.set_num_threads(3)
    data=dict(np.load(ROOT/'data/processed/designsafe_processed.npz'))
    split=json.loads((ROOT/'data/splits/split.json').read_text())
    train=np.asarray(split['buildings']['train'],dtype=int);tests=np.asarray(split['buildings']['test'],dtype=int)[:2]
    props=catalog_arrays(ROOT/'data/catalog/section_properties_source.txt',data['section_names'])
    val=np.log(props)[data['section_idx']];mu=val[train][data['mask'][train]].mean(0)
    std=np.maximum(val[train][data['mask'][train]].std(0),1e-5)
    catalog=((np.log(props)-mu)/std).astype('float32')
    role=section_role_mask(data['section_idx'],train,data['mask'],len(props))
    prior=np.ones((3,len(props)),np.float32)*.001
    for r in range(3):
        ids=data['section_idx'][train,:,r][data['mask'][train,:,r]]
        prior[r]+=np.bincount(ids,minlength=len(props));prior[r]/=prior[r].sum()
    bank=TopologyBank(data['geom_raw'])
    for seed in (42,2026,3407):
        state=torch.load(ROOT/f'checkpoints/advanced_seed_{seed}.pt',map_location='cpu',weights_only=False)
        net=HierarchicalDualDiT(len(props),property_dim=6,width=256,depth=12,heads=8,graph_layers=3,role_mask=np.ones_like(role))
        net.load_state_dict(state['network']);net.eval();process=PairedDiffusion(net,catalog,prior)
        rows=[]
        for i in tests:
            geom=torch.from_numpy(data['geom'][i:i+1]);mask=torch.from_numpy(data['mask'][i:i+1].reshape(1,57))
            target=np.array([.02,14.71,.01],np.float32)
            perf=torch.from_numpy(((np.log1p(target)-data['perf_mu'])/data['perf_std'])[None].astype('float32'))
            with torch.inference_mode():
                predicted,cat=process.sample(geom,perf,mask,bank.take([i],'cpu'),n_steps=12,seed=seed+int(i),guidance=1.5)
            cat=cat[0].reshape(19,3).numpy()
            n=int(data['geom_raw'][i,0]);reference=data['section_idx'][i]
            for r in range(3):
                choices=np.where(role[r])[0]
                for st in range(n):
                    if not role[r,cat[st,r]]:
                        zz=predicted[0,st*3+r].numpy()
                        cat[st,r]=choices[np.argmin(np.square(catalog[choices]-zz).sum(-1))]
            corrected,check=repair_scwb(cat,props,data['geom_raw'][i],role)
            rows.append({'seed':seed,'building_id':int(data['ids'][i]),'story_count':n,'bays':int(data['geom_raw'][i,1]),
                         'screen_reference':scwb_screen(reference,props,data['geom_raw'][i])['min_ratio'],
                         'screen_raw':scwb_screen(cat,props,data['geom_raw'][i])['min_ratio'],
                         'screen_repaired':check['min_ratio'],'screen_pass':check['screen_pass'],
                         'mass_reference':steel_mass(reference,props,data['geom_raw'][i]),
                         'mass_generated':steel_mass(cat,props,data['geom_raw'][i]),
                         'mass_repaired':steel_mass(corrected,props,data['geom_raw'][i]),
                         'sections_raw':cat[:n].tolist(),'sections_repaired':corrected[:n].tolist(),
                         'performance_condition':[.02,14.71,.01],
                         'seismic_finite_element_validation':'not executed'})
        path=ROOT/f'results/advanced/generated_seed_{seed}.json';path.write_text(json.dumps(rows,indent=2))
        print(seed,'generated',len(rows),flush=True)

if __name__=='__main__':main()
