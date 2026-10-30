"""W-section properties and preliminary strong-column weak-beam checks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS=('A','Ix','Zx','Iy','Zy')
FACTORS=(.0254**2,.0254**4,.0254**3,.0254**4,.0254**3)


def catalog_arrays(source,section_names):
    catalog=pd.read_csv(source,sep=r'\s+',names=['section size',*FIELDS]).set_index('section size')
    missing=set(section_names)-set(catalog.index)
    if missing:raise ValueError(f'Missing section properties: {sorted(missing)}')
    values=catalog.loc[section_names,list(FIELDS)].to_numpy(dtype=float)*np.array(FACTORS)
    depths=np.array([float(name.split('X')[0][1:])*.0254 for name in section_names])
    values=np.column_stack([values[:,0],depths,*values[:,1:].T])
    assert np.isfinite(values).all() and np.all(values>0)
    return values


def design_member_counts(bays):
    return np.array([2,bays-1,bays],dtype=np.float64)


def steel_mass(indices,section_props,geometry,rho=7850.):
    s=np.asarray(indices,dtype=int)
    ns,bays=int(geometry[0]),int(geometry[1]);width=float(geometry[2])
    heights=np.r_[float(geometry[3]),np.full(max(ns-1,0),float(geometry[4]))]
    areas=section_props[s[:ns],0]
    factors=design_member_counts(bays)
    return float(rho*np.sum(areas*np.column_stack([2*heights,(bays-1)*heights,bays*width*np.ones(ns)])))


def scwb_screen(indices,section_props,geometry,fy=345e6,minimum_ratio=1.):
    """Plastic-modulus joint screening; connection-specific AISC checks need extra inputs."""
    ns=int(geometry[0]);bays=int(geometry[1])
    x=np.asarray(indices,dtype=int)[:ns]
    zx=section_props[x,3]
    node_ratios=[]
    for story in range(ns):
        for joint in range(bays+1):
            role=0 if joint in (0,bays) else 1
            column=zx[story,role]+(zx[story+1,role] if story+1<ns else 0.)
            beam=zx[story,2]*(1 if joint in (0,bays) else 2)
            if beam>0:node_ratios.append((story+1,joint,column/beam))
    vals=np.array([x[2] for x in node_ratios])
    return {'min_ratio':float(vals.min()),'fraction_over_1':float((vals>=minimum_ratio).mean()),
            'screen_pass':bool((vals>=minimum_ratio).all()),'joints':node_ratios,
            'rule':'sum column plastic moments / sum adjacent beam plastic moments; axial-load and probable-strength effects excluded'}


def repair_scwb(indices,section_props,geometry,allowed_roles,max_passes=5):
    """Try stronger legal column sections; reject layouts that cannot satisfy the joint screen."""
    x=np.asarray(indices,dtype=int).copy();ns=int(geometry[0]);bays=int(geometry[1]);
    for _ in range(max_passes):
        changed=False
        for story in range(ns):
            for role in (0,1):
                if role==1 and bays==1:continue
                ratios=scwb_screen(x,section_props,geometry)['joints']
                failures=[r for s,j,r in ratios if s==story+1 and ((j in (0,bays))==(role==0)) and r<1]
                if not failures:continue
                candidates=np.where(allowed_roles[role])[0]
                cur=int(x[story,role]);best=cur
                for cand in candidates[np.argsort(section_props[candidates,0])]:
                    if section_props[cand,3]<section_props[cur,3]:continue
                    trial=x.copy();trial[story,role]=cand
                    newratios=scwb_screen(trial,section_props,geometry)['joints']
                    if all(r>=1 for s,j,r in newratios if s==story+1 and ((j in (0,bays))==(role==0))):
                        best=int(cand);break
                if best!=cur:x[story,role]=best;changed=True
        if not changed:break
    return x,scwb_screen(x,section_props,geometry)


def section_role_mask(section_ids,training_idx,mask,n_classes):
    r=np.zeros((3,n_classes),bool)
    for role in range(3):
        labels=section_ids[training_idx,:,role][mask[training_idx,:,role]]
        r[role,np.unique(labels)]=True
    return r
