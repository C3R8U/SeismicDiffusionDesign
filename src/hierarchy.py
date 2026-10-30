"""Physical member connectivity and differentiable four-level frame pooling."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MAX_STORIES=19
MAX_BAYS=5
MAX_MEMBERS=MAX_STORIES*(2*MAX_BAYS+1)
MAX_JOINTS=(MAX_STORIES+1)*(MAX_BAYS+1)
TOKENS=MAX_STORIES*3


def physical_topology(stories:int,bays:int):
    """Build columns, beams, and the actual grid joints from a regular frame."""
    assert 1<=stories<=MAX_STORIES and 1<=bays<=MAX_BAYS
    groups=[];starts=[];ends=[];levels=[];member_types=[]
    node=lambda level,grid: level*(bays+1)+grid
    for story in range(stories):
        for grid in range(bays+1):
            groups.append(story*3+(0 if grid in (0,bays) else 1))
            starts.append(node(story,grid));ends.append(node(story+1,grid))
            levels.append(story);member_types.append(0)
        for grid in range(bays):
            groups.append(story*3+2)
            starts.append(node(story+1,grid));ends.append(node(story+1,grid+1))
            levels.append(story);member_types.append(1)
    expected=stories*(2*bays+1)
    assert len(groups)==expected
    def pad(xs,length):return np.asarray(xs+[0]*(length-len(xs)),dtype=np.int64)
    return {'member_group':pad(groups,MAX_MEMBERS),'node_start':pad(starts,MAX_MEMBERS),
            'node_end':pad(ends,MAX_MEMBERS),'member_story':pad(levels,MAX_MEMBERS),
            'member_type':pad(member_types,MAX_MEMBERS),'member_mask':np.arange(MAX_MEMBERS)<expected,
            'node_mask':np.arange(MAX_JOINTS)<(stories+1)*(bays+1)}


class TopologyBank:
    def __init__(self,geometry):
        entries=[physical_topology(int(row[0]),int(row[1])) for row in geometry]
        self.data={key:torch.tensor(np.stack([e[key] for e in entries])) for key in entries[0]}

    def take(self,indices,device):
        if isinstance(indices,torch.Tensor):indices=indices.cpu().numpy()
        return {key:value[indices].to(device) for key,value in self.data.items()}


class FourLevelMechanics(nn.Module):
    """Aggregate member forces through joints, floors, and the whole lateral system."""
    def __init__(self,width:int,depth:int=3):
        super().__init__()
        self.member_type=nn.Embedding(2,width)
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(width*4,width),nn.GELU(),nn.Linear(width,width)) for _ in range(depth)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])

    @staticmethod
    def mean_scatter(features,indices,mask,count):
        b,n,d=features.shape
        result=torch.zeros((b,count,d),device=features.device,dtype=features.dtype)
        result.scatter_add_(1,indices.unsqueeze(-1).expand(-1,-1,d),features*mask.unsqueeze(-1))
        occurrences=torch.zeros((b,count,1),device=features.device,dtype=features.dtype)
        occurrences.scatter_add_(1,indices.unsqueeze(-1),mask.unsqueeze(-1).to(features.dtype))
        return result/occurrences.clamp(min=1)

    def forward(self,group_features,topology,group_mask):
        idx=topology['member_group'];active=topology['member_mask'];start=topology['node_start'];end=topology['node_end']
        story=topology['member_story'];kind=topology['member_type']
        member=group_features.gather(1,idx.unsqueeze(-1).expand(-1,-1,group_features.size(-1)))
        member=member+self.member_type(kind)
        for block,norm in zip(self.layers,self.norms):
            start_agg=self.mean_scatter(member,start,active,MAX_JOINTS)
            end_agg=self.mean_scatter(member,end,active,MAX_JOINTS)
            node_agg=(start_agg+end_agg)*.5
            joint_context=(node_agg.gather(1,start.unsqueeze(-1).expand_as(member))+
                           node_agg.gather(1,end.unsqueeze(-1).expand_as(member)))*.5
            floor_features=self.mean_scatter(member,story,active,MAX_STORIES)
            story_context=floor_features.gather(1,story.unsqueeze(-1).expand_as(member))
            overall=(member*active.unsqueeze(-1)).sum(1,keepdim=True)/active.sum(1,keepdim=True).clamp(min=1).unsqueeze(-1)
            overall=overall.expand_as(member)
            updated=block(torch.cat([member,joint_context,story_context,overall],dim=-1))
            member=norm(member+updated)*active.unsqueeze(-1)
        group=self.mean_scatter(member,idx,active,TOKENS)
        return group*group_mask.unsqueeze(-1)
