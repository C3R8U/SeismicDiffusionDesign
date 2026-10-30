"""Tail-risk guidance, section projection, and active-learning selection."""
import math
import torch
from torch.nn import functional as F


def response_in_physical_units(pred,mean,std):
    return torch.expm1(pred*std[None,:]+mean[None,:]).clamp(min=0)


def seismic_cvar(predictions,limits=(.02,14.71,.01),alpha=.9):
    """Compute the empirical upper-tail mean of dimensionless exceedance losses."""
    lim=predictions.new_tensor(limits)
    excess=F.relu(predictions/lim-1).square().mean(dim=-1)
    count=max(1,math.ceil((1-alpha)*excess.shape[-1]))
    return excess.topk(count,dim=-1).values.mean(dim=-1)


def guide_continuous(x0,mask,geom,surrogate,ground_motions,response_mean,response_std,
                     lambda_r=1.,lambda_w=.05,eta=.05):
    """Move the continuous design estimate toward lower proxy tail risk."""
    with torch.enable_grad():
        z=x0.detach().requires_grad_(True)
        b,t,_=z.shape;m=ground_motions.shape[0]
        repeated=z[:,None,:,:].expand(b,m,t,2).reshape(b*m,t,2)
        masks=mask[:,None,:].expand(b,m,t).reshape(b*m,t)
        geoms=geom[:,None,:].expand(b,m,geom.shape[-1]).reshape(b*m,-1)
        motions=ground_motions[None,:,:].expand(b,m,4).reshape(b*m,-1)
        mean,_=surrogate(repeated,masks,geoms,motions)
        actual=response_in_physical_units(mean,response_mean,response_std).reshape(b,m,3)
        risk=seismic_cvar(actual).mean()
        material=(z[:,:,0]*mask).sum()/(mask.sum().clamp(min=1))
        objective=lambda_r*risk+lambda_w*material
        grad=torch.autograd.grad(objective,z)[0]
        grad=grad/(grad.flatten(1).norm(dim=1).clamp(min=1)[:,None,None])
        return (z-eta*grad*mask[:,:,None]).detach(),risk.detach()


def project_sections(continuous,role_allowed,catalog,mask):
    """Select real catalog labels without inventing unavailable section properties."""
    b,n,_=continuous.shape
    roles=torch.arange(n,device=continuous.device)%3
    distances=(continuous[:,:,None,:]-catalog[None,None,:,:]).square().sum(-1)
    distances=distances.masked_fill(~role_allowed[roles][None,:,:],1e9)
    labels=distances.argmin(-1)
    return labels*mask.long()


def active_learning_score(std,loss,embeddings,known,uncertainty_weight=.4,boundary_weight=.4,diversity_weight=.2):
    """Balance uncertain responses, borderline designs, and unexplored structure types."""
    uncertainty=std.mean(-1)
    boundary=torch.exp(-(loss-1).abs()/.25)
    distance=torch.cdist(embeddings,known).min(dim=1).values
    def scale(a):
        return (a-a.min())/(a.max()-a.min()).clamp(min=1e-6)
    return uncertainty_weight*scale(uncertainty)+boundary_weight*scale(boundary)+diversity_weight*scale(distance)
