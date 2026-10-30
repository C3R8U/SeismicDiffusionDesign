"""Hierarchical transformer with coupled categorical and engineering-property diffusion."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from hierarchy import FourLevelMechanics,TOKENS


def time_embedding(t,width):
    d=width//2
    freq=torch.exp(torch.arange(d,device=t.device)*(-math.log(10000)/max(d-1,1)))
    v=t.float().unsqueeze(-1)*freq
    return torch.cat([v.sin(),v.cos()],-1)


class HierarchicalDualDiT(nn.Module):
    """A shared transformer lets W-shape identities and continuous stiffness exchange context."""
    def __init__(self,classes,property_dim=6,width=256,depth=12,heads=8,graph_layers=3,dropout=.1,role_mask=None):
        super().__init__()
        self.classes=classes
        self.property_dim=property_dim
        self.width=width
        self.sections=nn.Embedding(classes,width)
        self.properties=nn.Linear(property_dim,width)
        self.condition=nn.Sequential(nn.Linear(11,width),nn.GELU(),nn.Linear(width,width))
        self.times=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
        self.role=nn.Embedding(3,width)
        self.story=nn.Embedding(19,width)
        self.physical=FourLevelMechanics(width,depth=graph_layers)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(width,heads,width*4,dropout=dropout,batch_first=True,norm_first=True,activation='gelu') for _ in range(depth)])
        self.norm=nn.LayerNorm(width)
        self.output_noise=nn.Linear(width,property_dim)
        self.output_class=nn.Linear(width,classes)
        self.register_buffer('roles',torch.arange(TOKENS)%3)
        self.register_buffer('stories',torch.arange(TOKENS)//3)
        if role_mask is None:role_mask=torch.ones((3,classes),dtype=torch.bool)
        self.register_buffer('role_mask',torch.as_tensor(role_mask,dtype=torch.bool))

    def forward(self,z,section,t,geometry,performance,mask,topology,conditioning=True):
        common=self.condition(torch.cat([geometry,performance if conditioning else torch.zeros_like(performance)],-1))
        h=self.sections(section)+self.properties(z)+self.role(self.roles)[None]+self.story(self.stories)[None]
        h=h+common.unsqueeze(1)+self.times(time_embedding(t,self.width)).unsqueeze(1)
        h=self.physical(h,topology,mask)
        for layer in self.blocks:
            h=layer(h,src_key_padding_mask=~mask)
        h=self.norm(h)
        logits=self.output_class(h)
        allowed=self.role_mask[self.roles].unsqueeze(0)
        logits=logits.masked_fill(~allowed,-1e4)
        return self.output_noise(h)*mask.unsqueeze(-1),logits


class PairedDiffusion:
    def __init__(self,network,properties,role_prior,steps=500,beta_min=.0001,beta_max=.02):
        self.network=network
        self.properties=torch.as_tensor(properties,dtype=torch.float32,device=next(network.parameters()).device)
        self.prior=torch.as_tensor(role_prior,dtype=torch.float32,device=self.properties.device)
        self.steps=steps
        self.beta=torch.linspace(beta_min,beta_max,steps,device=self.properties.device)
        self.alpha=1-self.beta
        self.alpha_bar=torch.cumprod(self.alpha,0)

    def objective(self,continuous,sections,geometry,performance,mask,topology,lambda_cat=1.,lambda_consistency=.2):
        n=continuous.shape[0];t=torch.randint(self.steps,(n,),device=continuous.device)
        alpha=self.alpha_bar[t,None,None]
        noise=torch.randn_like(continuous)
        noisy=alpha.sqrt()*continuous+(1-alpha).sqrt()*noise
        roles=self.network.roles.unsqueeze(0).expand_as(sections)
        prior=self.prior[roles]
        replacement=torch.multinomial(prior.reshape(-1,self.network.classes),1).reshape_as(sections)
        noised_sections=torch.where(torch.rand_like(sections,dtype=torch.float32)<alpha.squeeze(-1),sections,replacement)
        eps,logits=self.network(noisy,noised_sections,t,geometry,performance,mask,topology,conditioning=torch.rand(())>.15)
        masked=mask.unsqueeze(-1)
        mse=((eps-noise).square()*masked).sum()/(mask.sum().clamp(min=1)*continuous.shape[-1])
        ce=F.cross_entropy(logits[mask],sections[mask])
        p=logits.softmax(-1)
        original=(noisy-(1-alpha).sqrt()*eps)/alpha.sqrt().clamp(min=1e-4)
        consistent=p@self.properties
        consistency=((original-consistent).square()*masked).sum()/(mask.sum().clamp(min=1)*continuous.shape[-1])
        total=mse+lambda_cat*ce+lambda_consistency*consistency
        return total,{'continuous':float(mse.detach()),'categorical':float(ce.detach()),'consistency':float(consistency.detach())}

    @torch.no_grad()
    def sample(self,geometry,performance,mask,topology,n_steps=20,seed=42,guidance=1.5,risk_fn=None,risk_step=.05):
        torch.manual_seed(seed)
        b=geometry.shape[0];device=geometry.device
        z=torch.randn((b,TOKENS,self.properties.shape[-1]),device=device)*mask.unsqueeze(-1)
        prior=self.prior[self.network.roles][None].expand(b,-1,-1)
        s=torch.multinomial(prior.reshape(-1,self.network.classes),1).reshape(b,TOKENS)
        t_values=torch.linspace(self.steps-1,0,n_steps,device=device).long().unique_consecutive()
        for i,t in enumerate(t_values):
            tt=torch.full((b,),int(t),device=device,dtype=torch.long)
            eps,logits=self.network(z,s,tt,geometry,performance,mask,topology)
            if guidance:
                eps_u,logits_u=self.network(z,s,tt,geometry,performance,mask,topology,conditioning=False)
                eps=eps+guidance*(eps-eps_u)
                logits=logits+guidance*(logits-logits_u)
            abar=self.alpha_bar[t]
            x0=(z-(1-abar).sqrt()*eps)/abar.sqrt().clamp(min=1e-4)
            if risk_fn is not None:
                with torch.enable_grad():
                    proposal=x0.detach().requires_grad_(True)
                    objective=risk_fn(proposal)
                    risk_gradient=torch.autograd.grad(objective,proposal)[0]
                    scale=risk_gradient.flatten(1).norm(dim=1,keepdim=True).clamp(min=1.)
                    risk_gradient=risk_gradient/scale.view(-1,1,1)
                x0=(x0-risk_step*risk_gradient).detach()
            p0=logits.softmax(-1)
            if i+1<len(t_values):
                nxt=self.alpha_bar[t_values[i+1]]
                z=nxt.sqrt()*x0+(1-nxt).sqrt()*eps
                # The exact categorical posterior is mixed over the denoiser's estimate of clean sections.
                retention=(abar/nxt).clamp(0,1)
                current_prior=prior.gather(-1,s.unsqueeze(-1))
                observed=F.one_hot(s,self.network.classes).float()
                q_t_clean=abar*observed+(1-abar)*current_prior
                clean_weight=p0/q_t_clean.clamp(min=1e-12)
                previous_mass=nxt*clean_weight+(1-nxt)*prior*clean_weight.sum(-1,keepdim=True)
                transition=retention*observed+(1-retention)*current_prior
                posterior=(previous_mass*transition).clamp(min=1e-16)
                posterior=posterior/posterior.sum(-1,keepdim=True)
                s=torch.multinomial(posterior.reshape(-1,self.network.classes),1).reshape(b,TOKENS)
            else:
                z=x0
                s=p0.argmax(-1)
            z=z*mask.unsqueeze(-1)
        return z,s
