"""Graph-conditioned discrete/continuous diffusion and seismic response prediction."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = 19 * 3


def make_graph():
    """Connect the three design groups at each story and neighboring stories."""
    a = torch.eye(TOKENS)
    for story in range(19):
        for role in range(3):
            i = story * 3 + role
            for other in range(3):
                a[i, story*3+other] = 1
            for next_story in (story-1,story+1):
                if 0 <= next_story < 19:
                    a[i,next_story*3+role]=1
    return a


class GraphContext(nn.Module):
    def __init__(self,width,layers=3):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(width*2,width) for _ in range(layers)])
        self.norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.register_buffer('edges',make_graph())

    def forward(self,h,mask):
        weights=self.edges[None,:,:] * mask[:,None,:].float()
        weights=weights/weights.sum(-1,keepdim=True).clamp(min=1)
        for linear,norm in zip(self.layers,self.norm):
            aggregate=weights @ h
            h=norm(h+F.gelu(linear(torch.cat([h,aggregate],dim=-1))))
            h=h*mask[:,:,None]
        return h


def sine_time(t,dim):
    half=dim//2
    freq=torch.exp(torch.arange(half,device=t.device,dtype=torch.float32)*(-math.log(10000)/max(half-1,1)))
    pos=t.float()[:,None]*freq[None,:]
    return torch.cat([pos.sin(),pos.cos()],dim=-1)


class DualDiffusion(nn.Module):
    """Each design token carries one W-shape class and two verified continuous attributes."""
    def __init__(self,classes,dim=256,layers=12,heads=8,graph_layers=3,dropout=.1,prior=None,allowed=None):
        super().__init__()
        self.classes=classes
        self.dim=dim
        self.class_embed=nn.Embedding(classes,dim)
        self.cont_embed=nn.Linear(2,dim)
        self.geom_embed=nn.Sequential(nn.Linear(8+3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.time_embed=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.role_embed=nn.Embedding(3,dim)
        self.story_embed=nn.Embedding(19,dim)
        self.graph=GraphContext(dim,graph_layers)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=dim*4,
                              dropout=dropout,batch_first=True,norm_first=True,activation='gelu') for _ in range(layers)])
        self.out_norm=nn.LayerNorm(dim)
        self.cont_out=nn.Linear(dim,2)
        self.class_out=nn.Linear(dim,classes)
        self.register_buffer('roles',torch.arange(TOKENS)//19 if False else torch.arange(TOKENS)%3)
        self.register_buffer('stories',torch.arange(TOKENS)//3)
        self.register_buffer('prior',torch.as_tensor(prior,dtype=torch.float32))
        self.register_buffer('allowed',torch.as_tensor(allowed,dtype=torch.bool))

    def forward(self,z,s,t,geom,perf,mask):
        batch=z.size(0)
        geom_cond=self.geom_embed(torch.cat([geom,perf],dim=-1))
        h=self.cont_embed(z)+self.class_embed(s)+self.role_embed(self.roles)[None,:,:]+self.story_embed(self.stories)[None,:,:]
        h=h+geom_cond[:,None,:]+self.time_embed(sine_time(t,self.dim))[:,None,:]
        h=self.graph(h,mask)
        for block in self.blocks:
            h=block(h,src_key_padding_mask=~mask)
        h=self.out_norm(h)
        logits=self.class_out(h)
        permitted=self.allowed[self.roles][None,:,:]
        logits=logits.masked_fill(~permitted,-1e4)
        return self.cont_out(h)*mask[:,:,None],logits


class DualDiffusionProcess:
    def __init__(self,net,steps=500,beta_start=.0001,beta_end=.02):
        self.net=net
        self.steps=steps
        self.beta=torch.linspace(beta_start,beta_end,steps,device=next(net.parameters()).device)
        self.alpha=1-self.beta
        self.alpha_bar=self.alpha.cumprod(0)

    def noisy(self,x,s,t,mask):
        abar=self.alpha_bar[t][:,None,None]
        noise=torch.randn_like(x)
        xt=abar.sqrt()*x+(1-abar).sqrt()*noise
        # The categorical chain keeps a section or draws a role-compatible prior sample.
        roles=self.net.roles[None,:].expand_as(s)
        probabilities=self.net.prior[roles]
        fresh=torch.multinomial(probabilities.reshape(-1,self.net.classes),1).reshape_as(s)
        keep=torch.rand_like(s.float())<abar.squeeze(-1)
        st=torch.where(keep,s,fresh)
        return xt*mask[:,:,None],st,noise

    def objective(self,x,s,geom,perf,mask,lambda_cat=1.,lambda_physics=.2):
        batch=x.size(0)
        t=torch.randint(0,self.steps,(batch,),device=x.device)
        xt,st,eps=self.noisy(x,s,t,mask)
        noise,logits=self.net(xt,st,t,geom,perf,mask)
        mse=((noise-eps).square()*mask[:,:,None]).sum()/mask.sum().clamp(min=1)/2
        ce=F.cross_entropy(logits[mask],s[mask])
        abar=self.alpha_bar[t][:,None,None]
        x0=(xt-(1-abar).sqrt()*noise)/abar.sqrt().clamp(min=1e-5)
        # The section map is expressed in the same normalized coordinates as the tokens.
        p=logits.softmax(-1)
        catalog=self.catalog
        catalog_expected=p@catalog
        physics=((x0-catalog_expected).square()*mask[:,:,None]).sum()/mask.sum().clamp(min=1)/2
        return mse+lambda_cat*ce+lambda_physics*physics,{'noise':mse.item(),'category':ce.item(),'consistency':physics.item()}

    def set_catalog(self,catalog):
        self.catalog=torch.as_tensor(catalog,device=self.beta.device,dtype=torch.float32)

    @torch.no_grad()
    def generate(self,geom,perf,mask,num_steps=50,seed=42,surrogate=None,
                 ground_motions=None,response_mean=None,response_std=None,
                 risk_weight=1.0,mass_weight=.05,guidance_step=.05):
        torch.manual_seed(seed)
        batch=geom.shape[0];device=geom.device
        z=torch.randn(batch,TOKENS,2,device=device)*mask[:,:,None]
        roles=self.net.roles[None,:].expand(batch,-1)
        s=torch.multinomial(self.net.prior[roles].reshape(-1,self.net.classes),1).reshape(batch,TOKENS)
        timesteps=torch.linspace(self.steps-1,0,num_steps,device=device).long().unique_consecutive()
        for j,t in enumerate(timesteps):
            tt=torch.full((batch,),int(t),device=device,dtype=torch.long)
            eps,logits=self.net(z,s,tt,geom,perf,mask)
            a=self.alpha_bar[t]
            x0=(z-(1-a).sqrt()*eps)/a.sqrt().clamp(min=1e-5)
            if surrogate is not None:
                from risk import guide_continuous
                if ground_motions is None or response_mean is None or response_std is None:
                    raise ValueError('Seismic inputs and response scaling are required for risk guidance')
                x0,_=guide_continuous(x0,mask,geom,surrogate,ground_motions,
                         response_mean,response_std,lambda_r=risk_weight,
                         lambda_w=mass_weight,eta=guidance_step)
            next_t=int(timesteps[j+1]) if j+1 < len(timesteps) else -1
            if next_t >= 0:
                an=self.alpha_bar[next_t]
                z=an.sqrt()*x0+(1-an).sqrt()*eps
                prior=self.net.prior[roles]
                p0=logits.softmax(-1)
                prev=(an*p0+(1-an)*prior)
                beta_eff=(1-(a/an)).clamp(min=0,max=1)
                selected=F.one_hot(s,self.net.classes).float()
                transition=(1-beta_eff)*selected+beta_eff*prior.gather(-1,s[:,:,None])
                probs=(prev*transition).clamp(min=1e-15)
                probs=probs/probs.sum(-1,keepdim=True)
                s=torch.multinomial(probs.reshape(-1,self.net.classes),1).reshape_as(s)
            else:
                z=x0
                s=logits.argmax(-1)
            z=z*mask[:,:,None]
        return z,s


class SeismicSurrogate(nn.Module):
    """A heteroscedastic response predictor for structure/record pairs."""
    def __init__(self,width=256,dropout=.1):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(TOKENS*2+TOKENS+8+4,width),nn.LayerNorm(width),nn.GELU(),
                    nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,width//2),nn.GELU(),nn.Linear(width//2,6))

    def forward(self,cont,mask,geom,gm):
        b=cont.shape[0]
        features=torch.cat([cont.reshape(b,-1)*mask.reshape(b,-1).repeat_interleave(2,1),mask.float(),geom,gm],dim=-1)
        pred=self.network(features)
        return pred[:,:3],pred[:,3:].clamp(-8,4)

    @staticmethod
    def loss(mean,logvar,y):
        return (.5*((y-mean).square()*(-logvar).exp()+logvar)).mean()


def parameters_count(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
