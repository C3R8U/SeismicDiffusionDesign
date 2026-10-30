"""Conditional variational encoder and a training-library retrieval benchmark."""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalSectionVAE(nn.Module):
    def __init__(self,classes,latent=48,hidden=256):
        super().__init__()
        self.embed=nn.Embedding(classes,24)
        self.encoder=nn.Sequential(nn.Linear(57*24+57+11,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.posterior=nn.Linear(hidden,latent*2)
        self.decoder=nn.Sequential(nn.Linear(latent+11,hidden),nn.GELU(),nn.Linear(hidden,hidden*2),nn.GELU(),nn.Linear(hidden*2,57*classes))
        self.classes=classes
        self.latent=latent

    def condition(self,geom,perf):
        return torch.cat((geom,perf),dim=-1)

    def forward(self,labels,mask,geom,perf):
        b=labels.shape[0]
        encoded=self.encoder(torch.cat([self.embed(labels).reshape(b,-1),mask.float(),self.condition(geom,perf)],dim=-1))
        mu,logvar=self.posterior(encoded).chunk(2,dim=-1)
        z=mu+(logvar*.5).exp()*torch.randn_like(mu)
        logits=self.decoder(torch.cat([z,self.condition(geom,perf)],dim=-1)).reshape(b,57,self.classes)
        cross=F.cross_entropy(logits[mask],labels[mask])
        kl=-.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).mean()/self.latent
        return cross+.01*kl,{'categorical':cross.item(),'kl':kl.item()}

    @torch.no_grad()
    def generate(self,geom,perf,mask):
        z=torch.randn(geom.shape[0],self.latent,device=geom.device)
        logits=self.decoder(torch.cat([z,self.condition(geom,perf)],dim=-1)).reshape(-1,57,self.classes)
        classes=logits.softmax(-1).reshape(-1,self.classes)
        labels=torch.multinomial(classes,1).reshape(geom.shape[0],57)
        return labels*mask.long()
