"""One-hop cardinal ConvGRU-style message passing with frozen observations.

Pointwise input encoder and action head; each update uses only four neighbors.
No pooling/global attention, coordinates, distance maps, or solver information.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

def neighbors(h):
    zero_r=mx.zeros_like(h[:,:1]); zero_c=mx.zeros_like(h[:,:,:1])
    return mx.concatenate([
        mx.concatenate([zero_r,h[:,:-1]],axis=1),
        mx.concatenate([h[:,:,1:],zero_c],axis=2),
        mx.concatenate([h[:,1:],zero_r],axis=1),
        mx.concatenate([zero_c,h[:,:,:-1]],axis=2)],axis=-1)

class LocalUpdate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gates=nn.Linear(6*width,2*width)
        self.proposal=nn.Linear(6*width,width)
        self.norm=nn.LayerNorm(width)
    def __call__(self,h,e,free):
        n=neighbors(h)
        z,r=mx.split(mx.sigmoid(self.gates(mx.concatenate([h,e,n],axis=-1))),2,axis=-1)
        candidate=mx.tanh(self.norm(self.proposal(mx.concatenate([r*h,e,n],axis=-1))))
        return ((1-z)*h+z*candidate)*free

class GridPlanner(nn.Module):
    def __init__(self,width=32,blocks=1,shared=True):
        super().__init__()
        self.encoder=nn.Linear(4,width)
        self.updates=[LocalUpdate(width) for _ in range(1 if shared else blocks)]
        self.head=nn.Linear(width,4)
        self.width=width; self.blocks=blocks; self.shared=shared
    def __call__(self,x,depths=(1,),return_states=False):
        if min(depths)<1 or max(depths)>16: raise ValueError("Depth outside validated range")
        if not self.shared and max(depths)>self.blocks: raise ValueError("Feedforward depth exceeds block count")
        free=x[...,1:2]; e=mx.tanh(self.encoder(x))*free
        h=e; agent=x[...,2:3]; out={}; previous=h
        for k in range(1,max(depths)+1):
            h=self.updates[0 if self.shared else k-1](h,e,free)
            if k in depths:
                readout=mx.sum(h*agent,axis=(1,2))
                record={"logits":self.head(readout),
                        "state_norm":mx.sqrt(mx.mean(h*h,axis=(1,2,3))),
                        "state_change":mx.sqrt(mx.mean((h-previous)**2,axis=(1,2,3)))}
                if return_states: record["states"]=h
                out[k]=record
            previous=h
        return out
    def parameter_count(self):
        return sum(v.size for _,v in tree_flatten(self.trainable_parameters()))

def set_loss(logits,optimal):
    """Negative log probability mass of ALL optimal actions; no arbitrary tie labels."""
    return mx.mean(mx.logsumexp(logits,axis=-1)-mx.logsumexp(mx.where(optimal,logits,-1e9),axis=-1))
