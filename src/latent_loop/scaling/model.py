"""Extended execution of immutable 5B parameters, plus explicit global control."""
import mlx.core as mx
import mlx.nn as nn
from latent_loop.planning.grid_recurrence import GridPlanner

def forward(model,x,depths=(8,),states=False,cut=None):
    if min(depths)<1 or max(depths)>32:raise ValueError('Validated limit 32')
    if not model.shared and max(depths)>model.blocks:raise ValueError('Untied block count exceeded')
    free=x[...,1:2];e=mx.tanh(model.encoder(x))*free;h=e;agent=x[...,2:3];out={}
    if hasattr(model,'context'):
        # Global CNN broadcasts pooled features and raw-channel coordinate moments.
        n=x.shape[1];rr=mx.arange(n,dtype=x.dtype)[None,:,None,None]/max(1,n-1);cc=mx.arange(n,dtype=x.dtype)[None,None,:,None]/max(1,n-1)
        moments=mx.concatenate([mx.sum(x[...,i:i+1]*coord,axis=(1,2)) for i in (2,3) for coord in (rr,cc)],axis=-1)
        context=model.context(mx.concatenate([mx.mean(e,axis=(1,2)),moments],axis=-1))[:,None,None,:]
        e=mx.tanh(e+context)*free;h=e
    initial=h
    for k in range(1,max(depths)+1):
        previous=h
        if cut is not None:h=h*cut
        h=model.updates[0 if model.shared else k-1](h,e,free)
        if k in depths:
            record={'logits':model.head(mx.sum(h*agent,axis=(1,2))), 'state_norm':mx.sqrt(mx.mean(h*h,axis=(1,2,3))), 'state_change':mx.sqrt(mx.mean((h-previous)**2,axis=(1,2,3))), 'state_drift':mx.sqrt(mx.mean((h-initial)**2,axis=(1,2,3))), 'state_cosine':mx.sum(h*previous,axis=(1,2,3))/(mx.sqrt(mx.sum(h*h,axis=(1,2,3))*mx.sum(previous*previous,axis=(1,2,3)))+1e-12)}
            if states:record['states']=h
            out[k]=record
    return out

class GlobalPlanner(GridPlanner):
    def __init__(self,width=32,blocks=4):
        super().__init__(width,blocks,False);self.context=nn.Linear(width+4,width)

def build(spec,seed):
    mx.random.seed(seed)
    return GlobalPlanner(spec['width'],spec['blocks']) if spec.get('global') else GridPlanner(spec['width'],spec['blocks'],spec['shared'])
