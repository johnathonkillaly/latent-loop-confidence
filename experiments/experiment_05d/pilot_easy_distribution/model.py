"""Permutation-equivariant graph recurrence and shared candidate heads (MLX)."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

def tensor_inputs(inputs):return {k:mx.array(v) for k,v in inputs.items()}

class Update(nn.Module):
    def __init__(self,width):
        super().__init__();self.gates=nn.Linear(3*width,2*width)
        self.proposal=nn.Linear(3*width,width);self.norm=nn.LayerNorm(width)
    def __call__(self,h,e,a,valid):
        # Neighbor means, no graph-wide pooling or ID embeddings. One edge per update.
        degree=mx.maximum(mx.sum(a,-1,keepdims=True),1)
        message=(a@h)/degree
        z,r=mx.split(mx.sigmoid(self.gates(mx.concatenate([h,e,message],-1))),2,-1)
        p=mx.tanh(self.norm(self.proposal(mx.concatenate([r*h,e,message],-1))))
        return ((1-z)*h+z*p)*valid

class GraphPolicy(nn.Module):
    def __init__(self,width=32,blocks=8,shared=True):
        super().__init__();self.width=width;self.blocks=blocks;self.shared=shared
        self.encoder=nn.Linear(4,width);self.updates=[Update(width) for _ in range(1 if shared else blocks)]
        self.head1=nn.Linear(2*width+1,32);self.head2=nn.Linear(32,1)
    def __call__(self,inputs,depths=(1,),states=False):
        if min(depths)<1 or max(depths)>32:raise ValueError('Depth must be 1–32')
        if not self.shared and max(depths)>self.blocks:raise ValueError('Untied depth exceeds blocks')
        a,x,valid=inputs['a'],inputs['x'],inputs['valid'];e=mx.tanh(self.encoder(x))*valid;h=e
        out={};b=mx.arange(len(x));ab=inputs['ab']
        for depth in range(1,max(depths)+1):
            old=h;h=self.updates[0 if self.shared else depth-1](h,e,a,valid)
            if depth in depths:
                start=h[b,inputs['starts']];candidate=h[b[:,None],inputs['idx']]*(1-ab)
                start=mx.broadcast_to(start[:,None,:],candidate.shape)
                features=mx.concatenate([start,candidate,ab],-1)
                logits=self.head2(nn.silu(self.head1(features)))[...,0]
                logits=mx.where(inputs['cm'],logits,-1e9)
                rec={'logits':logits,'features':features,
                     'norm':mx.sqrt(mx.sum(h*h,(1,2))/mx.maximum(mx.sum(valid,(1,2))*self.width,1)),
                     'drift':mx.sqrt(mx.sum((h-old)**2,(1,2))/mx.maximum(mx.sum(valid,(1,2))*self.width,1))}
                if states:rec['states']=h
                out[depth]=rec
        return out
    def parameter_count(self):return sum(v.size for _,v in tree_flatten(self.trainable_parameters()))

def scorer_features(record,mask,depth):
    logits=record['logits'];p=mx.softmax(logits,-1);center=mx.sum(mx.where(mask,logits,0),-1,keepdims=True)/mx.sum(mask,-1,keepdims=True)
    # Raw probabilities, centered clipped logits and known compute budget; no outcomes.
    return mx.concatenate([record['features'],p[...,None],mx.clip(logits-center,-20,20)[...,None],mx.full((*logits.shape,1),depth/8)],-1)

class Scorer(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.layer1=nn.Linear(2*width+4,32);self.layer2=nn.Linear(32,1)
    def __call__(self,features):return self.layer2(nn.silu(self.layer1(features)))[...,0]
    def parameter_count(self):return sum(v.size for _,v in tree_flatten(self.trainable_parameters()))

def set_loss(logits,target):
    return mx.mean(mx.logsumexp(logits,-1)-mx.logsumexp(mx.where(target,logits,-1e9),-1))

def score_loss(scores,target,mask,objective):
    if objective=='correctness':
        loss=mx.maximum(scores,0)-scores*target+mx.log1p(mx.exp(-mx.abs(scores)))
        return mx.sum(loss*mask)/mx.sum(mask)
    if objective=='ranking':
        pairs=target[:,:,None] & ~target[:,None,:] & mask[:,:,None] & mask[:,None,:]
        delta=scores[:,None,:]-scores[:,:,None]
        loss=mx.maximum(delta,0)+mx.log1p(mx.exp(-mx.abs(delta)))
        # Equal graph weighting; every graph includes abstain or a negative neighbor.
        return mx.mean(mx.sum(loss*pairs,(1,2))/mx.maximum(mx.sum(pairs,(1,2)),1))
    if objective=='direct':return set_loss(mx.where(mask,scores,-1e9),target)
    raise ValueError(objective)

def build(condition,seed):
    mx.random.seed(seed)
    if condition in ('one_pass','recurrent','extra_budget'):return GraphPolicy()
    if condition=='ff8':return GraphPolicy(32,8,False)
    if condition=='ff8_params':return GraphPolicy(11,8,False)
    raise ValueError(condition)
