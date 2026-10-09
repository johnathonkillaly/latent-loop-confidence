"""Minimal model-independent decision contracts; no LLM integration or graph IDs."""
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
import numpy as np

class StateEncoder(Protocol):
    def encode(self,observation:Any)->Any: ...
class RecurrentUpdater(Protocol):
    def update(self,state:Any,observation:Any)->Any: ...
class BasePolicy(Protocol):
    def logits(self,state:Any,candidates:Sequence[Any])->np.ndarray: ...
class CandidateScorer(Protocol):
    def scores(self,state:Any,candidates:Sequence[Any],base_logits:np.ndarray,depth:int)->np.ndarray: ...

def normalize_scores(scores,mask=None):
    q=np.asarray(scores,dtype=np.float64);mask=np.ones_like(q,dtype=bool) if mask is None else np.asarray(mask)
    mean=np.sum(np.where(mask,q,0),axis=-1,keepdims=True)/np.sum(mask,axis=-1,keepdims=True)
    centered=q-mean;rms=np.sqrt(np.sum(np.where(mask,centered**2,0),axis=-1,keepdims=True)/np.sum(mask,axis=-1,keepdims=True))
    return centered/np.maximum(rms,1)

def fuse(logits,scores,weight,mask=None):
    # Subtracting logsumexp is candidate-constant, so raw logits give identical argmax.
    return np.asarray(logits,dtype=np.float64)+weight*normalize_scores(scores,mask)

@dataclass
class Decision:
    candidate_index:int
    base_index:int
    depth:int
    scores:np.ndarray

class DecisionController:
    def __init__(self,weight=1):self.weight=weight
    def decide(self,logits,quality,depth=1,tie_keys=None):
        from .data import choose
        z=np.asarray(logits)[None,:];q=np.asarray(quality)[None,:]
        keys=np.arange(z.shape[1])[None,:] if tie_keys is None else np.asarray(tie_keys)[None,:]
        scores=fuse(z,q,self.weight)
        return Decision(int(choose(scores,keys)[0]),int(choose(z,keys)[0]),depth,scores[0])

def example():
    """Toy observable vectors illustrate the contract, not an experimental result."""
    hidden=np.array([.2,.7]);candidates=np.array([[1.,0.],[0.,1.],[.5,.5]])
    original=np.array([.5,.4,.1]);quality=candidates@hidden
    decision=DecisionController(.5).decide(original,quality,depth=1,tie_keys=[13,7,19])
    # Optional additional computation on unchanged observation (toy updater).
    hidden=np.tanh(hidden+np.array([0.,.2]));later=candidates@hidden
    final=DecisionController(.5).decide(original,later,depth=2,tie_keys=[13,7,19])
    return {'base':decision.base_index,'scored':decision.candidate_index,'after_optional_update':final.candidate_index}
