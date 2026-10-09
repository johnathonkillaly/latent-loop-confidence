"""Post-freeze metadata serialization repair; frozen numerical code unchanged."""
import time
import resource
import numpy as np
import mlx.core as mx
from latent_loop.planning.data import save_rows,write_json,digest,now
from latent_loop.scaling.data import OUT,EXP

def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    return value

def run():
    from latent_loop.scaling import diagnostics
    diagnostics.save_rows=lambda path,rows:save_rows(path,native(rows))
    t=time.perf_counter();diagnostics.run()
    write_json(OUT/'diagnostic_resources.json',{'completed':now(),'seconds':time.perf_counter()-t,'peak_mlx_gib':mx.get_peak_memory()/1024**3,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'rerun_reason':'Frozen diagnostic numeric computation succeeded, final NumPy coordinate serialization failed; rerun with metadata-only native-scalar export wrapper'})
    write_json(EXP/'diagnostic_export_amendment.json',{'recorded':now(),'frozen_diagnostics_sha256':digest('src/latent_loop/scaling/diagnostics.py'),'export_wrapper_sha256':digest(__file__),'change':'Convert NumPy scalar metadata recursively at save_rows boundary only; all frozen model/evaluation/diagnostic computation remains byte-identical','post_final_model_or_depth_selection':False})
if __name__=='__main__':run()
