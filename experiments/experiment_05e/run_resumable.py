"""Optional durable I/O wrapper around the unchanged, frozen scientific pipeline.

Use in a fresh run directory. Completed epoch transactions include best/latest
weights, optimizer state and summary. Recovery discards an uncommitted newer
state by restoring the last hash-verified transaction. No objective changes.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def atomic_bytes(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.pending-',suffix=path.suffix,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(content);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


class EpochJournal:
    def __init__(self, root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)

    def commit(self, key, files):
        directory=Path(tempfile.mkdtemp(prefix=key+'-',dir=self.root))
        hashes={}
        for name,path in files.items():
            content=Path(path).read_bytes();atomic_bytes(directory/name,content)
            hashes[name]=hashlib.sha256(content).hexdigest()
        metadata={'directory':directory.name,'files':{name:str(path) for name,path in files.items()},'sha256':hashes}
        atomic_bytes(self.root/(key+'.json'),json.dumps(metadata,sort_keys=True).encode())
        # Keep only the committed transaction for this fit, after the pointer switches.
        for old in self.root.glob(key+'-*'):
            if old.is_dir() and old!=directory:shutil.rmtree(old)

    def recover(self):
        restored=[]
        for pointer in self.root.glob('*.json'):
            meta=json.loads(pointer.read_text());directory=self.root/meta['directory']
            contents={name:(directory/name).read_bytes() for name in meta['files']}
            assert all(hashlib.sha256(contents[name]).hexdigest()==meta['sha256'][name] for name in contents)
            for name,path in meta['files'].items():atomic_bytes(path,contents[name])
            restored.append(pointer.stem)
        return restored


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','extract'])
    p.add_argument('--config',required=True)
    p.add_argument('--splits',nargs='*')
    a=p.parse_args();config=json.loads(Path(a.config).read_text())
    deadline=time.monotonic()+config['active_budget_seconds']
    import numpy as np
    from latent_loop.rendezvous import features,study
    from latent_loop.rendezvous.data import ROOT
    def atomic_array_writer(original):
        def writer(path,*args,**kwargs):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            directory=Path(tempfile.mkdtemp(prefix='.pending-',dir=path.parent))
            name=directory/('array'+path.suffix)
            try:
                original(str(name),*args,**kwargs)
                with open(name,'rb') as f:os.fsync(f.fileno())
                os.replace(name,path)
                if time.monotonic()>deadline:
                    raise TimeoutError('Configured active runtime reached; completed tensor write preserved')
            finally:
                shutil.rmtree(directory,ignore_errors=True)
        return writer
    if a.command=='extract':
        np.savez_compressed=atomic_array_writer(np.savez_compressed)
        features.extract(config,a.splits or ['train','validation'])
        return
    if (ROOT/'final_openings.json').exists():
        raise RuntimeError('Final set already opened: no fitting or recovery mutation permitted')
    import mlx.core as mx
    from latent_loop.backbone import environment
    environment(config['memory_gb'])
    from latent_loop.rendezvous.models import CandidateHead
    journal=EpochJournal(study.ART/'epoch_transactions');journal.recover()
    mx.savez=atomic_array_writer(mx.savez)
    original_weights=CandidateHead.save_weights
    def weights(self,path):
        return atomic_array_writer(lambda name:original_weights(self,name))(path)
    CandidateHead.save_weights=weights
    def save(path,value):
        atomic_bytes(path,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
        if Path(path).parent==ROOT/'training':
            key=Path(path).stem
            paths={'summary.json':Path(path),'best.safetensors':study.ART/'checkpoints'/f'{key}.safetensors','latest.safetensors':study.ART/'checkpoints'/f'{key}_latest.safetensors','optimizer.npz':study.ART/'checkpoints'/f'{key}_resume.npz'}
            journal.commit(key,paths)
    study.save=save
    study.train(config)


if __name__=='__main__':main()
