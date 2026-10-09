"""Regenerate every dataset in a fresh directory and verify byte identity (CPU)."""
import tempfile
import shutil
import json
from pathlib import Path
from . import data

def main():
    original_exp,original_out=data.EXP,data.OUT
    expected=json.loads((original_exp/'dataset_manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='decision-bridge-reproduce-') as tmp:
        data.EXP=Path(tmp)/'experiment';data.OUT=Path(tmp)/'results';data.EXP.mkdir()
        shutil.copyfile(original_exp/'config.json',data.EXP/'config.json');data.generate()
        actual=json.loads((data.EXP/'dataset_manifest.json').read_text());checks={}
        for split,entry in expected['splits'].items():
            sha=actual['splits'][split]['sha256'];assert sha==entry['sha256'],split
            checks[split]={'sha256':sha,'count':entry['count'],'byte_identical':True}
        count=data.audit()
    data.EXP, data.OUT=original_exp,original_out
    data.write(original_out/'independent_data_reproduction.json',{'datasets':checks,'labels_verified':count,'environment':'CPU only; no MLX imported','completed':data.now()})

if __name__=='__main__':main()
