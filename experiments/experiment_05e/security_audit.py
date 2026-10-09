"""Scan canonical reachable history and the proposed working snapshot; report no values."""
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

from latent_loop.rendezvous.study import save


def git(*args):
    return subprocess.check_output(['git',*args])


def main():
    patterns={
        'private_key':rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
        'github_token':rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})',
        'cloud_key':rb'(?:AKIA|ASIA)[A-Z0-9]{16}',
        'hf_token':rb'hf_[A-Za-z0-9]{30,}',
        'openai_token':rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}',
        'private_home_path':rb'/(?:Users|home)/[A-Za-z][A-Za-z0-9_.-]+/',
        'private_ipv4':rb'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b',
        'credential_assignment':rb'''(?i)(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*["']([A-Za-z0-9_-]{24,})''',
    }
    findings=[];blobs=0;compressed=0
    def inspect(name,content,origin):
        nonlocal compressed
        if content[:2]==b'\x1f\x8b':
            content=gzip.decompress(content);compressed+=1
        for category,pattern in patterns.items():
            if re.search(pattern,content):findings.append({'origin':origin,'file':name,'category':category})
    objects=git('rev-list','--objects','HEAD','v0.1.0','v0.2.0').decode().splitlines()
    names={line.split(' ',1)[0]:line.split(' ',1)[1] if ' ' in line else '' for line in objects}
    checked=subprocess.run(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input=('\n'.join(names)+'\n').encode(),stdout=subprocess.PIPE,check=True).stdout.decode().splitlines()
    blob_ids=[line.split()[0] for line in checked if line.split()[1]=='blob']
    batch=subprocess.run(['git','cat-file','--batch'],input=('\n'.join(blob_ids)+'\n').encode(),stdout=subprocess.PIPE,check=True).stdout
    position=0
    for oid in blob_ids:
        end=batch.index(b'\n',position);header=batch[position:end].split();size=int(header[2]);content=batch[end+1:end+1+size];position=end+2+size
        inspect(names[oid],content,'reachable_history');blobs+=1
    files=git('ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0')
    files=[name for name in files if name and Path(name).is_file()]
    largest=sorted([{'file':name,'bytes':Path(name).stat().st_size} for name in files],key=lambda r:r['bytes'],reverse=True)[:8]
    forbidden=[]
    for name in files:
        p=Path(name)
        if p.suffix in ('.safetensors','.npz','.pt','.pth','.pem') or p.name.startswith('.env') or name.startswith('artifacts/'):
            forbidden.append(name)
        inspect(name,p.read_bytes(),'working_snapshot')
    authors=git('log','--format=%an <%ae>|%cn <%ce>','HEAD').decode().splitlines()
    generic=all(line=='Latent Loop Confidence contributors <research@invalid.example>|Latent Loop Confidence contributors <research@invalid.example>' for line in authors)
    newfiles=[p for p in files if p.startswith(('src/latent_loop/rendezvous/','experiments/experiment_05e/','results/experiment_05e/','reports/experiment_05e/')) or p in ['reports/research_conclusion.md','tests/test_rendezvous.py'] or 'v0.3.0' in p]
    result={'scope':'All blobs reachable from canonical HEAD and immutable public release tags, plus all tracked/unignored proposed files. Unrelated private local refs are not publication candidates.','reachable_commits':len(git('rev-list','HEAD').splitlines()),'reachable_blobs':blobs,'gz_payloads_decompressed':compressed,'working_files_scanned':len(files),'new_release_files':len(newfiles),'findings_without_values':findings,'forbidden_checkpoint_or_cache_files':forbidden,'canonical_history_generic_git_identity':generic,'largest_files':largest,'new_file_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in newfiles if not p.endswith('security_audit.json')},'license':'MIT original source/generated tasks. No vendored third-party code. Qwen Apache-2.0 model/tokenizer downloaded separately, never committed. Existing dependency attribution retained.','destination':'https://github.com/johnathonkillaly/latent-loop-confidence','publication':'Prepared only; no push, new tag, or release action.'}
    # Findings are paths/categories only. Review any false positives before a release.
    save('results/experiment_05e/security_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('new_file_hashes','largest_files')},indent=2))


if __name__=='__main__':main()
