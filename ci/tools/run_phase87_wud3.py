#!/usr/bin/env python3
"""Run focused WU-D3 candidate evidence; large sample arrays stay local."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

UUIDS={'gfx1030':'GPU-76a08c022586fed6','gfx1201':'GPU-a8e9ddefa2d60f55'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=UUIDS,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['performance','boundary'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ROCR_VISIBLE_DEVICES=UUIDS[a.target],LD_LIBRARY_PATH='/opt/rocm/lib');env.pop('HIP_VISIBLE_DEVICES',None)
    jobs=[]
    if a.mode=='performance':
        candidates=['c1-tile32','c2-block-remap','c3-prefetch'] if a.target=='gfx1030' else ['control']
        jobs=[(c,8256,1,0,16,300,3,9) for c in candidates]
    else:
        jobs=[('c2-block-remap',length,m,0,1,0,1,1) for length in [1024,1025,8191,8192,8193,8256] for m in [1,2,3]]
        jobs += [('c2-block-remap',length,3,pattern,1,0,1,1) for length in [1025,8193] for pattern in [1,2]]
    report={'state':'RUNNING','target':a.target,'mode':a.mode,'probe':str(a.probe),
            'probe_sha256':hashlib.file_digest(a.probe.open('rb'),'sha256').hexdigest(),'jobs':[]}
    path=a.output/'execution.json'
    for index,(candidate,length,m,pattern,repeats,warmup,rounds,samples) in enumerate(jobs):
        log=a.output/f'{index:02d}-{candidate}-l{length}-m{m}-p{pattern}.log'
        cmd=[str(a.probe.resolve()),'--target='+a.target,'--candidate='+candidate,'--shape=both',
             f'--length={length}',f'--m={m}',f'--pattern={pattern}',f'--nv-repeats={repeats}',
             f'--warmup-ms={warmup}',f'--rounds={rounds}',f'--samples={samples}']
        item={'command':cmd,'log':str(log)};report['jobs'].append(item);path.write_text(json.dumps(report,indent=2)+'\n')
        print('START',a.target,index,candidate,length,m,pattern,flush=True)
        with log.open('w') as out:r=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
        item['exit_code']=r.returncode;item['log_sha256']=hashlib.file_digest(log.open('rb'),'sha256').hexdigest()
        path.write_text(json.dumps(report,indent=2)+'\n')
        if r.returncode:raise RuntimeError('probe failure: '+str(log))
        rows=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
        oracle=[x for x in rows if x['kind'] in ['attention_oracle','nv_oracle']]
        perf=[x for x in rows if x['kind']=='performance']
        if len(oracle)!=4 or len(perf)!=2 or not any(x['kind']=='cleanup' and x['state']=='PASS' for x in rows):
            raise ValueError('incomplete shape/oracle matrix')
        if any(x.get('state')!='PASS' for x in oracle):raise ValueError('failed oracle')
        for x in oracle:
            if x['kind']=='attention_oracle' and x.get('candidate_vs_control_bitwise') is not True:
                raise ValueError('candidate is not N0: '+str(log))
        for x in perf:
            expected=rounds*2*samples*(2 if candidate=='control' else 1)
            if len(x['control_nv_calls_ms'])!=expected:raise ValueError('incomplete performance samples')
            if any(len(y)!=repeats for y in x['control_nv_calls_ms']):raise ValueError('incomplete NV sequence')
        item['oracles']=oracle
        item['performance']=[{k:v for k,v in x.items() if not isinstance(v,list)} for x in perf]
        path.write_text(json.dumps(report,indent=2)+'\n');print('PASS',a.target,index,flush=True)
    report['state']='PASS';path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
