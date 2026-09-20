#!/usr/bin/env python3
"""Collect bounded WU-D1 measured regions and join NVFP4 counters to conditions."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess

SYMBOL = 'sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1'
PASSES = {
    'trace': [],
    'fetch': ['FETCH_SIZE','GRBM_COUNT'],
    'cache': ['L2CacheHit','MemUnitBusy','GRBM_GUI_ACTIVE','GRBM_COUNT'],
    'instruction': ['SQ_INSTS_VALU','SQ_WAVES','SQ_INSTS_SMEM','SQ_INST_CYCLES_VMEM','GRBM_COUNT'],
    'wait': ['WAVE_DEP_WAIT','OccupancyPercent','GRBM_GUI_ACTIVE','GRBM_COUNT'],
    'ea': ['GRBM_EA_BUSY','GRBM_COUNT'],
}


def digest(path):
    return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()


def analyze(folder, counters):
    output = folder/'stdout.log'
    samples = [dict(x.split('=',1) for x in shlex.split(line)[1:])
               for line in output.read_text().splitlines() if line.startswith('sample ')]
    if len(samples) != 160 or 'status=FAIL' in output.read_text():
        raise ValueError('incomplete samples or failed oracle')
    trace = next(folder.glob('traces/**/*kernel_trace.csv'))
    rows = sorted((r for r in csv.DictReader(trace.open()) if r['Kernel_Name']==SYMBOL),
                  key=lambda r:int(r['Start_Timestamp']))
    if len(rows) != len(samples):
        raise ValueError(f'selected NVFP4 dispatch count {len(rows)} != sample count {len(samples)}')
    metrics = collections.defaultdict(dict)
    hashes = {str(output):digest(output),str(trace):digest(trace)}
    if counters:
        counter_path = next(folder.glob('traces/**/*counter_collection.csv'))
        hashes[str(counter_path)] = digest(counter_path)
        for row in csv.DictReader(counter_path.open()):
            if row['Kernel_Name'] != SYMBOL:
                raise ValueError('counter filter selected another kernel')
            key = row['Dispatch_Id']
            value = float(row['Counter_Value'])
            if not math.isfinite(value):
                raise ValueError('nonfinite counter')
            if row['Counter_Name'] in metrics[key]:
                raise ValueError('duplicate metric for dispatch')
            metrics[key][row['Counter_Name']] = value
        if len(metrics) != len(rows):
            raise ValueError('missing counter dispatch')
    groups = collections.defaultdict(list)
    for sample,row in zip(samples,rows,strict=True):
        expected_grid = '139264' if sample['shape'].startswith('wide') else '40960'
        if row['Grid_Size_X'] != expected_grid:
            raise ValueError('condition-to-dispatch order mismatch')
        us = (int(row['End_Timestamp'])-int(row['Start_Timestamp']))/1000
        values = {'kernel_us':us}
        if counters:
            values.update(metrics[row['Dispatch_Id']])
            if not set(counters).issubset(values):
                raise ValueError('requested metric missing')
        groups[(sample['shape'],sample['mode'])].append(values)
    summary = []
    for (shape,mode), values in sorted(groups.items()):
        if len(values) != 16:
            raise ValueError('incomplete AB/BA cell count')
        summary.append({'shape':shape,'mode':mode,'count':len(values),
                        'mean':{k:statistics.mean(v[k] for v in values) for k in values[0]},
                        'median':{k:statistics.median(v[k] for v in values) for k in values[0]}})
    return {'state':'PASS','selected_dispatches':len(rows),'summary':summary,'hashes':hashes,
            'interpretation':'Profiled diagnostic counters; EA busy and wave wait do not identify MALL/DRAM latency.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--passes',default=','.join(PASSES))
    a = p.parse_args()
    env = dict(os.environ,ROCR_VISIBLE_DEVICES='GPU-76a08c022586fed6',LD_LIBRARY_PATH='/opt/rocm/lib')
    env.pop('HIP_VISIBLE_DEVICES',None)
    for name in a.passes.split(','):
        counters = PASSES[name]
        folder = a.output_dir/name
        folder.mkdir(parents=True,exist_ok=False)
        cmd = ['/opt/rocm/bin/rocprofv3','--selected-regions','true','--kernel-trace',
               '--kernel-include-regex','^'+SYMBOL+'$','--output-format','csv',
               '--output-directory',str(folder/'traces')]
        if counters:
            cmd += ['--pmc']+counters
        cmd += ['--',str(a.probe.resolve()),'--target=gfx1030','--shape=both','--mode=all',
                '--warmup-ms=300','--rounds=1','--samples=8']
        identity = {'command':cmd,'probe_sha256':digest(a.probe),'gpu_uuid':env['ROCR_VISIBLE_DEVICES']}
        (folder/'execution.json').write_text(json.dumps(identity,indent=2)+'\n')
        print('START',name,flush=True)
        with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
            result = subprocess.run(cmd,env=env,stdout=out,stderr=err)
        identity['exit_code']=result.returncode
        (folder/'execution.json').write_text(json.dumps(identity,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}')
        report = analyze(folder,counters)
        (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('COMPLETE',name,'dispatches',report['selected_dispatches'],flush=True)


if __name__ == '__main__':
    main()
