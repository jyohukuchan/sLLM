#!/usr/bin/env python3
"""Validate and summarize WU-D2 decay/intervention samples without raw payloads."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics


def fields(line):
    return dict(x.split('=',1) for x in shlex.split(line) if '=' in x)


def summarize(path):
    lines=path.read_text().splitlines()
    meta=next(fields(line) for line in lines if line.startswith('probe='))
    if meta['target'] != meta['runtime_gfx']:
        raise ValueError('exact GPU mismatch')
    if any('status=FAIL' in line for line in lines):
        raise ValueError('failed numerical check')
    oracles=[fields(x) for x in lines if x.startswith('numerical_oracle ')]
    if len(oracles)!=2 or any(x.get('status')!='PASS' for x in oracles):
        raise ValueError('missing two-shape numerical oracle')
    raw=[fields(x) for x in lines if x.startswith('sample ')]
    samples=int(meta['samples']); rounds=int(meta['rounds'])
    ns=[int(x) for x in meta['n_list'].split(',')]
    preds=['isolated','staged32-qhead6-split32','gqa-kvhead1-split128']
    interventions=['none'] if meta['intervention_selection']=='none' else ['none','quantize','rmsnorm','copy']
    shapes=[x['shape'] for x in oracles]
    intervention_oracles=[fields(x) for x in lines if x.startswith('intervention_oracle ')]
    if interventions != ['none']:
        required={(shape,kind) for shape in shapes for kind in ['quantize','rmsnorm','copy']}
        actual={(x['shape'],x['kind']) for x in intervention_oracles if x.get('status')=='PASS'}
        if actual != required or len(intervention_oracles)!=len(required):
            raise ValueError('missing intervention oracle')
    output_checks=[fields(x) for x in lines if x.startswith('output_compare ')]
    finite_checks=[fields(x) for x in lines if x.startswith('output_finite ')]
    if len(output_checks)<len(raw) or len(finite_checks)<len(raw):
        raise ValueError('missing measured-sequence output checks')
    expected={(s,p,i,n,str(r),o) for s in shapes for p in preds for i in interventions for n in ns for r in range(rounds) for o in ['AB','BA']}
    grouped=collections.defaultdict(list)
    for row in raw:
        n=int(row['n']);v=[float(x) for x in row['nvfp4_us'].split(',')]
        if len(v)!=n or any(not math.isfinite(x) or x<=0 for x in v):
            raise ValueError('invalid per-dispatch times')
        if abs(sum(v)-float(row['sum_us']))>max(.02,n*.0011):
            raise ValueError('sum accounting mismatch')
        key=(row['shape'],row['predecessor'],row['intervention'],n,row['round'],row['order'])
        grouped[key].append((int(row['sample']),v,float(row['wall_us'])))
    if set(grouped)!=expected:
        raise ValueError('incomplete condition matrix')
    cells={}
    for key,values in grouped.items():
        if sorted(x[0] for x in values)!=list(range(samples)):
            raise ValueError('missing or duplicate sample')
        n=key[3]
        cells[key]={'dispatch_us':[statistics.median(x[1][j] for x in values) for j in range(n)],
                    'sum_us':statistics.median(sum(x[1]) for x in values),
                    'wall_us':statistics.median(x[2] for x in values)}
    summary=[];comparisons=[]
    for s in shapes:
        for p in preds:
            for i in interventions:
                for n in ns:
                    keys=[(s,p,i,n,str(r),o) for r in range(rounds) for o in ['AB','BA']]
                    summary.append({'shape':s,'predecessor':p,'intervention':i,'n':n,'cells':len(keys),
                        'dispatch_us':[statistics.median(cells[k]['dispatch_us'][j] for k in keys) for j in range(n)],
                        'sum_us':statistics.median(cells[k]['sum_us'] for k in keys),'wall_us':statistics.median(cells[k]['wall_us'] for k in keys)})
                    if p=='isolated':continue
                    ratios=[];deltas=[]
                    for j in range(n):
                        paired=[(cells[k]['dispatch_us'][j],cells[(s,'isolated',i,n,k[4],k[5])]['dispatch_us'][j]) for k in keys]
                        ratios.append(statistics.median(a/b for a,b in paired));deltas.append(statistics.median(a-b for a,b in paired))
                    recovery=next((j+1 for j in range(n) if all(abs(x-1)<=.02 for x in ratios[j:])),None)
                    comparisons.append({'shape':s,'predecessor':p,'intervention':i,'n':n,'paired_median_ratio_vs_isolated':ratios,
                        'paired_median_delta_us_vs_isolated':deltas,'recovery_dispatch_within_2pct':recovery})
    return {'state':'PASS','source':str(path),'source_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
        'metadata':meta,'sample_count':len(raw),'oracles':oracles,'summary':summary,'comparisons':comparisons,
        'recovery_definition':'Descriptive threshold only: first 1-based dispatch with all subsequent median paired ratios within +/-2% of matched isolated; null means not recovered within measured sequence. Not an adoption gate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=summarize(a.input);a.output.write_text(json.dumps(d,indent=2)+'\n')
    print('PASS',d['metadata']['target'],'samples',d['sample_count'])
    for x in d['comparisons']:
        if x['predecessor']=='gqa-kvhead1-split128':print(json.dumps(x))


if __name__=='__main__':main()
