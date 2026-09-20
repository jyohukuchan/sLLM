#!/usr/bin/env python3
"""Summarize the WU-D1 same-process synthetic-neighbor timing probe."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics


def summarize(path):
    records = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        parts = shlex.split(line)
        if not parts:
            continue
        item = dict(p.split('=', 1) for p in parts[1:] if '=' in p)
        if item.get('status') == 'FAIL':
            raise ValueError('failed correctness check: ' + line)
        records[parts[0]].append(item)
    metadata = next((dict(p.split("=", 1) for p in shlex.split(line) if "=" in p)
                     for line in path.read_text().splitlines() if line.startswith("probe=")), None)
    if metadata is None:
        raise ValueError("missing probe metadata")
    samples = records['sample']
    if not samples or not records['numerical_oracle'] or not records['output_compare']:
        raise ValueError('missing timing or numerical evidence')
    for row in records['numerical_oracle'] + records['output_compare'] + records['predecessor_oracle']:
        if row.get('status') != 'PASS':
            raise ValueError('incomplete numerical check')
    grouped = collections.defaultdict(list)
    for row in samples:
        value = float(row['nvfp4_us'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('non-positive timing')
        grouped[(row['shape'], row['mode'], row['round'], row['order'])].append(value)
    expected_samples = int(metadata['samples'])
    expected_rounds = int(metadata['rounds'])
    if any(len(values) != expected_samples for values in grouped.values()):
        raise ValueError('incomplete samples in timing cell')
    shapes = {row['shape'] for row in records['numerical_oracle']}
    modes = {row['label'] for row in records['output_compare']}
    expected_keys = {(s,m,str(r),o) for s in shapes for m in modes
                     for r in range(expected_rounds) for o in ('AB','BA')}
    if set(grouped) != expected_keys:
        raise ValueError('incomplete AB/BA timing matrix')
    cells = {key: statistics.median(values) for key,values in grouped.items()}
    means = collections.defaultdict(list)
    for (shape,mode,round_id,order), value in cells.items():
        means[(shape,mode)].append(value)
    summary = [{'shape':shape, 'mode':mode, 'median_of_cell_medians_us':statistics.median(values),
                'min_cell_us':min(values), 'max_cell_us':max(values), 'cells':len(values)}
               for (shape,mode),values in sorted(means.items())]
    comparisons = []
    for shape in sorted({key[0] for key in cells}):
        for control,candidate in [('isolated','staged32-qhead6-split32'),
                                  ('isolated','gqa-kvhead1-split32'),
                                  ('isolated','gqa-kvhead1-split128'),
                                  ('staged32-qhead6-split32','gqa-kvhead1-split32'),
                                  ('staged32-qhead6-split32','gqa-kvhead1-split128')]:
            paired = []
            for key, value in cells.items():
                if key[0] != shape or key[1] != control:
                    continue
                other = (shape,candidate,key[2],key[3])
                if other in cells:
                    paired.append({'round':int(key[2]),'order':key[3],
                                   'delta_us':cells[other]-value,
                                   'ratio':cells[other]/value})
            if paired:
                comparisons.append({'shape':shape,'control':control,'candidate':candidate,
                                    'pairs':paired,'median_delta_us':statistics.median(x['delta_us'] for x in paired),
                                    'median_ratio':statistics.median(x['ratio'] for x in paired)})
    return {'state':'PASS','input':str(path),'input_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
            'metadata':metadata,'sample_count':len(samples),'correctness_checks':{k:records[k] for k in
                ['numerical_oracle','output_compare','predecessor_oracle']},
            'summary':summary,'paired_comparisons':comparisons,
            'caveat':'Synthetic KV reader, not full attention. Event timings and profiled kernel timings have different observation overhead.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    d = summarize(a.input)
    a.output.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d['summary'],indent=2))


if __name__ == '__main__':
    main()
