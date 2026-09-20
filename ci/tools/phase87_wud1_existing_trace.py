#!/usr/bin/env python3
"""Localize WU-D1's existing model trace difference without re-running GPUs."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics


def analyze(path):
    rows = sorted(csv.DictReader(path.open()), key=lambda r: int(r['Start_Timestamp']))
    # Both recorded MTP-off runs finish with 127 complete decode transitions.
    # Check the full repeated launch signature rather than assuming CSV order.
    rows = rows[-127 * 1171:]
    groups = collections.defaultdict(list)
    for token in range(127):
        block = rows[token * 1171:(token + 1) * 1171]
        assert len(block) == 1171
        assert 'embedding_gather' in block[2]['Kernel_Name']
        assert 'fixed_topk_final_support' in block[-2]['Kernel_Name']
        nv = [r for r in block if r['Kernel_Name'] == 'sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1']
        assert len(nv) == 168
        for index, row in enumerate(nv):
            layer, role = divmod(index, 3)
            assert int(row['Grid_Size_X']) == (40960 if role == 2 else 139264)
            ns = int(row['End_Timestamp']) - int(row['Start_Timestamp'])
            assert ns > 0
            groups[(layer, role)].append(ns / 1000)
    return {'path': str(path), 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
            'rows': [{'layer': layer, 'role': ['gate','up','down'][role],
                      'layer_mod4': layer % 4, 'samples': len(v),
                      'mean_us': statistics.mean(v), 'median_us': statistics.median(v)}
                     for (layer, role), v in sorted(groups.items())]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--current', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    old, current = analyze(a.old), analyze(a.current)
    groups = collections.defaultdict(list)
    for x,y in zip(old['rows'], current['rows'], strict=True):
        assert (x['layer'],x['role']) == (y['layer'],y['role'])
        groups[(x['layer_mod4'], x['role'])].append((x['mean_us'], y['mean_us']))
    summary = [{'layer_mod4': k[0], 'role': k[1],
                'old_mean_us': statistics.mean(x[0] for x in v),
                'current_mean_us': statistics.mean(x[1] for x in v),
                'delta_us': statistics.mean(x[1]-x[0] for x in v)}
               for k,v in sorted(groups.items())]
    a.output.write_text(json.dumps({'state':'PASS','old':old,'current':current,
        'summary':summary,'caveat':'Existing full-model profiled timings; grouping is observational, not a new causal control.'}, indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
