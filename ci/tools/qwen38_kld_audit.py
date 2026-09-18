#!/usr/bin/env python3
"""Audit the frozen Qwen3.8 capture matrix against inputs and current raw bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def expected_captures():
    names = [('llama-bf16-fp16', 'short'), ('llama-bf16-fp16-chunk1', 'short'),
             ('llama-bf16-fp16-long', 'long')]
    names += [(f'exl3-{bits}-fp16', 'short') for bits in [3, 4, 5]]
    names += [(name, 'short') for name in ['sllm-nvfp4-fp16', 'sllm-mxfp6-fp16-gfx1030',
                                         'sllm-mxfp8-fp16-gfx1030', 'vllm-fp8-bf16']]
    names += [('llama-bf16-q8_0', 'short'), ('llama-bf16-q4_0', 'short'), ('llama-bf16-q4_0-long', 'long')]
    names += [(f'exl3-{bits}-kv{kv}v{kv}', 'short') for bits in [3, 4, 5] for kv in [8, 4]]
    names += [('exl3-4-kv6v6', 'short'), ('exl3-4-kv4v8', 'short'),
              ('exl3-4-kv16v16-long', 'long'), ('exl3-4-kv4v4-long', 'long')]
    for kind in ['mxfp6', 'mxfp8']:
        names += [(f'sllm-{kind}-fp16-gfx1030-chunk{chunk}-run2', 'short') for chunk in [32, 64]]
        names += [(f'sllm-{kind}-kv-mxfp8-{kv}-gfx1030-chunk32-run2', 'short') for kv in ['e4', 'e5']]
    names += [('sllm-mxfp6-fp16-gfx1030-chunk32-long-run2', 'long'),
              ('sllm-mxfp6-kv-mxfp8-e4-gfx1030-chunk32-long-run2', 'long')]
    names += [('vllm-fp8-e4m3', 'short'), ('vllm-bf16-long', 'long'), ('vllm-fp8-e4m3-long', 'long')]
    names += [(f'sllm-nvfp4-{kv}-gfx1201-chunk32', 'short') for kv in ['fp16', 'kv-mxfp8-e4']]
    names += [(f'sllm-nvfp4-{kv}-gfx1030-chunk32', 'short') for kv in ['fp16', 'kv-mxfp8-e5']]
    assert len(names) == 40 and len(set(names)) == 40
    return names


def file_identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'capture-audit.json'
    cache = json.loads(output.read_text()).get('files', {}) if output.exists() else {}
    inputs = {scope: json.loads((root / name).read_text())['cases'] for scope, name in
              [('short', 'cases-v1.json'), ('long', 'cases-long-v1.json')]}
    records, files, pending = [], {}, []
    for name, scope in expected_captures():
        folder = root / 'results' / name
        manifest_path = folder / 'manifest.json'
        if not manifest_path.exists():
            pending.append(name)
            continue
        manifest = json.loads(manifest_path.read_text())
        expected = inputs[scope]
        observed = manifest['cases']
        assert [x['id'] for x in observed] == [x['id'] for x in expected], name
        if name.startswith('sllm-'):
            execution = json.loads((folder / 'execution.json').read_text())
            if execution.get('returncode') is None:
                pending.append(name)
                continue
            assert execution['returncode'] == 0 and execution['state'] == 'PASS', name
            report_path = folder / ('adapter.stdout' if 'nvfp4' in name else 'worker.stdout')
            report = json.loads(report_path.read_text())
            assert report['target'] in ['gfx1030', 'gfx1201'], name
            for case in report['cases']:
                assert case['all_dispatches_hip'] and not case['fallback_used'], name
                assert case['kernel_dispatch_count'] > 0, name
        elif name.startswith('llama-'):
            assert manifest['selected_devices'] == ['ROCm0', 'ROCm1'], name
            assert manifest['backend_error_count'] == 0, name
        elif name.startswith('vllm-'):
            assert manifest['state'] == 'PASS' and manifest['tp'] == 1, name
        elif name.startswith('exl3-'):
            assert manifest['complete'] and manifest['gpu'].startswith('gfx1201'), name
        size_total = 0
        rows_total = 0
        for item, case in zip(observed, expected):
            tokens = case['token_ids']
            positions = sorted(case.get('positions', range(len(tokens))))
            assert item['positions'] == positions, (name, case['id'])
            shape = item.get('shape', [item.get('rows'), item.get('vocab_size')])
            assert shape == [len(positions), 248320], (name, case['id'], shape)
            hashes = [hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
                      hashlib.sha256(struct.pack('<' + 'i' * len(tokens), *tokens)).hexdigest()]
            token_hash = item.get('input_token_ids_sha256', item.get('token_ids_sha256', '')).removeprefix('sha256:')
            assert token_hash in hashes, (name, case['id'])
            path = folder / item['logits_file']
            identity = file_identity(path)
            assert identity[2] == len(positions) * 248320 * 4, path
            key = str(path)
            prior = cache.get(key)
            if prior and prior['identity'] == identity:
                check = prior
            else:
                digest = hashlib.sha256()
                nonfinite = 0
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(16 << 20), b''):
                        digest.update(block)
                        nonfinite += int((~np.isfinite(np.frombuffer(block, dtype='<f4'))).sum())
                assert identity == file_identity(path), f'File changed while read: {path}'
                check = dict(identity=identity, sha256=digest.hexdigest(), nonfinite_count=nonfinite)
            assert check['nonfinite_count'] == 0, path
            declared = item.get('sha256', item.get('logits_file_sha256'))
            if declared:
                assert check['sha256'] == declared.removeprefix('sha256:'), path
            files[key] = check
            size_total += identity[2]
            rows_total += len(positions)
        records.append(dict(name=name, scope=scope, cases=len(observed), rows=rows_total,
                            bytes=size_total, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()))
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(expected_captures=40, validated=records, pending=pending,
                                             files={**cache, **files}), indent=2) + '\n')
        temporary.replace(output)
        print(json.dumps(records[-1]), flush=True)
    unavailable = []
    for name, log_path, message in [
        ('NVFP4 E5 KV on gfx1201', root / 'results/sllm-nvfp4-kv-mxfp8-e5-gfx1201-chunk32/adapter.stderr',
         'OcpE5M2 is incompatible with target gfx1201'),
        ('vLLM serialized FP8 checkpoint with E5 KV', root / 'logs/vllm-fp8-e5m2-patched-full.log',
         'fp8_e5m2 kv-cache is not supported with fp8 checkpoints'),
    ]:
        assert message in log_path.read_text(), f'Missing rejection evidence: {name}'
        unavailable.append(dict(name=name, evidence=str(log_path), reason=message))
    final = dict(expected_captures=40, validated=records, pending=pending, files=files, unavailable=unavailable,
                 state='COMPLETE_RAW_AUDIT' if not pending else 'PENDING_CAPTURES',
                 note='Current full-vocabulary bytes/input mapping and capture metadata. Final completion also checks execution ledgers, numerical comparisons, and recorded unsupported variants.')
    output.write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps(dict(validated=len(records), pending=pending)), flush=True)


if __name__ == '__main__':
    main()
