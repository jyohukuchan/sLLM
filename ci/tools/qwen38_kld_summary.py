#!/usr/bin/env python3
"""Collect full-corpus KLD reports; execution success is audited separately."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    scopes = {
        hashlib.sha256((root / name).read_bytes()).hexdigest(): scope
        for name, scope in [('cases-v1.json', 'short'), ('cases-long-v1.json', 'long')]
    }
    expected_vocab = hashlib.sha256((root / 'vocab-v1.json').read_bytes()).hexdigest()
    rows = {}
    for path in sorted((root / 'results').glob('kld-*.json')):
        result = json.loads(path.read_text())
        scope = scopes.get(result.get('inputs_sha256'))
        if scope is None:
            continue
        if result['vocab_sha256'] != expected_vocab:
            raise ValueError(f'Vocabulary differs: {path}')
        reference = Path(result['reference']).name
        candidate = Path(result['candidate']).name
        for name in [reference, candidate]:
            if not (root / 'results' / name / 'manifest.json').is_file():
                raise ValueError(f'Missing capture manifest for {name}')
        stats = result['primary_excluding_repeat_control']
        expected_count = 2632 if scope == 'short' else 514
        if stats['count'] != expected_count:
            raise ValueError(f'Position count differs: {path}')
        row = dict(scope=scope, reference=reference, candidate=candidate,
                   positions=stats['count'], valid_vocabulary=result['valid_vocab_size'],
                   mean_kld=stats['mean'], median_kld=stats['median'],
                   p95_kld=stats['p95'], p99_kld=stats['p99'], max_kld=stats['max'],
                   top1_agreement=result['primary_top1_agreement'])
        if not all(math.isfinite(row[key]) for key in
                   ['mean_kld', 'median_kld', 'p95_kld', 'p99_kld', 'max_kld', 'top1_agreement']):
            raise ValueError(f'Nonfinite metric: {path}')
        key = (scope, reference, candidate)
        if key in rows:
            prior = {k: v for k, v in rows[key].items() if k != 'reports'}
            if prior != row:
                raise ValueError(f'Conflicting duplicate comparisons: {path}')
            rows[key]['reports'].append(str(path))
        else:
            rows[key] = dict(row, reports=[str(path)])
    ordered = [rows[key] for key in sorted(rows)]
    if not ordered:
        raise ValueError('No full-corpus comparison reports selected')
    output = root / 'results' / 'comparison-summary.json'
    output.write_text(json.dumps({
        'definition': 'KL(reference || candidate), temperature 1, nats',
        'note': 'Metrics only. Capture exit status/GPU evidence must be audited separately. Smoke and derived-prefix tests excluded.',
        'comparisons': ordered,
    }, indent=2, allow_nan=False) + '\n')
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        for row in ordered:
            writer.writerow(dict(row, reports=';'.join(row['reports'])))
    print(json.dumps({'comparisons': len(ordered), 'json': str(output), 'csv': str(csv_path)}))


if __name__ == '__main__':
    main()
