"""Create a bounded exact-reference suite, or verify its solver output.

The eight official paths check the imported solver. The additional 48 queries
reserve three source coordinates for structural generalization checks.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from displacement_analyze import OUT
from project_paths import DATA, REPO_ROOT


def prepare():
    tokens = (OUT / 'oracle/oracle_arch.txt').read_text().split()
    pos = tokens.index('NSRB')
    count = int(tokens[pos + 1])
    srbs = np.array([[int(tokens[pos + 3 + i*3]), int(tokens[pos + 4 + i*3])]
                     for i in range(count)], dtype=np.int32)
    ports = json.loads((DATA / 'arch/SRB_Port.json').read_text())['Port']
    ids = np.loadtxt(OUT / 'oracle/oracle_ids.txt', dtype=np.int32)
    sources, targets = ids[:208], ids[208:]
    queries = []
    with (DATA / 'data/delay_estimate_check.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            queries.append([row['From'], row['To'], int(row['Min Delay'])])
    reserved = []
    rng = np.random.default_rng(20261021)
    for desired in ((20, 25), (60, 275), (100, 525)):
        sx, sy = srbs[np.argmin(np.abs(srbs - desired).sum(axis=1))].tolist()
        reserved.append([sx, sy])
        near = np.argsort(np.abs(srbs - [sx, sy]).sum(axis=1))
        target_sites = list(near[[5, 30, 150, 600]]) + rng.choice(count, 4, replace=False).tolist()
        for source_index in (0, 207):
            for index, srb in enumerate(target_sites):
                tx, ty = srbs[srb].tolist()
                p0 = ports[int(sources[source_index])]['Name']
                p1 = ports[int(targets[(index * 37 + source_index) % len(targets)])]['Name']
                queries.append([f'SRB_{sx}_{sy}/{p0}', f'SRB_{tx}_{ty}/{p1}', -1])
    with (OUT / 'oracle_queries.csv').open('w', newline='') as stream:
        csv.writer(stream, lineterminator='\n').writerows(queries)
    (OUT / 'oracle_request.csv').write_text('From,To\n' + ''.join(f'{a},{b}\n' for a, b, _ in queries))
    spec = {'golden_rows': 8, 'new_rows': len(queries)-8, 'reserved_source_coordinates': reserved,
            'seed': 20261021, 'note': 'New synthetic endpoint pairs; coordinates excluded from residual fitting'}
    (OUT / 'oracle_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(spec)


def check():
    with (OUT / 'oracle_result.csv').open(newline='') as stream:
        rows = list(csv.reader(stream))
    spec = json.loads((OUT / 'oracle_spec.json').read_text())
    assert len(rows) == spec['golden_rows'] + spec['new_rows']
    for row in rows[:8]:
        assert int(row[2]) == int(row[3]), row
    assert all(int(row[3]) >= 0 for row in rows)
    with (OUT / 'oracle_answers.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['From', 'To', 'delay'])
        writer.writerows((row[0], row[1], int(row[3])) for row in rows)
    report = dict(spec, golden_exact_matches=8, rows_verified=len(rows),
                  limitation='Bounded structural smoke suite, not representative hidden-set accuracy')
    (REPO_ROOT / 'docs/displacement_oracle_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else check()
