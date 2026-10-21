"""Fit the existing tiny calibration table using training rows only.

The core already pays for this lookup/multiply with unity coefficients.
Validate three fixed shrink factors plus unity, then freeze the table.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from displacement_analyze import OUT, points, split
from displacement_assets import sha
from project_paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    geometry = np.load(OUT / 'geometry.npz')
    y = geometry['y']
    pred = pd.read_csv(OUT / (args.name + '_result.csv'), usecols=['delay']).delay.to_numpy(np.int32)
    train, valid, _ = split(len(y))
    reserved = np.zeros(len(y), bool)
    spec = json.loads((OUT / 'oracle_spec.json').read_text())
    for sx, sy in spec['reserved_source_coordinates']:
        reserved |= (geometry['sx'] == sx) & (geometry['sy'] == sy)
    train = train[~reserved[train] & (y[train] > 0) & (pred[train] > 0)]
    bins = np.searchsorted([700, 1000, 2000, 4000], pred, side='right')
    group = geometry['ti'] * 5 + bins
    ratio = y[train].astype(float) / pred[train]
    weight = pred[train].astype(float) / y[train]
    order = np.lexsort((ratio, group[train]))
    g = group[train][order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    ends = np.r_[starts[1:], len(g)]
    raw = np.ones(288*5, np.float64)
    counts = np.zeros(288*5, np.int32)
    for start, end in zip(starts, ends):
        indexes = order[start:end]
        gid = int(g[start])
        counts[gid] = end-start
        if end-start < 50:
            continue
        csum = np.cumsum(weight[indexes])
        median = ratio[indexes[np.searchsorted(csum, csum[-1]*.5)]]
        raw[gid] = 1 + (np.clip(median, .95, 1.05)-1) * (end-start) / (end-start+50)
    results, best = [], None
    for alpha in (0., .25, .5, 1.):
        factors = (1 + (raw-1)*alpha).astype(np.float32)
        corrected = np.floor(pred[valid].astype(float) * factors[group[valid]].astype(float) + .5)
        score = float(points(y[valid], corrected).mean())
        item = {'shrink_factor': alpha, 'valid_accuracy': score}
        results.append(item)
        print(item, flush=True)
        if best is None or score > best[0]:
            best = (score, alpha, factors)
    score, alpha, factors = best
    output = OUT / 'calibration_train.bin'
    factors.astype('<f4').tofile(output)
    report = {'base_candidate': args.name, 'fit_rows': len(train),
              'reserved_source_coordinates': spec['reserved_source_coordinates'],
              'minimum_group_count': 50, 'count_prior': 50,
              'objective': 'weighted median ratio with pred/truth weights; relative L1 proxy',
              'configs': results, 'selected_shrink_factor': alpha,
              'selected_valid_accuracy': score, 'output_sha256': sha(output),
              'label_scope': 'rng7 training rows only; validation selects shrink; test unused'}
    (REPO_ROOT / 'docs/displacement_train_calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    print('frozen calibration', alpha, sha(output), flush=True)


if __name__ == '__main__':
    main()
