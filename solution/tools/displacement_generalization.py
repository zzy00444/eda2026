"""Check reserved exact queries and a position-disjoint M-correction split."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from displacement_analyze import OUT, describe, points, split
from project_paths import REPO_ROOT


def mean_delta_interval(delta, position):
    _, group = np.unique(position, return_inverse=True)
    counts = np.bincount(group)
    sums = np.bincount(group, weights=delta)
    mean = float(delta.mean())
    g = len(counts)
    se = float(np.sqrt(g/(g-1) * np.sum((sums - mean*counts)**2)) / len(delta))
    return {'mean_accuracy_delta': mean, 'source_coordinate_cluster_95_interval': [mean-1.96*se, mean+1.96*se],
            'source_coordinate_groups': g, 'note': 'Approximate cluster interval, conditional on this evaluated dataset'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    truth = pd.read_csv(OUT / 'oracle_answers.csv')
    report = {'oracle_rows': len(truth), 'oracle_new_rows': len(truth)-8, 'candidates': {}}
    for name in args.names:
        result = pd.read_csv(OUT / (name + '_oracle.csv'))
        assert result.From.equals(truth.From) and result.To.equals(truth.To)
        report['candidates'][name] = {
            'official_eight': describe(truth.delay.to_numpy()[:8], result.delay.to_numpy()[:8]),
            'reserved_48': describe(truth.delay.to_numpy()[8:], result.delay.to_numpy()[8:])}
    geometry = np.load(OUT / 'geometry.npz')
    y = geometry['y']
    base = np.load(OUT / 'displacement_baseline_prediction.npy')
    trace = np.fromfile(OUT / 'displacement_trace_trace.bin', '<i4').reshape(-1, 4)
    pairs = trace[:, 0]
    main_train, valid, test = split(len(y))
    position = geometry['sx'].astype(np.int64)*550 + geometry['sy']
    reserved = np.zeros(len(y), bool)
    spec = json.loads((OUT / 'oracle_spec.json').read_text())
    for sx, sy in spec['reserved_source_coordinates']:
        reserved |= (geometry['sx'] == sx) & (geometry['sy'] == sy)
    held = ((geometry['sx'].astype(np.int64)*1009 + geometry['sy'].astype(np.int64)*9176) % 5) == 0
    train = main_train[~held[main_train] & ~reserved[main_train]]
    assess = main_train[held[main_train] & ~reserved[main_train]]
    selected = json.loads((REPO_ROOT / 'docs/displacement_refit.json').read_text())['selected']
    if selected['components'] == 0:
        residual = y.astype(float) - base
        weight = (2000 / np.maximum(y[train], 200)) / np.maximum(np.abs(residual[train]), 4.)
        weight /= weight.mean()
        numerator = np.bincount(pairs[train], weights=weight*residual[train], minlength=59904)
        denominator = np.bincount(pairs[train], weights=weight, minlength=59904) + selected['penalty']
        correction = np.rint(numerator/denominator)
        pred = np.maximum(0, base[assess] + correction[pairs[assess]])
        delta = points(y[assess], pred) - points(y[assess], base[assess])
        report['M_only_position_disjoint_check'] = {
            'fit_rows': len(train), 'assess_rows': len(assess), 'frozen_penalty': selected['penalty'],
            'baseline': describe(y[assess], base[assess]), 'updated': describe(y[assess], pred),
            **mean_delta_interval(delta, position[assess]),
            'scope': 'M is refitted on other source coordinates; final deployed M uses the main training split'}
    for name in args.names:
        prediction_path = OUT / (name + '_prediction.npy')
        if prediction_path.exists():
            pred = np.load(prediction_path)
            report['candidates'][name]['paired_valid_delta_vs_baseline'] = mean_delta_interval(
                points(y[valid], pred[valid])-points(y[valid], base[valid]), position[valid])
            report['candidates'][name]['paired_test_delta_vs_baseline'] = mean_delta_interval(
                points(y[test], pred[test])-points(y[test], base[test]), position[test])
    (REPO_ROOT / 'docs/displacement_generalization.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
