"""Fit small regularized updates to existing M/U, with no new online features.

Only the rng7 training rows are used to fit; three oracle source coordinates
are excluded. Validation selects among nine fixed configurations. Predictions
used for screening keep the baseline candidate branch and integer rounding;
the selected candidate must subsequently be scored through C++.
"""
from __future__ import annotations

import json

import numpy as np

from displacement_analyze import OUT, points, split
from displacement_assets import sha
from project_paths import REPO_ROOT


def main():
    geometry = np.load(OUT / 'geometry.npz')
    trace = np.fromfile(OUT / 'displacement_trace_trace.bin', dtype='<i4').reshape(-1, 4)
    baseline = np.load(OUT / 'displacement_baseline_prediction.npy')
    y = geometry['y']
    assert len(trace) == len(y) and np.array_equal(trace[:, 3], baseline)
    assert (trace[:, :2] >= 0).all()
    pair, row = trace[:, 0], trace[:, 1]
    train, valid, _ = split(len(y))
    spec = json.loads((OUT / 'oracle_spec.json').read_text())
    reserved = np.zeros(len(y), bool)
    for sx, sy in spec['reserved_source_coordinates']:
        reserved |= (geometry['sx'] == sx) & (geometry['sy'] == sy)
    original_train_rows = len(train)
    train = train[~reserved[train]]
    with (OUT / 'oracle/oracle_vt.bin').open('rb') as stream:
        vh = np.fromfile(stream, dtype='<i4', count=6)
        scales = np.fromfile(stream, dtype='<f4', count=int(vh[0]))
    vt = np.memmap(OUT / 'oracle/oracle_vt.bin', dtype='<i2', mode='r',
                   offset=24 + 4*int(vh[0]), shape=(int(vh[1]), int(vh[0])))
    max_k = 16
    raw_features = vt[row, :max_k].astype(np.float64)
    norm = np.maximum(raw_features[train].std(axis=0), 1.)
    features = np.column_stack((np.ones(len(y)), raw_features / norm))
    del raw_features
    target = y.astype(float) - baseline
    weights = (2000 / np.maximum(y[train], 200)) / np.maximum(np.abs(target[train]), 4.)
    weights /= weights.mean()
    n_pairs = 208 * 288
    xt = features[train]
    p = pair[train]
    gram = np.zeros((n_pairs, max_k+1, max_k+1), np.float64)
    rhs = np.empty((n_pairs, max_k+1), np.float64)
    print('fitting', len(train), 'rows; excluded', original_train_rows-len(train), flush=True)
    for i in range(max_k+1):
        rhs[:, i] = np.bincount(p, weights=weights * xt[:, i] * target[train], minlength=n_pairs)
        for j in range(i+1):
            entry = np.bincount(p, weights=weights * xt[:, i] * xt[:, j], minlength=n_pairs)
            gram[:, i, j] = gram[:, j, i] = entry
    best = None
    records = []
    for k in (0, 4, 16):
        for penalty in (4., 16., 64.):
            system = gram[:, :k+1, :k+1].copy()
            system[:, np.arange(k+1), np.arange(k+1)] += penalty
            coef = np.linalg.solve(system, rhs[:, :k+1])
            # M stays int16. Account for this in the validation approximation.
            coef[:, 0] = np.rint(coef[:, 0])
            delta = np.einsum('ij,ij->i', features[valid, :k+1], coef[pair[valid]])
            pred = np.maximum(0, baseline[valid] + np.rint(delta))
            accuracy = float(points(y[valid], pred).mean())
            record = {'components': k, 'penalty': penalty, 'approx_valid_accuracy': accuracy,
                      'mean_abs_correction_ps': float(np.abs(delta).mean())}
            records.append(record)
            print(record, flush=True)
            if best is None or accuracy > best[0]:
                best = (accuracy, k, penalty, coef)
    accuracy, k, penalty, coef = best
    with (OUT / 'oracle/oracle_U.bin').open('rb') as stream:
        uh = np.fromfile(stream, '<i4', count=6)
        u = np.fromfile(stream, '<f4').reshape(n_pairs, -1)
    with (OUT / 'oracle/oracle_M.bin').open('rb') as stream:
        mh = np.fromfile(stream, '<i4', count=3)
        m = np.fromfile(stream, '<i2').astype(np.int32)
    if k:
        u[:, :k] += (coef[:, 1:] / norm[:k] / scales[:k]).astype(np.float32)
    m += coef[:, 0].astype(np.int32)
    assert m.min() >= -32768 and m.max() <= 32767
    u_path, m_path = OUT / 'U_refit.bin', OUT / 'M_refit.bin'
    with u_path.open('wb') as stream:
        uh.tofile(stream)
        u.astype('<f4').tofile(stream)
    with m_path.open('wb') as stream:
        mh.tofile(stream)
        m.astype('<i2').tofile(stream)
    np.savez_compressed(OUT / 'refit_coefficients.npz', coefficients=coef, norm=norm[:k])
    report = {'train_rows': len(train), 'excluded_train_rows': original_train_rows-len(train),
              'excluded_source_coordinates': spec['reserved_source_coordinates'],
              'objective_proxy': 'relative-error weighted robust least squares around baseline',
              'screening': 'Fixed baseline branch; integer correction approximation, actual C++ required',
              'validation_configs': records,
              'selected': {'components': k, 'penalty': penalty, 'approx_valid_accuracy': accuracy},
              'baseline_valid_accuracy': float(points(y[valid], baseline[valid]).mean()),
              'U_sha256': sha(u_path), 'M_sha256': sha(m_path)}
    (REPO_ROOT / 'docs/displacement_refit.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
