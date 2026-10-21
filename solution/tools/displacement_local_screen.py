"""Validation-only screening of generated local factors on the old selected branch.

This proxy keeps the old branch choice. Deployment must separately validate the
real two-branch minimum and its output against an independent reference.
"""
from __future__ import annotations
import os
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '4'
import argparse
import json
import numpy as np
from displacement_analyze import OUT, split, points
from displacement_assets import sha
from project_paths import REPO_ROOT


def quantize(coeff, bits):
    limit = (1 << (bits-1))-1
    scale = np.maximum(np.max(np.abs(coeff), axis=0)/limit, np.finfo(np.float32).tiny)
    q = np.rint(coeff/scale).clip(-limit, limit).astype('<i2' if bits == 16 else 'i1')
    return q, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--factors', default='stage10_local_weighted')
    args = ap.parse_args()
    path = OUT/(args.factors+'_factors.npz')
    f = np.load(path)
    g = np.load(OUT/'geometry.npz')
    base = np.load(OUT/'displacement_cal16_prediction.npy')
    trace = np.fromfile(OUT/'displacement_caltrace_trace.bin', '<i4').reshape(-1, 4)
    mapping = np.full(266305, -1, np.int32)
    mapping[f['locations']] = np.arange(len(f['locations']))
    local = mapping[trace[:, 1]]
    _, valid, _ = split(len(base))
    vi = valid[local[valid] >= 0]
    v = f['v'].T.copy()
    cal = np.fromfile(OUT/'calibration_train.bin', '<f4')
    distance = np.abs(g['tx']-g['sx'])+np.abs(g['ty']-g['sy'])
    records = []
    for bits in (8, 16):
        for rank in (32, 64, 96, 128, 160):
            uq, us = quantize(f['coefficients'][:, :rank], bits)
            u = (uq.astype(np.float32)*us)*f['scales'][:rank]
            delta = np.empty(len(vi), np.float32)
            for start in range(0, len(vi), 4096):
                ix = vi[start:start+4096]
                delta[start:start+4096] = np.sum(u[trace[ix, 0]]*v[local[ix], :rank], axis=1)
            rounded = np.where(delta >= 0, np.floor(delta+.5), np.ceil(delta-.5)).astype(np.int32)
            for gate in (30, 50, 75, 100, 150, 1000):
                use = distance[vi] <= gate
                raw = trace[vi[use], 3]+rounded[use]
                bins = np.searchsorted([700,1000,2000,4000], raw, side='right')
                updated = np.maximum(0, np.floor(raw.astype(np.float64)*cal[g['ti'][vi[use]]*5+bins]+.5)).astype(np.int32)
                pred = base[valid].copy()
                slots = np.flatnonzero(local[valid] >= 0)[use]
                pred[slots] = updated
                item = dict(bits=bits, rank=rank, distance_gate=gate,
                            valid_accuracy=float(points(g['y'][valid], pred).mean()),
                            valid_coverage=len(slots)/len(valid))
                records.append(item)
                print(json.dumps(item), flush=True)
    report = dict(factors_sha256=sha(path), baseline_valid=float(points(g['y'][valid], base[valid]).mean()),
                  selected_branch_proxy=True, test_labels_used=False, curve=records,
                  best=max(records, key=lambda r:r['valid_accuracy']))
    (REPO_ROOT/'docs'/(args.factors+'_screen.json')).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
