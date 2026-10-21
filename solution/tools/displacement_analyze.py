"""Score deployment output and attribute its error without fitting to holdouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from displacement_assets import sha
from experiment import pin_parts
from project_paths import BUILD, DATA, REPO_ROOT

OUT = BUILD / 'displacement'


def split(n):
    order = np.random.default_rng(7).permutation(n)
    valid, test = order[300000:350000], order[350000:400000]
    train = np.r_[order[:300000], order[400000:]]
    return train, valid, test


def points(y, pred):
    error = np.abs(pred - y)
    return np.where(y > 0, 100 * (1 - np.tanh(4 * error / np.maximum(y, 1))),
                    100 * (pred == 0))


def load_dataset():
    return pd.read_csv(DATA / 'data/delay_estimate_ans.csv')


def describe(y, pred):
    error = np.abs(pred - y)
    return {'rows': len(y), 'accuracy': float(points(y, pred).mean()),
            'mae_ps': float(error.mean()), 'rmse_ps': float(np.sqrt(np.mean(error.astype(float)**2))),
            'abs_error_p50_p90_p99': np.percentile(error, [50, 90, 99]).tolist(),
            'signed_error_ps': float(np.mean(pred - y))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    frame = load_dataset()
    result_path = OUT / (args.name + '_result.csv')
    result = pd.read_csv(result_path)
    if list(result.columns) != ['From', 'To', 'delay'] or len(result) != len(frame):
        raise ValueError('Result schema or row count mismatch')
    if not result.From.equals(frame.From) or not result.To.equals(frame.To):
        raise ValueError('Output names/order mismatch')
    y = frame.delay.to_numpy(np.int32)
    pred = result.delay.to_numpy()
    if not np.issubdtype(pred.dtype, np.integer) or (pred < 0).any():
        raise ValueError('Output must contain nonnegative integers')
    pred = pred.astype(np.int32)
    sx, sy, source = pin_parts(frame.From)
    tx, ty, target = pin_parts(frame.To)
    dx, dy = tx - sx, ty - sy
    distance = np.abs(dx) + np.abs(dy)
    cross = ((sx <= 75) & (tx >= 90)) | ((tx <= 75) & (sx >= 90))
    train, valid, test = split(len(y))
    report = {'candidate': args.name, 'output_sha256': sha(result_path),
              'full_public': describe(y, pred), 'valid': describe(y[valid], pred[valid]),
              'test': describe(y[test], pred[test]),
              'caveat': 'Historical rng7 regression slices; repeatedly inspected, not a new blind test',
              'groups': {}}
    families = lambda series: series.str.replace(r'\[\d+\]|\d+$', '', regex=True).to_numpy()
    groups = {
        'distance': pd.cut(distance, [-1, 9, 29, 99, 299, 1000]).astype(str),
        'source_family': families(source), 'target_family': families(target),
        'wall_crossing': cross.astype(str),
        'source_x': pd.cut(sx, [-1, 19, 59, 75, 89, 109, 120]).astype(str),
        'source_y_band': sy // 50,
        'boundary_endpoint': ((np.minimum(sx, tx) < 10) | (np.maximum(sx, tx) >= 110)
                              | (np.minimum(sy, ty) < 12) | (np.maximum(sy, ty) >= 538)).astype(str),
    }
    for group_name, values in groups.items():
        result_groups = {}
        for value in np.unique(values):
            ix = valid[values[valid] == value]
            if not len(ix):
                continue
            item = describe(y[ix], pred[ix])
            item['whole_valid_score_loss'] = (100 - item['accuracy']) * len(ix) / len(valid)
            result_groups[str(value)] = item
        report['groups'][group_name] = result_groups
    worst = valid[np.argsort(points(y[valid], pred[valid]))[:20]]
    report['worst_valid_examples'] = [
        {'From': frame.From.iloc[i], 'To': frame.To.iloc[i], 'truth': int(y[i]),
         'prediction': int(pred[i])} for i in worst]
    np.save(OUT / (args.name + '_prediction.npy'), pred)
    # Small, shared geometry cache; no training feature pickle or 32 GB dense matrices.
    if not (OUT / 'geometry.npz').exists():
        ids = np.loadtxt(OUT / 'oracle/oracle_ids.txt', dtype=np.int32)
        ports = json.loads((DATA / 'arch/SRB_Port.json').read_text(encoding='utf-8'))['Port']
        names = [p['Name'] for p in ports]
        pids = {name: i for i, name in enumerate(names)}
        source_id = source.map(pids).to_numpy(np.int32)
        target_id = target.map(pids).to_numpy(np.int32)
        sm = {int(p): i for i, p in enumerate(ids[:208])}
        tm = {int(p): i for i, p in enumerate(ids[208:])}
        si = np.array([sm.get(int(p), -1) for p in source_id], np.int32)
        ti = np.array([tm.get(int(p), -1) for p in target_id], np.int32)
        if (si < 0).any() or (ti < 0).any():
            raise ValueError('Public endpoint outside model maps')
        np.savez_compressed(OUT / 'geometry.npz', sx=sx, sy=sy, tx=tx, ty=ty,
                            si=si, ti=ti, source_id=source_id, target_id=target_id, y=y)
    report['input_sha256'] = sha(DATA / 'data/delay_estimate_ans.csv')
    target = REPO_ROOT / 'docs' / (args.name + '_analysis.json')
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('candidate', 'full_public', 'valid', 'test')}, indent=2))


if __name__ == '__main__':
    main()
