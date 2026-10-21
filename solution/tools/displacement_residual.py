"""Train stage-9 residuals with a source-coordinate holdout and validation selection."""
from __future__ import annotations

import argparse
import json
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from displacement_analyze import OUT, split, points, describe
from displacement_assets import sha
from project_paths import BUILD, DATA, REPO_ROOT

FEATURES = ['sx', 'sy', 'tx', 'ty', 'dx', 'dy', 'ax', 'ay', 'distance',
            'source', 'target', 'source_family', 'target_family', 'source_bit', 'target_bit',
            'ndx', 'ndy', 'rx', 'ry', 'rax', 'ray', 'overx', 'overy',
            'dx_mod', 'dy_mod', 'rx_mod', 'ry_mod', 'sx_mod', 'sy_mod', 'tx_mod', 'ty_mod',
            'source_band', 'target_band', 'source_phase', 'target_phase',
            'cross', 'landing_cross', 'source_wall', 'landing_wall', 'target_wall',
            'source_door_distance', 'landing_door_distance', 'target_door_distance',
            'edge_x', 'edge_y', 'gap_x', 'gap_y', 'net_gap_x', 'net_gap_y',
            'base', 'base_minus_gap', 'net_length', 'direction']
CATEGORIES = ['source', 'target', 'source_family', 'target_family', 'direction']


def architecture():
    ports = json.loads((DATA / 'arch/SRB_Port.json').read_text(encoding='utf-8'))['Port']
    names = [p['Name'] for p in ports]
    families = [re.sub(r'\[\d+\]|\d+$', '', name) for name in names]
    unique = sorted(set(families))
    family = np.array([unique.index(f) for f in families], np.int32)
    bit = np.array([int(re.findall(r'\d+', n)[-1]) if re.findall(r'\d+', n) else -1 for n in names], np.int32)
    nets = json.loads((DATA / 'arch/SRB_Net.json').read_text(encoding='utf-8'))['Nets']
    index = {name: i for i, name in enumerate(names)}
    ndx, ndy = np.zeros(len(names), np.int32), np.zeros(len(names), np.int32)
    for net in nets:
        ndx[index[net['from']]] = net['delta x']
        ndy[index[net['from']]] = net['delta y']
    gaps = json.loads((DATA / 'arch/SRB_Gap.json').read_text(encoding='utf-8'))['Gap']
    vp, hp = np.zeros(122, np.int32), np.zeros(552, np.int32)
    for line in gaps['Line']:
        (vp if line['direction'] == 'vertical' else hp)[line['site']+1] += line['delay']
    vp, hp = vp.cumsum(), hp.cumsum()
    # Match the baseline architecture loader's wall/door convention.
    doors = np.ones(550, bool)
    for block in gaps['Block']:
        if block['left'] <= 80 <= block['right']:
            doors[block['lower']:block['upper']+1] = False
    door_rows = np.flatnonzero(doors)
    door_distance = np.abs(np.arange(550)[:, None]-door_rows).min(axis=1)
    return names, family, bit, ndx, ndy, vp, hp, door_distance


def make_features(geometry, base):
    _, family, bit, netdx, netdy, vp, hp, door_distance = architecture()
    sx, sy, tx, ty = (geometry[k].astype(np.int32) for k in ('sx', 'sy', 'tx', 'ty'))
    source, target = (geometry[k].astype(np.int32) for k in ('source_id', 'target_id'))
    dx, dy = tx-sx, ty-sy
    ax, ay = np.abs(dx), np.abs(dy)
    ndx, ndy = netdx[source], netdy[source]
    lx, ly = np.clip(sx+ndx, 0, 119), np.clip(sy+ndy, 0, 549)
    rx, ry = dx-ndx, dy-ndy
    cross = lambda a, b: ((a <= 75) & (b >= 90)) | ((b <= 75) & (a >= 90))
    wall = lambda x, y: (x >= 76) & (x <= 89) & (door_distance[y] > 0)
    gap_x = vp[np.maximum(sx, tx)]-vp[np.minimum(sx, tx)]
    gap_y = hp[np.maximum(sy, ty)]-hp[np.minimum(sy, ty)]
    net_gap_x = vp[np.maximum(sx, lx)]-vp[np.minimum(sx, lx)] + vp[np.maximum(lx, tx)]-vp[np.minimum(lx, tx)]
    net_gap_y = hp[np.maximum(sy, ly)]-hp[np.minimum(sy, ly)] + hp[np.maximum(ly, ty)]-hp[np.minimum(ly, ty)]
    values = [sx, sy, tx, ty, dx, dy, ax, ay, ax+ay,
              source, target, family[source], family[target], bit[source], bit[target],
              ndx, ndy, rx, ry, np.abs(rx), np.abs(ry),
              np.maximum(np.abs(rx)-ax, 0), np.maximum(np.abs(ry)-ay, 0),
              dx%10, dy%12, rx%10, ry%12, sx%10, sy%12, tx%10, ty%12,
              sy//100, ty//100, sy%100, ty%100,
              cross(sx, tx), cross(lx, tx), wall(sx, sy), wall(lx, ly), wall(tx, ty),
              door_distance[sy], door_distance[ly], door_distance[ty],
              np.minimum(np.minimum(sx, tx), 119-np.maximum(sx, tx)),
              np.minimum(np.minimum(sy, ty), 549-np.maximum(sy, ty)),
              gap_x, gap_y, net_gap_x, net_gap_y,
              base, base-net_gap_x-net_gap_y, np.abs(ndx)+np.abs(ndy), 2*(dx<0)+(dy<0)]
    assert len(values) == len(FEATURES)
    return pd.DataFrame({k: v.astype(np.float32) for k, v in zip(FEATURES, values)})


def indices(geometry):
    train, valid, test = split(len(geometry['y']))
    sx, sy = geometry['sx'].astype(np.int64), geometry['sy'].astype(np.int64)
    held = ((sx*1009 + sy*9176) % 5) == 0
    reserved = np.zeros(len(sx), bool)
    for x, y in json.loads((OUT / 'oracle_spec.json').read_text())['reserved_source_coordinates']:
        reserved |= (sx == x) & (sy == y)
    return train[~held[train] & ~reserved[train]], valid, test, train[held[train] & ~reserved[train]]


def add_latent_features(x, geometry, count):
    if not count:
        return x
    with (OUT / 'U_refit_q16.bin').open('rb') as stream:
        h = np.fromfile(stream, '<i4', count=6)
        scale = np.fromfile(stream, '<f4', count=int(h[2]))
        u = np.fromfile(stream, '<i2').reshape(-1, int(h[2])).astype(np.float32)*scale
    with (OUT / 'oracle/oracle_vt.bin').open('rb') as stream:
        vh = np.fromfile(stream, '<i4', count=6)
        vscale = np.fromfile(stream, '<f4', count=int(vh[0]))
    u = u*vscale
    v = np.memmap(OUT / 'oracle/oracle_vt.bin', '<i2', 'r',
                  offset=24+4*int(vh[0]), shape=(int(vh[1]), int(vh[0])))
    trace = np.fromfile(OUT / 'displacement_caltrace_trace.bin', '<i4').reshape(-1, 4)
    assert len(trace) == len(x) and (trace[:, :2] >= 0).all()
    pair, row = trace[:, 0], trace[:, 1]
    for k in range(count):
        x['u'+str(k)] = u[pair, k]
        x['v'+str(k)] = v[row, k].astype(np.float32)
        x['uv'+str(k)] = x['u'+str(k)].to_numpy()*x['v'+str(k)].to_numpy()
    return x


def architecture_bound(geometry, x):
    meta = json.loads((BUILD / 'architecture_meta.json').read_text())
    table = np.load(BUILD / 'architecture_table.npy', mmap_mode='r')
    sm = {int(p): i for i, p in enumerate(meta['sources'])}
    tm = {int(p): i for i, p in enumerate(meta['targets'])}
    sp = np.array([sm[int(p)] for p in geometry['source_id']])
    tp = np.array([tm[int(p)] for p in geometry['target_id']])
    dx = x['dx'].to_numpy().astype(np.int32)
    dy = x['dy'].to_numpy().astype(np.int32)
    residue = dx%10*12+dy%12
    bounds = []
    for d in range(4):
        cost = table[d, sp, residue, tp].astype(np.int32)
        linear = meta['slopes_scaled12'][d][0]*dx+meta['slopes_scaled12'][d][1]*dy
        bounds.append((cost+linear)/12)
    return np.maximum.reduce(bounds)+x['net_gap_x'].to_numpy()+x['net_gap_y'].to_numpy()


def corrected(base, delta):
    return np.maximum(0, np.floor(base.astype(np.float64)+delta+.5)).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='residual_screen')
    parser.add_argument('--rounds', type=int, default=160)
    parser.add_argument('--leaves', type=int, default=127)
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--objective', choices=['regression_l1', 'regression'], default='regression_l1')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--latent', type=int, default=0)
    parser.add_argument('--bound', action='store_true')
    args = parser.parse_args()
    geometry = np.load(OUT / 'geometry.npz')
    y = geometry['y']
    base = np.load(OUT / 'displacement_cal16_prediction.npy')
    x = make_features(geometry, base)
    x = add_latent_features(x, geometry, args.latent)
    if args.bound:
        x['bound'] = architecture_bound(geometry, x).astype(np.float32)
        x['base_minus_bound'] = base-x['bound'].to_numpy()
    train, valid, _, assess = indices(geometry)
    residual = y.astype(np.float64)-base
    weight = 2000 / np.maximum(y[train], 200)
    if args.objective == 'regression':
        weight /= np.maximum(np.abs(residual[train]), 4)
    weight /= weight.mean()
    params = dict(objective=args.objective, learning_rate=args.learning_rate,
                  num_leaves=args.leaves, min_data_in_leaf=150, lambda_l2=10,
                  max_bin=255, num_threads=args.threads, verbosity=-1,
                  seed=20261021, deterministic=True, force_col_wise=True,
                  cat_smooth=30, cat_l2=20, max_cat_threshold=32)
    data = lgb.Dataset(x.iloc[train], residual[train], weight=weight,
                       categorical_feature=CATEGORIES, free_raw_data=False)
    start = time.perf_counter()
    model = lgb.train(params, data, num_boost_round=args.rounds)
    model.save_model(str(OUT / (args.name + '.txt')))
    checkpoints = sorted(set([n for n in (4, 8, 16, 32, 64, 96, 128, 160, args.rounds) if n <= args.rounds]))
    records = []
    for n in checkpoints:
        pred = corrected(base[valid], model.predict(x.iloc[valid], num_iteration=n, num_threads=args.threads))
        item = {'trees': n, 'valid_accuracy': float(points(y[valid], pred).mean()),
                'valid_mae': float(np.abs(y[valid]-pred).mean())}
        records.append(item)
        print(json.dumps(item), flush=True)
    best = max(records, key=lambda item: item['valid_accuracy'])
    pred = corrected(base[assess], model.predict(x.iloc[assess], num_iteration=best['trees'], num_threads=args.threads))
    report = {'name': args.name, 'params': params, 'feature_names': list(x.columns), 'latent_components': args.latent,
              'categorical_features': CATEGORIES, 'architecture_bound': args.bound,
              'fit_rows': len(train), 'position_assess_rows': len(assess), 'valid_rows': len(valid),
              'baseline_valid': describe(y[valid], base[valid]), 'curve': records,
              'selected_by_validation': best, 'position_assess_baseline': describe(y[assess], base[assess]),
              'position_assess_updated': describe(y[assess], pred), 'model_sha256': sha(OUT / (args.name+'.txt')),
              'elapsed_seconds': time.perf_counter()-start, 'test_labels_used': False,
              'caveat': 'Only the new residual excludes held source coordinates; inherited stage8 used main training'}
    (REPO_ROOT / 'docs' / (args.name + '.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('selected_by_validation', 'position_assess_baseline', 'position_assess_updated')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
