"""Fit small additive lookup residuals; select only on the historical validation slice."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from displacement_analyze import OUT, points, describe
from displacement_residual import make_features, indices, architecture_bound
from project_paths import REPO_ROOT


def compact_features(g, base):
    x = make_features(g, base)
    bound_path = OUT / 'stage10_bound.npy'
    bound = np.load(bound_path) if bound_path.exists() else architecture_bound(g, x).astype(np.int32)
    x['bound'] = bound
    x['margin'] = base-bound
    # Definitions are deliberately integer-valued for exact C++ export.
    bins = {
        'margin': [-50,-20,-10,-5,0,5,10,15,20,30,40,60,80,100,150,200,250,300,400],
        'distance': [10,20,30,50,75,100,150,200,300,400],
        'base': [500,700,1000,1500,2000,3000,4000,6000],
        'ax': [5,10,15,20,30,40,60,80],
        'ay': [5,10,20,30,50,75,100,150,200,300,400],
        'rx': [-60,-30,-15,-5,0,5,15,30,60],
        'ry': [-300,-150,-75,-30,-10,0,10,30,75,150,300],
        'source_door_distance': [0,1,3,6,10,15,20,25],
        'landing_door_distance': [0,1,3,6,10,15,20,25],
        'target_door_distance': [0,1,3,6,10,15,20,25],
        'edge_x': [0,1,3,5,10,20,30],
        'edge_y': [0,1,3,6,12,24,48,96],
        'sx': [20,40,60,75,89,100],
        'tx': [20,40,60,75,89,100],
    }
    values = {k: x[k].to_numpy().astype(np.int32) for k in x}
    for key, edges in bins.items():
        values[key+'_bin'] = np.searchsorted(edges, values[key], side='left').astype(np.int32)
    return values, bound, bins


def fit_medians(keys, residual, weights, size, shrink):
    order = np.lexsort((residual, keys))
    k, r, w = keys[order], residual[order], weights[order]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1])+1]
    ends = np.r_[starts[1:], len(k)]
    cs = np.r_[0., np.cumsum(w)]
    mid = (cs[starts]+cs[ends])/2
    at = np.minimum(np.searchsorted(cs, mid, side='right')-1, ends-1)
    count = ends-starts
    result = np.zeros(size, np.float64)
    result[k[starts]] = r[at]*count/(count+shrink)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', default='stage10_compact')
    ap.add_argument('--cycles', type=int, default=12)
    ap.add_argument('--shrink', type=float, default=50)
    ap.add_argument('--rate', type=float, default=.8)
    ap.add_argument('--teacher', action='store_true')
    ap.add_argument('--terms', choices=['small','medium','large'], default='small')
    args = ap.parse_args()
    start = time.perf_counter()
    g = np.load(OUT/'geometry.npz')
    y = g['y']; base = np.load(OUT/'displacement_cal16_prediction.npy')
    v, bound, bins = compact_features(g, base)
    train, valid, _, assess = indices(g)
    target = np.load(OUT/'stage9_accurate64_prediction.npy') if args.teacher else y
    weight = 2000/np.maximum(y[train],200)
    definitions = [
        ['source','margin_bin'], ['target','margin_bin'], ['margin_bin','distance_bin'],
        ['landing_wall','landing_door_distance_bin','rx_bin','ry_bin'],
        ['source','base_bin'], ['target','base_bin'],
        ['landing_cross','margin_bin','ax_bin'],
        ['source','direction'], ['target','direction'],
    ]
    if args.terms in ('medium','large'):
        definitions += [
            ['source','target_family'], ['source_family','target'],
            ['source','rx_mod','ry_mod'], ['target','rx_mod','ry_mod'],
            ['cross','source_phase','target_phase'],
            ['landing_wall','source','landing_door_distance_bin'],
            ['edge_x_bin','margin_bin'], ['edge_y_bin','margin_bin'],
            ['landing_cross','source_door_distance_bin','target_door_distance_bin'],
        ]
    if args.terms == 'large':
        definitions += [
            ['source','target','direction'],
            ['source','distance_bin','direction'], ['target','distance_bin','direction'],
            ['sx_bin','tx_bin','margin_bin'],
            ['source','landing_cross','landing_door_distance_bin'],
            ['target','landing_cross','target_door_distance_bin'],
        ]
    keys, sizes, shapes = [], [], []
    for term in definitions:
        shape = [int(v[k].max())+1 for k in term]
        key = np.zeros(len(y), np.int64)
        for k,n in zip(term,shape): key=key*n+v[k]
        keys.append(key); shapes.append(shape); sizes.append(int(np.prod(shape)))
    tables = [np.zeros(n) for n in sizes]
    delta = np.zeros(len(y)); residual = target[train].astype(float)-base[train]
    curve = []; best = -1
    for cycle in range(args.cycles):
        for i,key in enumerate(keys):
            old = tables[i][key]
            fitted = fit_medians(key[train],residual-delta[train]+old[train],weight,sizes[i],args.shrink)
            update = (fitted-tables[i])*args.rate
            tables[i] += update; delta += update[key]
        pred = np.maximum(bound,np.maximum(0,np.floor(base+delta+.5))).astype(np.int32)
        accuracy = float(points(y[valid],pred[valid]).mean())
        row = {'cycle':cycle+1,'valid_accuracy':accuracy}
        print(json.dumps(row),flush=True); curve.append(row)
        if accuracy>best:
            best=accuracy
            np.savez_compressed(OUT/(args.name+'.npz'),**{'table'+str(i):z for i,z in enumerate(tables)})
            np.save(OUT/(args.name+'_reference.npy'),pred)
            best_cycle=cycle+1
    pred=np.load(OUT/(args.name+'_reference.npy'))
    report = dict(name=args.name,args=vars(args),definitions=definitions,shapes=shapes,bins=bins,
                  curve=curve,selected_cycle=best_cycle,valid=describe(y[valid],pred[valid]),
                  position_assess=describe(y[assess],pred[assess]),fit_rows=len(train),
                  cells=sum(sizes),elapsed_seconds=time.perf_counter()-start,test_labels_used=False,
                  caveat='Only new residual excludes held coordinates; historical validation is not blind')
    (REPO_ROOT/'docs'/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('name','valid','position_assess','cells','elapsed_seconds')}),flush=True)


if __name__ == '__main__': main()
