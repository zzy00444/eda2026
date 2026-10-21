"""Probe whether the periodic architecture envelope recovers low-rank underestimates."""
import json
import numpy as np
from displacement_analyze import OUT, split, points
from project_paths import BUILD, REPO_ROOT

g = np.load(OUT / 'geometry.npz')
base = np.load(OUT / 'displacement_cal16_prediction.npy')
meta = json.loads((BUILD / 'architecture_meta.json').read_text())
table = np.load(BUILD / 'architecture_table.npy', mmap_mode='r')
sm = {int(p): i for i, p in enumerate(meta['sources'])}
tm = {int(p): i for i, p in enumerate(meta['targets'])}
sp = np.array([sm[int(p)] for p in g['source_id']])
tp = np.array([tm[int(p)] for p in g['target_id']])
dx = g['tx'].astype(np.int32)-g['sx']
dy = g['ty'].astype(np.int32)-g['sy']
direction = 2*(dx<0)+(dy<0)
from displacement_residual import make_features
x = make_features(g, base)
valid = split(len(base))[1]
results = []
for kind, dxx in [('physical', dx), ('virtual', dx-np.where(x['cross'].to_numpy()>0, np.sign(dx)*14, 0))]:
    residue = dxx%10*12+dy%12
    values = []
    for d in range(4):
        linear = meta['slopes_scaled12'][d][0]*dxx+meta['slopes_scaled12'][d][1]*dy
        values.append((table[d, sp, residue, tp].astype(np.float64)+linear)/12)
    bounds = np.maximum.reduce(values)+x['net_gap_x'].to_numpy()+x['net_gap_y'].to_numpy()
    if kind == 'virtual':
        bounds += 292*(x['cross'].to_numpy()>0)
    for gate, use in [('all', np.ones(len(base), bool)), ('no_wall', (x['cross'].to_numpy()==0)&(x['landing_wall'].to_numpy()==0)),
                      ('long_no_wall', (x['cross'].to_numpy()==0)&(x['landing_wall'].to_numpy()==0)&(np.abs(dx)+np.abs(dy)>=100))]:
        for margin in (0, 4, 8, 16):
            pred = np.maximum(base, np.where(use, np.ceil(bounds-margin), base)).astype(np.int32)
            item = dict(kind=kind, gate=gate, margin=margin,
                        valid_accuracy=float(points(g['y'][valid], pred[valid]).mean()),
                        rows_changed=int(np.sum(pred[valid]!=base[valid])),
                        bound_above_truth=int(np.sum(use[valid] & (bounds[valid]-margin > g['y'][valid]))))
            results.append(item)
            print(item, flush=True)
report = {'baseline_valid': float(points(g['y'][valid], base[valid]).mean()), 'configs': results, 'test_labels_used': False}
(REPO_ROOT / 'docs/residual_bound_probe.json').write_text(json.dumps(report, indent=2)+'\n')
