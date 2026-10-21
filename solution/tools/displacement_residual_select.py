"""Select a deployable prefix using validation; do not read test labels."""
import argparse
import json
import lightgbm as lgb
import numpy as np
from displacement_residual import make_features, architecture_bound, indices, corrected
from displacement_analyze import OUT, points, describe
from project_paths import REPO_ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('names', nargs='+')
args = parser.parse_args()
g = np.load(OUT / 'geometry.npz')
base = np.load(OUT / 'displacement_cal16_prediction.npy')
x = make_features(g, base)
bound = architecture_bound(g, x)
_, valid, _, assess = indices(g)
records = []
for name in args.names:
    report = json.loads((REPO_ROOT / 'docs' / (name+'.json')).read_text())
    model = lgb.Booster(model_file=str(OUT / (name+'.txt')))
    f = x.copy()
    if report.get('architecture_bound'):
        f['bound'] = bound.astype(np.float32)
        f['base_minus_bound'] = base-f['bound'].to_numpy()
    for count in sorted(set([n for n in (4, 8, 12, 16, 24, 32, 64, 128) if n <= model.num_trees()])):
        pred = corrected(base[valid], model.predict(f.iloc[valid], num_iteration=count, num_threads=6))
        for clamp in (False, True):
            p = np.maximum(pred, bound[valid]).astype(np.int32) if clamp else pred
            item = dict(model=name, trees=count, bound_clamp=clamp,
                        valid_accuracy=float(points(g['y'][valid], p).mean()))
            records.append(item)
            print(item, flush=True)
qualified = [r for r in records if r['valid_accuracy'] >= 98.05]
if qualified:
    selected = min(qualified, key=lambda r: (r['trees'], -r['valid_accuracy']))
else:
    selected = max(records, key=lambda r: r['valid_accuracy'])
fast_qualified = [r for r in records if r['valid_accuracy'] >= 98.02]
fast = min(fast_qualified, key=lambda r: (r['trees'], -r['valid_accuracy'])) if fast_qualified else None
result = dict(configs=records, selected=selected, fast_selected=fast,
              criterion='fewest trees with validation>=98.05; otherwise highest validation',
              fast_criterion='fewest trees with validation>=98.02; otherwise no fast candidate',
              test_labels_used=False)
(REPO_ROOT / 'docs/residual_selection.json').write_text(json.dumps(result, indent=2)+'\n')
print('SELECTED', selected, flush=True)
