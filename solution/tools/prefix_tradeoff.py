"""Evaluate primary-model prefixes on validation data before CPU screening."""
import json
import lightgbm as lgb
import numpy as np
from experiment import BUILD, competition_score
from topology_experiment import load_features, topology_features, split

x, y = load_features()
x, _ = topology_features(x)
valid, _, _ = split(len(y))
x = x.iloc[valid]
y = y[valid]
base = x.architecture_delay.to_numpy()
m = lgb.Booster(model_file=str(BUILD / 's3_topology_lr32_511l_25r.txt'))
distance = x.manhattan.to_numpy()
full = base + m.predict(x, num_iteration=25, num_threads=4)
rows = []
for n in (1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 25):
    p = base + m.predict(x, num_iteration=n, num_threads=4)
    row = {'trees': n, 'valid': competition_score(y, p),
           'valid_short_refine': competition_score(y, np.where(distance < 100, full, p))}
    rows.append(row)
    print(json.dumps(row), flush=True)
(BUILD / 'prefix_tradeoff_valid.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
