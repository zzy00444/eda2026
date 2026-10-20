"""Train a correction that can be folded into the architecture table.

A correction is foldable when it is a function of (direction, source port,
residue, target port) only.  Excluded for that reason:

* sx/sy/tx/ty and every block or band feature: absolute position;
* dx/dy/ax/ay/manhattan and distance buckets: the continuous displacement;
* gap_x/gap_y and architecture_delay: they contain the crossed gap lines, which
  depend on the absolute position;
* remaining_ax_mod / remaining_ay_mod: |dx - net_delta| needs the sign of the
  difference, not just dx mod 10;
* architecture_envelope: the max over directions needs the slope term, which
  depends on dx/dy themselves.  A foldable proxy, the max of the four table
  entries at the same key, is supplied instead.

The model keeps the raw table value (table_scaled) and that proxy, both of which
are key functions.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# scripts in this directory import each other and project_paths directly
from project_paths import BUILD  # noqa: E402
from experiment import competition_score  # noqa: E402
from topology_experiment import load_features, topology_features  # noqa: E402
from third_stage_experiment import split  # noqa: E402

FOLDABLE = [
    "source_port", "source_family", "source_bit",
    "target_port", "target_family", "target_bit",
    "first_dx", "first_dy",
    "dx_mod", "dy_mod", "ax_mod", "ay_mod",
    "remaining_x_mod", "remaining_y_mod",
    "port_pair", "family_pair", "bit_pair", "direction",
    "table_scaled", "envelope_proxy",
]
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
LEAVES = int(sys.argv[3]) if len(sys.argv) > 3 else 511
NAME = sys.argv[4] if len(sys.argv) > 4 else "foldable_lr08_300r"

x, y = load_features()
x, _ = topology_features(x)
base = x.architecture_delay.to_numpy().copy()
meta = json.loads((BUILD / "architecture_meta.json").read_text())
slopes = np.array(meta["slopes_scaled12"])
direction = (2 * (x.dx.to_numpy() < 0) + (x.dy.to_numpy() < 0)).astype(np.int64)
linear12 = slopes[direction, 0] * x.dx.to_numpy() + slopes[direction, 1] * x.dy.to_numpy()
gaps = x.gap_x.to_numpy() + x.gap_y.to_numpy()
table = np.load(BUILD / "architecture_table.npy", mmap_mode="r")
# source_port/target_port hold SB_Port indices, not names.
src_map = {p: i for i, p in enumerate(meta["sources"])}
tgt_map = {p: i for i, p in enumerate(meta["targets"])}
sidx = np.array([src_map[p] for p in x.source_port.astype(np.int32).to_numpy()], dtype=np.int64)
tidx = np.array([tgt_map[p] for p in x.target_port.astype(np.int32).to_numpy()], dtype=np.int64)
r = (x.dx.to_numpy() % 10) * 12 + (x.dy.to_numpy() % 12)
bounds = np.stack([table[d, sidx, r, tidx] for d in range(4)])
bounds = np.where(bounds == 32767, -32768, bounds)
x = x.copy()
x["table_scaled"] = np.rint((base - gaps) * 12 - linear12)
x["envelope_proxy"] = bounds.max(axis=0)
x = x[FOLDABLE]

va, te, tr = split(len(y))
cat = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
data = lgb.Dataset(x.iloc[tr], label=y[tr] - base[tr],
                   weight=1 / np.maximum(y[tr], 200), categorical_feature=cat)
valid = lgb.Dataset(x.iloc[va], label=y[va] - base[va],
                    weight=1 / np.maximum(y[va], 200), categorical_feature=cat,
                    reference=data)
started = time.perf_counter()
model = lgb.train({"objective": "regression_l1", "metric": "l1",
                   "learning_rate": RATE, "num_leaves": LEAVES, "min_data_in_leaf": 40,
                   "feature_fraction": .95, "verbosity": -1, "num_threads": 8,
                   "force_col_wise": True, "seed": 7},
                  data, num_boost_round=ROUNDS, valid_sets=[valid],
                  callbacks=[lgb.log_evaluation(50)])
print("train seconds", round(time.perf_counter() - started, 1), flush=True)
model.save_model(BUILD / f"{NAME}.txt")
results = {}
for trees in (25, 50, 100, 150, 200, ROUNDS):
    scores = {}
    for name, ix in (("valid", va), ("test", te)):
        p = base[ix] + model.predict(x.iloc[ix], num_iteration=trees, num_threads=8)
        scores[name] = competition_score(y[ix], p)
    results[str(trees)] = scores
    print(f"trees={trees:3d} valid={scores['valid']:.4f} test={scores['test']:.4f}", flush=True)
(BUILD / f"{NAME}_metrics.json").write_text(json.dumps({"rounds": ROUNDS, "rate": RATE, "leaves": LEAVES,
    "features": FOLDABLE, "scores": results}, indent=2), encoding="utf-8")
