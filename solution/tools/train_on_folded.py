"""Train a residual tree on top of the folded prediction.

The folded table cannot see the continuous displacement or the absolute
position, so its residual still has structure a tree can exploit.  Using it as
the baseline instead of the periodic lower bound raises the starting accuracy
from 91.40 to 94.61 without changing the runtime table.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# run from solution/tools so sibling modules import directly
from project_paths import BUILD  # noqa: E402
from experiment import competition_score  # noqa: E402
from topology_experiment import load_features, topology_features  # noqa: E402
from third_stage_experiment import split  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
LEAVES = int(sys.argv[2]) if len(sys.argv) > 2 else 511
RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.32
NAME = sys.argv[4] if len(sys.argv) > 4 else "s8_onfold_511l_25r"
FOLD = BUILD / "generated_s7_fold" / "architecture.bin"

x, y = load_features()
x, _ = topology_features(x)
meta = json.loads((BUILD / "architecture_meta.json").read_text())
slopes = np.array(meta["slopes_scaled12"])
n_src, n_tgt = len(meta["sources"]), len(meta["targets"])
table = np.fromfile(FOLD, dtype="<i2").reshape(4, n_src, 120, n_tgt)
src_map = {p: i for i, p in enumerate(meta["sources"])}
tgt_map = {p: i for i, p in enumerate(meta["targets"])}
sidx = np.array([src_map[p] for p in x.source_port.astype(np.int32).to_numpy()], dtype=np.int64)
tidx = np.array([tgt_map[p] for p in x.target_port.astype(np.int32).to_numpy()], dtype=np.int64)
dx = x.dx.to_numpy(np.int64)
dy = x.dy.to_numpy(np.int64)
direction = (2 * (dx < 0) + (dy < 0)).astype(np.int64)
r = (dx % 10) * 12 + (dy % 12)
value = table[direction, sidx, r, tidx].astype(np.int64)
gaps = x.gap_x.to_numpy(np.int64) + x.gap_y.to_numpy(np.int64)
base = (slopes[direction, 0] * dx + slopes[direction, 1] * dy + value) / 12.0 + gaps
x = x.copy()
x["architecture_delay"] = base

va, te, tr = split(len(y))
print("folded base valid %.4f test %.4f" % (
    competition_score(y[va], base[va]), competition_score(y[te], base[te])), flush=True)
cat = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
data = lgb.Dataset(x.iloc[tr], label=y[tr] - base[tr],
                   weight=1 / np.maximum(y[tr], 200), categorical_feature=cat)
valid = lgb.Dataset(x.iloc[va], label=y[va] - base[va],
                    weight=1 / np.maximum(y[va], 200), categorical_feature=cat, reference=data)
started = time.perf_counter()
model = lgb.train({"objective": "regression_l1", "metric": "l1",
                   "learning_rate": RATE, "num_leaves": LEAVES, "min_data_in_leaf": 80,
                   "feature_fraction": .9, "verbosity": -1, "num_threads": 8,
                   "force_col_wise": True, "seed": 7},
                  data, num_boost_round=ROUNDS, valid_sets=[valid],
                  callbacks=[lgb.log_evaluation(0)])
print("train seconds", round(time.perf_counter() - started, 1), flush=True)
model.save_model(BUILD / f"{NAME}.txt")
results = {}
for trees in (1, 2, 3, 5, 10, ROUNDS):
    scores = {}
    for name, ix in (("valid", va), ("test", te)):
        p = base[ix] + model.predict(x.iloc[ix], num_iteration=trees, num_threads=8)
        scores[name] = competition_score(y[ix], p)
    results[str(trees)] = scores
    print(f"trees={trees:3d} valid={scores['valid']:.4f} test={scores['test']:.4f}", flush=True)
(BUILD / f"{NAME}_metrics.json").write_text(json.dumps(
    {"rounds": ROUNDS, "leaves": LEAVES, "rate": RATE, "base": "folded stage 7",
     "scores": results}, indent=2), encoding="utf-8")
