"""Evaluate fixed, label-independent tree-budget gates for the stage-3 teacher."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiment import BUILD, DATA, competition_score, features, stage3_features


def split(n: int):
    order = np.random.default_rng(7).permutation(n)
    return order[300000:350000], order[350000:400000]


def main():
    frame = pd.read_csv(DATA / "data/delay_estimate_ans.csv")
    y = frame.delay.to_numpy(np.float32)
    base = np.load(BUILD / "architecture_prediction.npy").astype(np.float32)
    x, _ = features(frame)
    x["architecture_delay"] = base
    x = stage3_features(x, base)
    model = lgb.Booster(model_file=str(BUILD / "arch_s3_900_1023_lr10_220.txt"))
    va, te = split(len(y))
    indices = np.r_[va, te]
    xs = x.iloc[indices]
    ys = y[indices]
    bs = base[indices]
    preds = {}
    for k in (25, 45, 60, 80, 100, 120, 140, 160, 180, 200, 220):
        preds[k] = bs + model.predict(xs, num_iteration=k, num_threads=4)
        print("prefix", k, "valid", competition_score(y[va], preds[k][:len(va)]),
              "test", competition_score(y[te], preds[k][len(va):]), flush=True)

    sx = x.sx.to_numpy()[indices]; sy = x.sy.to_numpy()[indices]
    tx = x.tx.to_numpy()[indices]; ty = x.ty.to_numpy()[indices]
    dx = x.dx.to_numpy()[indices]; dy = x.dy.to_numpy()[indices]
    dist = x.manhattan.to_numpy()[indices]
    first_dx = x.first_dx.to_numpy()[indices]
    # Rank query classes using the validation set only, then check the same
    # decision on the untouched test set.
    base_score = _row_score(ys, preds[25])
    full_score = _row_score(ys, preds[220])
    gain = full_score - base_score
    gates = {
      "near": dist < 100,
      "short": dist < 300,
      "same_row": sy == ty,
      "same_col": sx == tx,
      "source_step10": np.abs(first_dx) == 10,
      "same_region": (sx // 10) == (tx // 10),
      "dist_0_30": dist < 30,
      "dist_30_100": (dist >= 30) & (dist < 100),
      "dist_100_300": (dist >= 100) & (dist < 300),
      "dist_300_plus": dist >= 300,
      "horizontal": np.abs(dx) > np.abs(dy),
      "vertical": np.abs(dy) >= np.abs(dx),
      "cross_block_band": ((sx < 90) & (tx > 75)) | ((tx < 90) & (sx > 75)),
      "dir0": (dx >= 0) & (dy >= 0),
      "dir1": (dx >= 0) & (dy < 0),
      "dir2": (dx < 0) & (dy >= 0),
      "dir3": (dx < 0) & (dy < 0),
    }
    print("groups", flush=True)
    for name, mask in gates.items():
        for part, offset in (("valid", 0), ("test", len(va))):
            m = mask[offset:offset+len(va)]
            g = gain[offset:offset+len(va)]
            print(name, part, int(m.sum()), "gain", float(g[m].mean()) if m.any() else 0,
                  "total", float(g[m].sum()), "base", float(base_score[offset:offset+len(va)][m].mean()) if m.any() else 0,
                  "full", float(full_score[offset:offset+len(va)][m].mean()) if m.any() else 0)
    # Test deterministic prefix-gate combinations selected by simple feature
    # predicates.  More expensive trees are charged in proportion to gated rows.
    print("adaptive", flush=True)
    for name, mask in gates.items():
        for k in (45, 80, 120, 160, 220):
            p = preds[25].copy()
            p[mask] = preds[k][mask]
            scores = [competition_score(y[idx], p[j*len(va):(j+1)*len(va)])
                      for j, idx in enumerate((va, te))]
            trees = 25 + float(mask.mean()) * (k - 25)
            print(name, k, "trees", round(trees, 2), "valid", scores[0], "test", scores[1], flush=True)


def _row_score(y, p):
    p = np.maximum(0, np.rint(p))
    out = np.zeros(len(y), dtype=np.float32)
    nz = y != 0
    out[nz] = 100 * (1 - np.tanh(4 * np.abs(p[nz] - y[nz]) / y[nz]))
    out[~nz] = 100 * (p[~nz] == 0)
    return out


if __name__ == "__main__":
    main()
