"""Distill a high-accuracy LightGBM model into cheap table corrections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiment import BUILD, DATA, competition_score, features, stage3_features


def split(n: int):
    o = np.random.default_rng(7).permutation(n)
    return o[300000:350000], o[350000:400000], np.r_[o[:300000], o[400000:]]


def gm(key, value, train, prior=0.0, shrink=20.0):
    key = key.astype(np.int64, copy=False)
    m = int(key.max()) + 1
    s = np.bincount(key[train], weights=value[train], minlength=m)
    n = np.bincount(key[train], minlength=m)
    out = np.full(m, prior, np.float32)
    np.divide(s, n + shrink, out=out, where=n > 0)
    return out[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="arch_s3_900_25l127_lr18")
    ap.add_argument("--teacher", default="arch_s3_900_1023_lr10_220")
    args = ap.parse_args()
    frame = pd.read_csv(DATA / "data/delay_estimate_ans.csv")
    y = frame.delay.to_numpy(np.float32)
    base = np.load(BUILD / "architecture_prediction.npy").astype(np.float32)
    x, _ = features(frame)
    x["architecture_delay"] = base
    x = stage3_features(x, base)
    print("features", len(x.columns), flush=True)
    bm = lgb.Booster(model_file=str(BUILD / f"{args.base_model}.txt"))
    tm = lgb.Booster(model_file=str(BUILD / f"{args.teacher}.txt"))
    bp = base + bm.predict(x, num_threads=4)
    tp = base + tm.predict(x, num_threads=4)
    va, te, tr = split(len(y))
    print("base", competition_score(y[va], bp[va]), competition_score(y[te], bp[te]),
          "teacher", competition_score(y[va], tp[va]), competition_score(y[te], tp[te]), flush=True)
    d = y - bp
    td = tp - bp

    # Distill the teacher's incremental correction into a second compact
    # booster.  It is evaluated as base + student_delta, so the deployed
    # estimator can use a small base model and a small residual model.
    categorical = ["source_port", "source_family", "target_port", "target_family"]
    for leaves, rounds, lr in ((127, 25, 0.18), (255, 25, 0.18),
                               (255, 40, 0.12), (511, 25, 0.18)):
        train_data = lgb.Dataset(x.iloc[tr], label=td[tr],
                                 weight=1 / np.maximum(y[tr], 200),
                                 categorical_feature=categorical,
                                 free_raw_data=False)
        valid_data = lgb.Dataset(x.iloc[va], label=td[va],
                                 weight=1 / np.maximum(y[va], 200),
                                 categorical_feature=categorical,
                                 reference=train_data, free_raw_data=False)
        student = lgb.train({"objective": "regression_l1", "metric": "l1",
                             "learning_rate": lr, "num_leaves": leaves,
                             "min_data_in_leaf": 80, "feature_fraction": 0.9,
                             "verbosity": -1, "num_threads": 4,
                             "force_col_wise": True, "seed": 17},
                            train_data, num_boost_round=rounds,
                            valid_sets=[valid_data],
                            callbacks=[lgb.early_stopping(10, verbose=False)])
        spred = student.predict(x, num_threads=4)
        pred = bp + spred
        print(f"student leaves {leaves} rounds {rounds} best {student.best_iteration}",
              competition_score(y[va], pred[va]), competition_score(y[te], pred[te]),
              "delta_mae", float(np.mean(np.abs(spred[te] - td[te]))), flush=True)

    # Fit corrections to the actual residual, then separately to the teacher
    # delta.  A correction is accepted only if it improves both fixed slices.
    sx = x.sx.to_numpy(np.int64); sy = x.sy.to_numpy(np.int64)
    tx = x.tx.to_numpy(np.int64); ty = x.ty.to_numpy(np.int64)
    sp = x.source_port.cat.codes.to_numpy(np.int64)
    tpcode = x.target_port.cat.codes.to_numpy(np.int64)
    sb = x.source_bit.to_numpy(np.int64); tb = x.target_bit.to_numpy(np.int64)
    dx = x.dx.to_numpy(np.int64); dy = x.dy.to_numpy(np.int64)
    ap = x.architecture_minus_linear.to_numpy(np.int64)
    dist = x.manhattan.to_numpy(np.int64)
    keys = {
      "sp": sp, "tp": tpcode, "pair": sp*512+tpcode,
      "sb": sb+32, "tb": tb+32, "bitpair": (sb+32)*80+(tb+32),
      "sx": sx, "tx": tx, "sy100": sy%100, "ty100": ty%100,
      "dx": dx+120, "dy": dy+552, "dxmod": dx%10, "dymod": dy%12,
      "dir": 2*(dx<0)+(dy<0), "distbucket": (dist>=10)+(dist>=30)+(dist>=100)+(dist>=300),
      "archlin": np.clip(ap+400, 0, 1000),
    }
    keys = {name: np.asarray(value, dtype=np.int64) for name, value in keys.items()}
    order = ["pair", "sp", "tp", "bitpair", "tb", "sb", "sx", "tx", "sy100", "ty100",
             "dx", "dy", "dxmod", "dymod", "dir", "distbucket", "archlin"]
    # Keep one effect table per key.  Updating the same aggregate twice would
    # cancel the update; explicit effect tables make the backfit meaningful.
    effects = {name: np.zeros(int(keys[name].max()) + 1, np.float32)
               for name in order}
    cur = np.zeros(len(y), np.float32)
    # Alternating backfit of independent effects.  The table itself is small;
    # this experiment reports how much of the teacher gap is recoverable.
    for it in range(4):
        for name in order:
            k = keys[name]
            old = effects[name]
            cur -= old[k]
            new = gm(k, d-cur, tr, shrink=30.0)
            effects[name] = new
            cur += new[k]
        for lam in (0.5, 1.0, 1.5):
            p = bp + lam*cur
            print(f"add {it+1} lam {lam}", competition_score(y[va],p[va]),
                  competition_score(y[te],p[te]), flush=True)

    # One pair-conditioned displacement table with shrinkage.  This is larger
    # but still only 59,904*120 float entries and can be packed as int16.
    pair = sp*512+tpcode
    md = (dx % 10)*12 + (dy % 12)
    for sh in (2, 8, 30, 100):
        k = pair*120 + md
        c = gm(k, d, tr, shrink=sh)
        for lam in (0.5, 1.0):
            p = bp + lam*c
            print(f"pairmod sh {sh} lam {lam}", competition_score(y[va],p[va]),
                  competition_score(y[te],p[te]), flush=True)


if __name__ == "__main__":
    main()
