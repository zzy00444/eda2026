"""Stage-3 feature and compact-model experiments.

This script keeps the fixed split used by the earlier experiments and adds
cheap features describing the six horizontal block bands.  They are useful
because the periodic architecture table is a lower bound that does not model
the detour around a block.  The script only writes model files; it does not
touch build/generated.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiment import BUILD, DATA, competition_score, features, stage3_features


def add_block_features(x: pd.DataFrame) -> pd.DataFrame:
    """Add integer/categorical features that can be reproduced in C++."""
    out = x.copy()
    sx = out["sx"].to_numpy(np.int32)
    sy = out["sy"].to_numpy(np.int32)
    tx = out["tx"].to_numpy(np.int32)
    ty = out["ty"].to_numpy(np.int32)
    dx = out["dx"].to_numpy(np.int32)
    dy = out["dy"].to_numpy(np.int32)
    xlo = np.minimum(sx, tx)
    xhi = np.maximum(sx, tx)
    ylo = np.minimum(sy, ty)
    yhi = np.maximum(sy, ty)
    direction = 2 * (dx < 0) + (dy < 0)
    block = out[[c for c in out.columns if c.startswith("cross_block_")
                 and c.removeprefix("cross_block_").isdigit()]].to_numpy(np.int8)
    cross = block.sum(axis=1).astype(np.int8)
    out["cross_block_count"] = cross
    out["cross_block_direction"] = pd.Series(
        (cross * 4 + direction).astype(np.int16), index=out.index, dtype="category")
    out["sy_mod50"] = sy % 50
    out["ty_mod50"] = ty % 50
    out["ylo_mod50"] = ylo % 50
    out["yhi_mod50"] = yhi % 50
    out["y_band"] = pd.Series((ylo // 100).astype(np.int8), index=out.index, dtype="category")
    out["same_y_band"] = ((sy // 100) == (ty // 100)).astype(np.int8)
    out["xlo_to_block"] = xlo - 76
    out["block_to_xhi"] = 89 - xhi
    out["x_span_block"] = np.maximum(0, np.minimum(xhi, 89) - np.maximum(xlo, 76))
    out["near_block_left"] = ((xlo < 76) & (xhi >= 76)).astype(np.int8)
    out["near_block_right"] = ((xlo <= 89) & (xhi > 89)).astype(np.int8)
    out["block_y_span"] = yhi - ylo
    # The detour choice is controlled by the smaller distance to either end of
    # a 50-row block band.  Keep both endpoint distances for tree splits.
    out["sy_edge50"] = np.minimum(sy % 50, 49 - sy % 50)
    out["ty_edge50"] = np.minimum(ty % 50, 49 - ty % 50)
    out["ylo_edge50"] = np.minimum(ylo % 50, 49 - ylo % 50)
    out["yhi_edge50"] = np.minimum(yhi % 50, 49 - yhi % 50)
    for c in ("cross_block_direction", "y_band"):
        out[c] = out[c].astype("category")
    return out


def split(n: int):
    order = np.random.default_rng(7).permutation(n)
    return order[300000:350000], order[350000:400000], np.r_[order[:300000], order[400000:]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="arch_s3_900_25l4095_lr18")
    ap.add_argument("--train-size", type=int, default=900000)
    ap.add_argument("--leaves", type=int, nargs="+", default=[511, 1023, 2047])
    ap.add_argument("--rounds", type=int, nargs="+", default=[25])
    ap.add_argument("--learning-rate", type=float, default=0.18)
    ap.add_argument("--prefix", default="stage3_block")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    frame = pd.read_csv(DATA / "data/delay_estimate_ans.csv")
    y = frame.delay.to_numpy(np.float32)
    base = np.load(BUILD / "architecture_prediction.npy").astype(np.float32)
    x, _ = features(frame)
    x["architecture_delay"] = base
    x = stage3_features(x, base)
    x = add_block_features(x)
    va, te, tr = split(len(y))
    tr = tr[:args.train_size]
    print("features", len(x.columns), "train", len(tr), flush=True)
    print("baseline", competition_score(y[va], base[va]), competition_score(y[te], base[te]), flush=True)
    residual = y - base
    categorical = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
    for leaves in args.leaves:
        for rounds in args.rounds:
            name = f"{args.prefix}_{leaves}l_{rounds}r"
            train_data = lgb.Dataset(x.iloc[tr], label=residual[tr],
                                     weight=1 / np.maximum(y[tr], 200),
                                     categorical_feature=categorical,
                                     free_raw_data=False)
            valid_data = lgb.Dataset(x.iloc[va], label=residual[va],
                                     weight=1 / np.maximum(y[va], 200),
                                     categorical_feature=categorical,
                                     reference=train_data, free_raw_data=False)
            started = time.perf_counter()
            model = lgb.train({"objective": "regression_l1", "metric": "l1",
                               "learning_rate": args.learning_rate, "num_leaves": leaves,
                               "min_data_in_leaf": 80, "feature_fraction": 0.9,
                               "verbosity": -1, "num_threads": args.threads,
                               "force_col_wise": True, "seed": 7},
                              train_data, num_boost_round=rounds,
                              valid_sets=[valid_data],
                              callbacks=[lgb.early_stopping(8, verbose=False)])
            pred = base + model.predict(x, num_threads=args.threads)
            vm = competition_score(y[va], pred[va])
            tm = competition_score(y[te], pred[te])
            print(name, "best", model.best_iteration, "valid", vm, "test", tm,
                  "train_seconds", time.perf_counter() - started, flush=True)
            model.save_model(BUILD / f"{name}.txt")
            (BUILD / f"{name}_features.json").write_text(json.dumps(list(x.columns)), encoding="utf-8")
            (BUILD / f"{name}_metrics.json").write_text(json.dumps({
                "train_size": len(tr), "valid": {"score": vm}, "test": {"score": tm},
                "best_iteration": model.best_iteration, "features": len(x.columns),
                "learning_rate": args.learning_rate, "leaves": leaves,
                "rounds": rounds, "training_seconds": time.perf_counter() - started,
            }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
