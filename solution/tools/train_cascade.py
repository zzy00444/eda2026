"""Train a compact residual model for the 100..300 Manhattan-distance band.

The primary model is evaluated with its first ``base-trees`` trees and the
new model learns only the remaining residual on the fixed training split.
The exported C++ cascade can then add this correction only for that band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiment import BUILD, competition_score
from topology_experiment import load_features, topology_features, split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="s3_topology_lr32_511l_25r")
    ap.add_argument("--base-trees", type=int, default=18)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=0.15)
    ap.add_argument("--min-distance", type=int, default=100)
    ap.add_argument("--max-distance", type=int, default=300)
    ap.add_argument("--name", default="medium_cascade_127l_10r")
    args = ap.parse_args()

    x, y = load_features()
    x, _ = topology_features(x)
    y = y.astype(np.float32)
    teacher = lgb.Booster(model_file=str(BUILD / f"{args.teacher}.txt"))
    if args.base_trees <= 0 or args.base_trees > teacher.num_trees():
        ap.error("base-trees must be within the teacher model")
    base = x.architecture_delay.to_numpy(dtype=np.float64)
    teacher_base = base + teacher.predict(x, num_iteration=args.base_trees, num_threads=4)
    teacher_full = base + teacher.predict(x, num_iteration=teacher.num_trees(), num_threads=4)
    distance = x.manhattan.to_numpy(dtype=np.int32)
    band = (distance >= args.min_distance) & (distance < args.max_distance)
    valid, test, train = split(len(y))
    train = train[band[train]]
    categorical = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
    data = lgb.Dataset(
        x.iloc[train], label=(y - teacher_base)[train],
        weight=1 / np.maximum(y[train], 200),
        categorical_feature=categorical, free_raw_data=False,
    )
    model = lgb.train(
        {
            "objective": "regression_l1", "metric": "l1",
            "learning_rate": args.learning_rate, "num_leaves": args.leaves,
            "min_data_in_leaf": 40, "feature_fraction": 0.9,
            "verbosity": -1, "num_threads": 4, "seed": 7,
            "force_col_wise": True,
        },
        data, num_boost_round=args.rounds,
    )
    correction = model.predict(x, num_threads=4)
    # Keep the primary model's full refinement for the short-distance band;
    # the cascade only replaces the 18-tree path in the medium band.
    combined = np.where(distance < args.min_distance, teacher_full, teacher_base)
    combined[band] += correction[band]
    metrics = {
        "teacher": args.teacher, "base_trees": args.base_trees,
        "leaves": args.leaves, "rounds": args.rounds,
        "learning_rate": args.learning_rate,
        "min_distance": args.min_distance, "max_distance": args.max_distance,
        "train_rows": int(len(train)), "features": int(x.shape[1]),
        "valid": competition_score(y[valid], combined[valid]),
        "test": competition_score(y[test], combined[test]),
    }
    model.save_model(str(BUILD / f"{args.name}.txt"))
    (BUILD / f"{args.name}_features.json").write_text(
        json.dumps(list(x.columns), indent=2), encoding="utf-8"
    )
    (BUILD / f"{args.name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
