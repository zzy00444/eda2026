"""Compare cheap architecture features without changing deployed artifacts."""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiment import BUILD, DATA, competition_score, features, stage3_features
from third_stage_experiment import add_block_features, split


def load_features():
    cache = BUILD / "stage3_features.pkl"
    if cache.exists():
        x = pd.read_pickle(cache)
        y = pd.read_csv(DATA / "data/delay_estimate_ans.csv", usecols=["delay"]).delay.to_numpy()
    else:
        frame = pd.read_csv(DATA / "data/delay_estimate_ans.csv")
        x, _ = features(frame)
        base_path = BUILD / "architecture_prediction.npy"
        if base_path.exists():
            base = np.load(base_path)
        else:
            from architecture import predict
            base = predict(frame)
            np.save(base_path, base)
        x["architecture_delay"] = base
        x = stage3_features(x, base)
        y = frame.delay.to_numpy()
        x.to_pickle(cache)
    return x, y


def topology_features(x):
    meta = json.loads((BUILD / "architecture_meta.json").read_text())
    table = np.load(BUILD / "architecture_table.npy", mmap_mode="r")
    sm = np.full(len(meta["ports"]), -1, np.int32)
    tm = sm.copy()
    sm[meta["sources"]] = np.arange(len(meta["sources"]))
    tm[meta["targets"]] = np.arange(len(meta["targets"]))
    sp = sm[x.source_port.astype(np.int32).to_numpy()]
    tp = tm[x.target_port.astype(np.int32).to_numpy()]
    assert (sp >= 0).all() and (tp >= 0).all()
    dx = x.dx.to_numpy(np.int32)
    dy = x.dy.to_numpy(np.int32)
    r = dx % 10 * 12 + dy % 12
    slopes = np.array(meta["slopes_scaled12"])
    bounds = np.array([(slopes[d, 0] * dx + slopes[d, 1] * dy + table[d, sp, r, tp]) / 12
                       for d in range(4)])
    gap = x.gap_x.to_numpy() + x.gap_y.to_numpy()
    envelope = bounds.max(axis=0) + gap
    out = add_block_features(x)
    out["architecture_envelope"] = envelope
    out["envelope_delta"] = envelope - x.architecture_delay.to_numpy()
    return out, envelope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaves", type=int, nargs="+", default=[127, 255, 511])
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--learning-rate", type=float, default=0.22)
    ap.add_argument("--envelope-baseline", action="store_true")
    ap.add_argument("--prefix", default="s3_topology")
    args = ap.parse_args()
    x, y = load_features()
    oldbase = x.architecture_delay.to_numpy().copy()
    x, envelope = topology_features(x)
    va, te, tr = split(len(y))
    base = envelope if args.envelope_baseline else oldbase
    print("envelope_delta", np.quantile(envelope - oldbase, [0, .5, .9, .99, 1]), flush=True)
    for label, ix in (("valid", va), ("test", te)):
        print(label, "old_base", competition_score(y[ix], oldbase[ix]),
              "envelope", competition_score(y[ix], envelope[ix]), flush=True)
    cat = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
    data = lgb.Dataset(x.iloc[tr], label=y[tr]-base[tr],
                       weight=1 / np.maximum(y[tr], 200), categorical_feature=cat)
    valid = lgb.Dataset(x.iloc[va], label=y[va]-base[va],
                        weight=1 / np.maximum(y[va], 200), categorical_feature=cat, reference=data)
    for leaves in args.leaves:
        start = time.perf_counter()
        name = f"{args.prefix}_{leaves}l_{args.rounds}r"
        m = lgb.train({"objective": "regression_l1", "metric": "l1",
                       "learning_rate": args.learning_rate, "num_leaves": leaves,
                       "min_data_in_leaf": 80, "feature_fraction": .9,
                       "verbosity": -1, "num_threads": 4, "force_col_wise": True, "seed": 7},
                      data, num_boost_round=args.rounds, valid_sets=[valid],
                      callbacks=[lgb.log_evaluation(0)])
        result = {"model": name, "leaves": leaves, "rounds": args.rounds,
                  "envelope_baseline": args.envelope_baseline,
                  "train_size": len(tr), "learning_rate": args.learning_rate}
        for label, ix in (("valid", va), ("test", te)):
            p = base[ix] + m.predict(x.iloc[ix], num_threads=4)
            result[label] = {"score": competition_score(y[ix], p),
                             "mae": float(np.abs(np.rint(p) - y[ix]).mean())}
        result["training_seconds"] = time.perf_counter() - start
        print(json.dumps(result), flush=True)
        m.save_model(BUILD / (name + ".txt"))
        (BUILD / (name + "_metrics.json")).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
