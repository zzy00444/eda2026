"""Try compact objectives that better match the relative-error score."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiment import BUILD, DATA, competition_score, features, stage3_features


def main():
    frame = pd.read_csv(DATA / "data/delay_estimate_ans.csv")
    y = frame.delay.to_numpy(np.float32)
    base = np.load(BUILD / "architecture_prediction.npy").astype(np.float32)
    x, _ = features(frame)
    x["architecture_delay"] = base
    x = stage3_features(x, base)
    order = np.random.default_rng(7).permutation(len(y))
    va, te = order[300000:350000], order[350000:400000]
    tr = np.r_[order[:300000], order[400000:]]
    residual = y - base
    denoms = {"floor200": np.maximum(y, 200), "actual": np.maximum(y, 1),
              "sqrt": np.maximum(np.sqrt(y), 10), "base": np.maximum(base, 200)}
    categorical = ["source_port", "source_family", "target_port", "target_family"]
    for objective in ("regression_l1",):
      for wname, denom in denoms.items():
        for target_name in ("residual", "relative", "log"):
          if target_name == "residual":
              label = residual
          elif target_name == "relative":
              label = residual / denom
          else:
              label = np.log1p(np.maximum(y, 0)) - np.log1p(np.maximum(base, 0))
          train = lgb.Dataset(x.iloc[tr], label=label[tr], weight=1 / denom[tr],
                              categorical_feature=categorical, free_raw_data=False)
          valid = lgb.Dataset(x.iloc[va], label=label[va], weight=1 / denom[va],
                              categorical_feature=categorical, reference=train,
                              free_raw_data=False)
          model = lgb.train({"objective": objective, "metric": "l1",
                             "learning_rate": 0.18, "num_leaves": 1023,
                             "min_data_in_leaf": 80, "feature_fraction": 0.9,
                             "verbosity": -1, "num_threads": 4,
                             "force_col_wise": True, "seed": 23}, train,
                            num_boost_round=25, valid_sets=[valid],
                            callbacks=[lgb.early_stopping(8, verbose=False)])
          q = model.predict(x, num_threads=4)
          if target_name == "residual":
              pred = base + q
          elif target_name == "relative":
              pred = base + q * denom
          else:
              pred = np.expm1(np.log1p(np.maximum(base, 0)) + q)
          print(objective, wname, target_name, model.best_iteration,
                competition_score(y[va], pred[va]), competition_score(y[te], pred[te]),
                "mae", float(np.mean(np.abs(np.rint(pred[te])-y[te]))), flush=True)


if __name__ == "__main__":
    main()
