"""Train a compact cascade correction over an existing stage-3 model."""
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from experiment import DATA, BUILD, competition_score, features, stage3_features
from architecture import predict

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--base-trees", type=int, required=True)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--leaves", type=int, default=63)
parser.add_argument("--learning-rate", type=float, default=0.1)
parser.add_argument("--name", required=True)
args = parser.parse_args()
frame = pd.read_csv(DATA / "data" / "delay_estimate_ans.csv")
x, _ = features(frame)
baseline = predict(frame)
x["architecture_delay"] = baseline
x = stage3_features(x, baseline)
y = frame.delay.to_numpy(dtype=np.float32)
order = np.random.default_rng(7).permutation(len(frame))
valid, test = order[300000:350000], order[350000:400000]
train = np.concatenate((order[:300000], order[400000:]))
base_model = lgb.Booster(model_file=str(BUILD / (args.base + ".txt")))
base_res = base_model.predict(x, num_iteration=args.base_trees, num_threads=4)
base_pred = baseline + base_res
target = y - base_pred
categorical = ["source_port", "source_family", "target_port", "target_family"]
train_data = lgb.Dataset(x.iloc[train], label=target[train], weight=1/np.maximum(y[train],200),
                         categorical_feature=categorical, free_raw_data=False)
valid_data = lgb.Dataset(x.iloc[valid], label=target[valid], reference=train_data,
                         weight=1/np.maximum(y[valid],200), categorical_feature=categorical,
                         free_raw_data=False)
model = lgb.train({"objective":"regression_l1","metric":"l1","learning_rate":args.learning_rate,
                   "num_leaves":args.leaves,"min_data_in_leaf":80,"feature_fraction":0.9,
                   "verbosity":-1,"num_threads":4,"force_col_wise":True,"seed":7},
                  train_data, num_boost_round=args.rounds, valid_sets=[valid_data],
                  callbacks=[lgb.log_evaluation(0)])
for label, idx in (("valid",valid),("test",test)):
    corr=model.predict(x.iloc[idx],num_threads=4)
    pred=base_pred[idx]+corr
    print(label, competition_score(y[idx],pred), np.mean(np.abs(np.rint(pred)-y[idx])))
model.save_model(str(BUILD / (args.name+".txt")))
(BUILD/(args.name+"_meta.json")).write_text(json.dumps({"base":args.base,"base_trees":args.base_trees,
    "rounds":args.rounds,"leaves":args.leaves},indent=2))
