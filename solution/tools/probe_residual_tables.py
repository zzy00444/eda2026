"""Probe cheap residual lookup keys against the fixed split."""
import numpy as np
import pandas as pd
from experiment import DATA, BUILD, competition_score, features
from architecture import predict

frame = pd.read_csv(DATA / "data" / "delay_estimate_ans.csv")
x, _ = features(frame)
base = predict(frame)
y = frame.delay.to_numpy(dtype=np.float64)
res = y - base
order = np.random.default_rng(7).permutation(len(frame))
valid, test = order[300000:350000], order[350000:400000]
train = np.concatenate((order[:300000], order[400000:]))
work = x.copy()
work["direction"] = 2 * (work.dx.to_numpy() < 0) + (work.dy.to_numpy() < 0)
work["res"] = res
work["base"] = base
keys = {
    "port_dxmod": ["source_port", "target_port", "dx_mod", "dy_mod"],
    "port_dir_dxmod": ["source_port", "target_port", "direction", "dx_mod", "dy_mod"],
    "family_dxmod": ["source_family", "target_family", "dx_mod", "dy_mod"],
    "family_dir_dxmod": ["source_family", "target_family", "direction", "dx_mod", "dy_mod"],
    "port_distance": ["source_port", "target_port", "manhattan"],
    "geometry": ["direction", "dx_mod", "dy_mod", "gap_x", "gap_y"],
}
for name, cols in keys.items():
    # Use training rows only so reported scores remain out-of-sample.
    train_frame = work.iloc[train]
    means = train_frame.groupby(cols, observed=True)["res"].agg(["mean", "count"])
    joined = work.iloc[test][cols].merge(means, left_on=cols, right_index=True, how="left")
    correction = joined["mean"].fillna(0).to_numpy()
    pred = base[test] + correction
    print(name, "groups", len(means), "coverage", np.mean(joined["count"].notna()),
          "score", competition_score(y[test], pred), "mae", np.mean(np.abs(np.rint(pred)-y[test])))
    for min_count in (2, 3, 5, 10, 20):
        use = joined["count"].fillna(0).to_numpy() >= min_count
        blended = base[test] + correction * use
        print("  min", min_count, "coverage", use.mean(), "score", competition_score(y[test], blended))
