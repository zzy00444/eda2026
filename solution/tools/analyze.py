"""Evaluate architecture-only predictions on the immutable development split."""
import json
import numpy as np
import pandas as pd
from experiment import DATA, BUILD, competition_score, pin_parts
from architecture import predict

frame = pd.read_csv(DATA / "data/delay_estimate_ans.csv")
y = frame.delay.to_numpy()
prediction = predict(frame)
order = np.random.default_rng(7).permutation(len(frame))
valid, test = order[300000:350000], order[350000:400000]
for name, indices in (("valid", valid), ("test", test)):
    print(name, competition_score(y[indices], prediction[indices]),
          "mae", np.mean(np.abs(np.rint(prediction[indices])-y[indices])), flush=True)
    pd.DataFrame({"row": indices, "actual": y[indices], "prediction": prediction[indices]}).to_csv(
        BUILD / f"architecture_{name}.csv", index=False)
sx, sy, sp = pin_parts(frame.From)
tx, ty, tp = pin_parts(frame.To)
distance = np.abs(tx-sx)+np.abs(ty-sy)
for lo, hi in ((0,10),(10,30),(30,100),(100,300),(300,1000)):
    indices = test[(distance[test] >= lo) & (distance[test] < hi)]
    print("distance", lo, hi, len(indices), competition_score(y[indices], prediction[indices]), flush=True)
out = frame.iloc[test].copy()
out["prediction"] = prediction[test]
out["error"] = prediction[test]-y[test]
print(out.reindex(out.error.abs().sort_values(ascending=False).index).head(15).to_string(index=False), flush=True)
np.save(BUILD / "architecture_prediction.npy", prediction)
