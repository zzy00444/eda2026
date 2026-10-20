"""Check all CSV rows and C++/LightGBM parity, with held-out group scores."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from experiment import DATA, BUILD, competition_score, features, pin_parts, stage3_features
from architecture import predict

parser = argparse.ArgumentParser()
parser.add_argument("--result", required=True)
parser.add_argument("--name", default="cpp_75")
parser.add_argument("--generated-dir", default="generated")
parser.add_argument("--all-rows", action="store_true", help="Also verify Python parity on all public rows")
args = parser.parse_args()
frame = pd.read_csv(DATA / "data/delay_estimate_ans.csv")
result = pd.read_csv(args.result)
assert len(result) == len(frame)
assert list(result.columns) == ["From", "To", "delay"]
assert result[["From","To"]].equals(frame[["From","To"]])
assert np.isfinite(result.delay).all() and (result.delay>=0).all()
assert (result.delay==np.rint(result.delay)).all()
export = json.loads((BUILD / args.generated_dir / "export.json").read_text())
order = np.random.default_rng(7).permutation(len(frame))
valid, test = order[300000:350000], order[350000:400000]
y, p = frame.delay.to_numpy(), result.delay.to_numpy()
metrics = {"model":export["model"], "trees":export["trees"], "output_rows":len(result)}
metrics["adaptive"] = export.get("adaptive",False)
metrics["fast_trees"] = export.get("fast_trees", 0)
metrics["refine_distance"] = export.get("refine_distance")
metrics["float32"] = export.get("float", False)
metrics["cascade_model"] = export.get("cascade_model", "")
metrics["cascade_trees"] = export.get("cascade_trees", 0)
metrics["net_correction"] = export.get("net_correction", "")
metrics["net_coefficients"] = export.get("net_coefficients", [])
metrics["packed_ports"] = export.get("packed_ports", False)
metrics["model_sha256"] = hashlib.sha256((BUILD / (export["model"]+".txt")).read_bytes()).hexdigest()
metrics["generated_header_sha256"] = hashlib.sha256((BUILD / args.generated_dir / "model.hpp").read_bytes()).hexdigest()
# Interactions in stage3_features use pandas categorical codes.  Build those
# codes on the same full public feature universe as training, then select the
# held-out rows. Recreating categories independently on a 50k slice can shift
# rare source-port codes and produce a false parity failure.
x_all, _ = features(frame)
base_all = predict(frame)
x_all["architecture_delay"] = base_all
if export.get("stage3_features", False) or "port_pair" in export["feature_names"]:
    x_all = stage3_features(x_all, base_all)
if export.get("topology_features", False):
    from topology_experiment import topology_features
    x_all, _ = topology_features(x_all)
model = lgb.Booster(model_file=str(BUILD / (export["model"]+".txt")))
cascade_model = None
if export.get("cascade_model"):
    cascade_model = lgb.Booster(model_file=str(BUILD / (export["cascade_model"]+".txt")))
    if cascade_model.feature_name() != model.feature_name():
        raise ValueError("cascade model feature names do not match primary model")
def python_prediction(x, base):
    residual = (model.predict(x,num_iteration=export["trees"],num_threads=4)
                if export["trees"] else np.zeros(len(x)))
    if export.get("adaptive"):
        refined = (x.manhattan.to_numpy()<100)|(np.abs(x.first_dx.to_numpy())==10)
        fast = model.predict(x,num_iteration=50,num_threads=4)
        residual = np.where(refined,residual,fast)
    if export.get("fast_trees"):
        refined = x.manhattan.to_numpy() < export["refine_distance"]
        fast = model.predict(x, num_iteration=export["fast_trees"], num_threads=4)
        residual = np.where(refined, residual, fast)
    if cascade_model is not None:
        distance = x.manhattan.to_numpy()
        cascade = cascade_model.predict(x, num_iteration=export["cascade_trees"], num_threads=4)
        in_band = ((distance >= export["cascade_min_distance"]) &
                   (distance < export["cascade_max_distance"]))
        residual = residual + np.where(in_band, cascade, 0)
    if export.get("net_coefficients"):
        from net_correction_experiment import net_correction_features
        net_features, _ = net_correction_features(x)
        residual += net_features.to_numpy() @ np.asarray(export["net_coefficients"])
    prediction = np.maximum(0,np.rint(base+residual))
    identical = ((x.sx.to_numpy()==x.tx.to_numpy()) & (x.sy.to_numpy()==x.ty.to_numpy()) &
                 (x.source_port.astype(int).to_numpy()==x.target_port.astype(int).to_numpy()))
    prediction[identical] = 0
    return prediction

for name, indices in (("valid",valid),("test",test)):
    prediction = python_prediction(x_all.iloc[indices], base_all[indices])
    mismatch = int(np.sum(prediction != p[indices]))
    metrics[name] = {"score":competition_score(y[indices],p[indices]),
                     "mae":float(np.abs(p[indices]-y[indices]).mean()),
                     "python_mismatch_rows":mismatch}
    print(name,metrics[name],flush=True)
    assert mismatch==0, "C++ export changed predictions"
if args.all_rows:
    prediction = python_prediction(x_all, base_all)
    mismatch = int(np.sum(prediction != p))
    metrics["full_parity"] = {"rows":len(p), "python_cpp_mismatch_rows":mismatch}
    assert mismatch==0, "Full-public C++/Python parity failed"
    print("full parity", metrics["full_parity"], flush=True)
sx,sy,sp = pin_parts(frame.From)
tx,ty,tp = pin_parts(frame.To)
distance = np.abs(tx-sx)+np.abs(ty-sy)
groups = {}
for lo,hi in ((0,10),(10,30),(30,100),(100,300),(300,1000)):
    idx = test[(distance[test]>=lo)&(distance[test]<hi)]
    groups[f"distance_{lo}_{hi}"] = {"count":len(idx),"score":competition_score(y[idx],p[idx])}
for prefix, ports in (("source",sp),("target",tp)):
    family = ports.str.split("[").str[0].to_numpy()
    for name in sorted(set(family[test])):
        idx = test[family[test]==name]
        groups[f"{prefix}_{name}"]={"count":len(idx),"score":competition_score(y[idx],p[idx])}
metrics["groups"] = groups
idx=test[((sx[test]<90)&(tx[test]>75))|((tx[test]<90)&(sx[test]>75))]
metrics["block_band"]={"count":len(idx),"score":competition_score(y[idx],p[idx])}
metrics["zero_samples"]={"count":int(np.sum(y==0)),"all_correct":bool(np.all(p[y==0]==0))}
print("distance groups", {k:v for k,v in groups.items() if k.startswith("distance")},flush=True)
print("zero samples", metrics["zero_samples"],flush=True)
(BUILD / f"{args.name}_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
