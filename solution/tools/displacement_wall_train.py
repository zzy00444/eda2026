"""Train a sparse Net-landing wall residual without an architecture envelope."""
import argparse
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from displacement_analyze import OUT, points, describe
from displacement_residual import make_features, indices, CATEGORIES
from displacement_assets import sha
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base')
    ap.add_argument('--name',required=True)
    args=ap.parse_args()
    g=np.load(OUT/'geometry.npz');y=g['y']
    pred_path=OUT/(args.base+'_prediction.npy')
    # Rebuild from this run's CSV; a stale NPY must never select training labels.
    base=pd.read_csv(OUT/(args.base+'_result.csv'),usecols=['delay']).delay.to_numpy(np.int32)
    assert len(base)==len(y)
    np.save(pred_path,base)
    x=make_features(g,base);wall=x.landing_wall.to_numpy()>0
    tr,va,_,held=indices(g);tr=tr[wall[tr]];vw=va[wall[va]]
    pars=dict(objective='regression_l1',learning_rate=.1,num_leaves=31,min_data_in_leaf=25,
              lambda_l2=5,max_bin=255,num_threads=6,verbosity=-1,seed=20261021,
              deterministic=True,force_col_wise=True,cat_smooth=30,cat_l2=20,max_cat_threshold=32)
    model=lgb.train(pars,lgb.Dataset(x.iloc[tr],(y-base)[tr],weight=2000/np.maximum(y[tr],200),
                                   categorical_feature=CATEGORIES),num_boost_round=160)
    path=OUT/(args.name+'.txt');model.save_model(str(path))
    curve=[]
    for n in (16,32,64,96,128,160):
        pred=np.maximum(0,np.floor(base[vw]+model.predict(x.iloc[vw],num_iteration=n,num_threads=6)+.5)).astype(np.int32)
        item=dict(trees=n,wall_valid_accuracy=float(points(y[vw],pred).mean()),
                  whole_valid_accuracy=float((points(y[va],base[va]).sum()-points(y[vw],base[vw]).sum()+points(y[vw],pred).sum())/len(va)))
        curve.append(item);print(json.dumps(item),flush=True)
    best=max(curve,key=lambda a:a['whole_valid_accuracy'])
    hw=held[wall[held]]
    ph=np.maximum(0,np.floor(base[hw]+model.predict(x.iloc[hw],num_iteration=best['trees'],num_threads=6)+.5)).astype(np.int32)
    report=dict(name=args.name,base=args.base,base_prediction_sha256=sha(OUT/(args.base+'_prediction.npy')),
                params=pars,fit_rows=len(tr),held_wall_rows=len(hw),features=list(x.columns),
                curve=curve,selected=best,held_wall_baseline=describe(y[hw],base[hw]),held_wall_updated=describe(y[hw],ph),
                model_sha256=sha(path),test_labels_used=False,architecture_bound_used=False,
                caveat='New residual excludes held source coordinates; inherited stage8 does not')
    (REPO_ROOT/'docs'/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
