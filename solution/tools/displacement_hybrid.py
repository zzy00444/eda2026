"""Screen a few global residual trees and a gated wall model atop a frozen LUT."""
import argparse
import json
import numpy as np
import lightgbm as lgb
from displacement_analyze import OUT, points, describe
from displacement_residual import make_features, indices, CATEGORIES
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',default='stage10_hybrid')
    ap.add_argument('--lut',default='stage10_lut_small')
    ap.add_argument('--leaves',type=int,default=127)
    ap.add_argument('--rounds',type=int,default=16)
    args=ap.parse_args()
    g=np.load(OUT/'geometry.npz');y=g['y'];base=np.load(OUT/'displacement_cal16_prediction.npy')
    lut=np.load(OUT/(args.lut+'_reference.npy'));bound=np.load(OUT/'stage10_bound.npy')
    x=make_features(g,base);x['bound']=bound.astype(np.float32);x['base_minus_bound']=(base-bound).astype(np.float32)
    x['lut_minus_base']=(lut-base).astype(np.float32)
    tr,va,_,held=indices(g); wall=x.landing_wall.to_numpy()>0
    pars=dict(objective='regression_l1',learning_rate=.6,num_leaves=args.leaves,min_data_in_leaf=150,
              lambda_l2=10,max_bin=255,num_threads=6,verbosity=-1,seed=20261021,
              deterministic=True,force_col_wise=True,cat_smooth=30,cat_l2=20,max_cat_threshold=32)
    tr0=tr[~wall[tr]]
    data=lgb.Dataset(x.iloc[tr0],(y-lut)[tr0],weight=2000/np.maximum(y[tr0],200),categorical_feature=CATEGORIES)
    model=lgb.train(pars,data,num_boost_round=args.rounds);model.save_model(str(OUT/(args.name+'_global.txt')))
    # A separate model is evaluated only on forced Net landings inside a wall.
    trw=tr[wall[tr]]
    wp=dict(pars,num_leaves=31,min_data_in_leaf=25,learning_rate=.1,lambda_l2=5)
    wm=lgb.train(wp,lgb.Dataset(x.iloc[trw],(y-base)[trw],weight=2000/np.maximum(y[trw],200),categorical_feature=CATEGORIES),num_boost_round=128)
    wm.save_model(str(OUT/(args.name+'_wall.txt')))
    curve=[];wallpred={}
    for n in [16,32,64,128]:
        pred=np.maximum(bound[va[wall[va]]],np.floor(base[va[wall[va]]]+wm.predict(x.iloc[va[wall[va]]],num_iteration=n,num_threads=6)+.5))
        score=float(points(y[va[wall[va]]],pred).mean());wallpred[n]=pred
        print('wall',n,score,flush=True)
    wn=max(wallpred,key=lambda n:points(y[va[wall[va]]],wallpred[n]).mean())
    best=-1
    for n in [1,2,3,4,6,8,12,16]:
        if n>args.rounds:continue
        pred=np.maximum(bound[va],np.floor(lut[va]+model.predict(x.iloc[va],num_iteration=n,num_threads=6)+.5))
        pred[wall[va]]=wallpred[wn]
        row={'trees':n,'valid_accuracy':float(points(y[va],pred).mean())}
        print(json.dumps(row),flush=True);curve.append(row)
        if row['valid_accuracy']>best:best=row['valid_accuracy'];bn=n
    report=dict(name=args.name,lut=args.lut,leaves=args.leaves,global_curve=curve,wall_trees=wn,
                wall_train_rows=len(trw),selected_global_trees=bn,test_labels_used=False)
    (REPO_ROOT/'docs'/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
