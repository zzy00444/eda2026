"""Stage-11 real-branch baseline/calibration ablation; train/valid only."""
import json
import numpy as np
from displacement_analyze import OUT, points
from displacement_assets import sha
from displacement_local_verify import load_base, load_local, predict, round_away
from displacement_residual import indices, make_features
from project_paths import REPO_ROOT


def fit_calibration(raw,g,train,prior=None):
    prior=np.ones(1440,np.float32) if prior is None else prior.copy()
    bins=np.searchsorted([700,1000,2000,4000],raw,side='right')
    groups=g['ti']*5+bins
    tr=train[(g['y'][train]>0)&(raw[train]>0)]
    ratio=g['y'][tr].astype(float)/raw[tr]
    weights=raw[tr].astype(float)/g['y'][tr]
    factors=prior.copy();counts=np.bincount(groups[tr],minlength=1440)
    for group in np.unique(groups[tr]):
        ix=np.flatnonzero(groups[tr]==group)
        if len(ix)<50:continue
        order=ix[np.argsort(ratio[ix])];cs=np.cumsum(weights[order])
        median=ratio[order[np.searchsorted(cs,cs[-1]*.5)]]
        factors[group]=prior[group]+(np.clip(median,.9,1.1)-prior[group])*len(ix)/(len(ix)+50)
    return factors,counts


def calibrated(raw,g,cal):
    bins=np.searchsorted([700,1000,2000,4000],raw,side='right')
    return np.maximum(0,round_away(raw.astype(float)*cal[g['ti']*5+bins])).astype(np.int32)


def main():
    g=np.load(OUT/'geometry.npz');tr,va,_,held=indices(g)
    local=load_local(OUT/'stage10_local96.local');base=load_base()
    old=np.fromfile(OUT/'calibration_train.bin','<f4')
    original_m=np.fromfile(OUT/'oracle/oracle_M.bin','<i2',offset=12).astype(np.int32)
    bias=original_m-base[2].astype(np.int32)
    fixed_wall=np.load(OUT/'stage10_local96w64_prediction.npy')
    wall=make_features(g,fixed_wall).landing_wall.to_numpy()>0
    tr=tr[~wall[tr]];records=[]
    for align in (False,True):
        name='stage11_match_'+('aligned' if align else 'current')
        raw,_=predict(g,base,local,local_bias=bias if align else None,return_raw=True)
        np.save(OUT/(name+'_raw.npy'),raw)
        fitted,counts=fit_calibration(raw,g,tr,old)
        for alpha in (None,0.,.5,1.):
            cal=old if alpha is None else (old+(fitted-old)*alpha).astype(np.float32)
            if alpha==0.:cal=np.ones_like(cal)
            pred=calibrated(raw,g,cal);pred[wall]=fixed_wall[wall]
            item=dict(aligned=align,calibration='old' if alpha is None else ('unity' if alpha==0 else str(alpha)),
                      valid_accuracy=float(points(g['y'][va],pred[va]).mean()),
                      held_accuracy=float(points(g['y'][held],pred[held]).mean()))
            records.append(item);print(item,flush=True)
            if alpha is None and not align:
                known=np.load(OUT/'stage10_local96_prediction.npy');p=calibrated(raw,g,cal)
                assert np.array_equal(p,known),np.count_nonzero(p!=known)
            if alpha in (.5,1.):cal.tofile(OUT/(name+'_cal'+str(alpha).replace('.','')+'.bin'))
        assert len(counts)==1440
    report=dict(real_both_branches=True,test_labels_used=False,wall_outputs_fixed_for_isolation=True,
                training_rows=len(tr),validation_rows=len(va),configs=records,
                best=max(records,key=lambda a:a['valid_accuracy']),m_local_bias_range=[int(bias.min()),int(bias.max())],
                frozen_local_sha256=sha(OUT/'stage10_local96.local'))
    (REPO_ROOT/'docs/stage11_match.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
