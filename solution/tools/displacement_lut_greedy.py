"""Validation-selected extra additive tables on ordinary requests only."""
import itertools
import json
import numpy as np
from displacement_compact import compact_features,fit_medians
from displacement_residual import indices
from displacement_analyze import OUT,points,describe
from project_paths import REPO_ROOT


def main():
    g=np.load(OUT/'geometry.npz');y=g['y'];base=np.load(OUT/'displacement_cal16_prediction.npy')
    name='stage10_lut_greedy';v,bd,bins=compact_features(g,base)
    lm=json.loads((REPO_ROOT/'docs/stage10_lut_small.json').read_text());old=np.load(OUT/'stage10_lut_small.npz')
    definitions=lm['definitions'];shapes=lm['shapes'];tables=[old['table'+str(i)] for i in range(len(definitions))]
    delta=np.zeros(len(y))
    for term,shape,tb in zip(definitions,shapes,tables):
        key=np.zeros(len(y),np.int64)
        for col,n in zip(term,shape):key=key*n+v[col]
        delta+=tb[key]
    tr,va,_,_=indices(g);tr=tr[v['landing_wall'][tr]==0];va=va[v['landing_wall'][va]==0]
    wt=2000/np.maximum(y[tr],200);target=y.astype(float)-base
    pool=[['source','target_family'],['source_family','target'],['source','target','direction'],
          ['source','rx_bin'],['target','rx_bin'],['source','ry_bin'],['target','ry_bin'],
          ['source','distance_bin'],['target','distance_bin'],
          ['source','landing_cross'],['target','landing_cross'],
          ['source','margin_bin','target_family'],['source_family','margin_bin','target'],
          ['source','source_phase'],['target','target_phase'],
          ['source','rx_mod'],['target','ry_mod'],['source','ry_mod'],['target','rx_mod'],
          ['source','sx_mod'],['target','tx_mod'],['source','sy_mod'],['target','ty_mod'],
          ['source_family','target_family','margin_bin'],
          ['margin_bin','rx_bin'],['margin_bin','ry_bin'],['margin_bin','cross'],
          ['landing_cross','source_door_distance_bin','target_door_distance_bin'],
          ['source','edge_x_bin'],['target','edge_x_bin'],['source','edge_y_bin'],['target','edge_y_bin']]
    keys=[]
    for term in pool:
        shape=[int(v[k].max())+1 for k in term];key=np.zeros(len(y),np.int64)
        for k,n in zip(term,shape):key=key*n+v[k]
        keys.append((key,shape,int(np.prod(shape))))
    curve=[]
    def score(d):
        p=np.maximum(bd[va],np.maximum(0,np.floor(base[va]+d+.5)))
        return float(points(y[va],p).mean())
    for step in range(8):
        best=(score(delta[va]),None)
        for term,(key,shape,size) in zip(pool,keys):
            if size>4_000_000:continue
            for shrink in [16,64,256]:
                tb=fit_medians(key[tr],target[tr]-delta[tr],wt,size,shrink)
                sc=score(delta[va]+tb[key[va]])
                if sc>best[0]:best=(sc,(term,shape,key,tb,shrink))
        if best[1] is None:break
        term,shape,key,tb,shrink=best[1];delta+=tb[key]
        definitions.append(term);shapes.append(shape);tables.append(tb)
        row=dict(step=step+1,term=term,shrink=shrink,valid_free_accuracy=best[0]);curve.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(OUT/(name+'.npz'),**{'table'+str(i):z for i,z in enumerate(tables)})
        p=np.maximum(bd,np.maximum(0,np.floor(base+delta+.5))).astype(np.int32)
        np.save(OUT/(name+'_reference.npy'),p)
        report=dict(name=name,definitions=definitions,shapes=shapes,bins=bins,curve=curve,test_labels_used=False,
                    valid_free=describe(y[va],p[va]),fit_rows=len(tr),cells=sum(t.size for t in tables),
                    caveat='Forward selection on historical validation; inherited LUT and base have prior training scope')
        (REPO_ROOT/'docs'/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
