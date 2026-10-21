"""Rank disjoint local prefixes on validation only, keeping old branch choices."""
import argparse
import itertools
import json
import numpy as np
from displacement_analyze import OUT,points,split
from displacement_local_screen import quantize
from displacement_local_verify import fma_dot,round_away
from displacement_match import calibrated
from displacement_residual import make_features,indices
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--part',action='append',required=True)
    ap.add_argument('--name',default='stage11_partition_screen')
    ap.add_argument('--sparse-pairs',action='store_true');ap.add_argument('--bits',type=int,choices=(8,16),default=8)
    ap.add_argument('--no-combinations',action='store_true')
    args=ap.parse_args()
    g=np.load(OUT/'geometry.npz');_,va,_=split(len(g['y']));vg={k:g[k][va] for k in g.files}
    trace=np.fromfile(OUT/'displacement_caltrace_trace.bin','<i4').reshape(-1,4)[va]
    old=np.fromfile(OUT/'calibration_train.bin','<f4');base=np.load(OUT/'displacement_cal16_prediction.npy')[va]
    wall=make_features(vg,base).landing_wall.to_numpy()>0
    fixed=np.load(OUT/'stage10_local96w64_prediction.npy')[va];base[wall]=fixed[wall]
    selected=np.arange(59904);pairmap=np.arange(59904)
    if args.sparse_pairs:
        tr=indices(g)[0];selected=np.unique(g['si'][tr]*288+g['ti'][tr])
        pairmap=np.full(59904,-1,np.int32);pairmap[selected]=np.arange(len(selected))
    baseline=float(points(vg['y'],base).mean());curves=[]
    for spec in args.part:
        name,ranks=spec.split(':');f=np.load(OUT/(name+'_factors.npz'))
        training=json.loads((REPO_ROOT/'docs'/(name+'_training.json')).read_text());direct=training.get('direct',False)
        mapping=np.full(266305,-1,np.int32);mapping[f['locations']]=np.arange(len(f['locations']))
        col=mapping[trace[:,1]];use=(col>=0)&~wall&(pairmap[trace[:,0]]>=0);ix=np.flatnonzero(use);items=[]
        for rank in map(int,ranks.split(',')):
            if rank:
                uq,us=quantize(f['coefficients'][selected,:rank],args.bits);u=(uq.astype(np.float32)*us)*f['scales'][:rank]
                delta=round_away(fma_dot(u[pairmap[trace[ix,0]]],f['v'][:rank,col[ix]].T))
                raw=(trace[ix,2] if direct else trace[ix,3])+delta
                subg={k:vg[k][ix] for k in vg};pred=calibrated(raw,subg,old)
                gain=float((points(vg['y'][ix],pred)-points(vg['y'][ix],base[ix])).sum()/len(va))
            else:gain=0.
            size=len(selected)*rank*(args.bits//8)+len(f['locations'])*rank*2+rank*8+(59904*4 if args.sparse_pairs else 0)
            item=dict(factors=name,rank=rank,bits=args.bits,direct=direct,valid_proxy_gain=gain,coverage=len(ix)/len(va),
                      asset_bytes_excluding_map=size,expected_extra_dot_dimensions=rank*len(ix)/len(va))
            print(item,flush=True);items.append(item)
        curves.append(items)
    combos=[]
    for choice in ([] if args.no_combinations else itertools.product(*curves)):
        combos.append(dict(parts=[{'factors':c['factors'],'rank':c['rank']} for c in choice],
                           valid_proxy_accuracy=baseline+sum(c['valid_proxy_gain'] for c in choice),
                           local_bytes=4*266305+32+16*len(choice)+sum(c['asset_bytes_excluding_map'] for c in choice),
                           extra_dot_dimensions=sum(c['expected_extra_dot_dimensions'] for c in choice)))
    combos.sort(key=lambda c:c['valid_proxy_accuracy'],reverse=True)
    report=dict(selected_branch_proxy=True,test_labels_used=False,wall_outputs_fixed=True,
                baseline_valid_accuracy=baseline,curves=curves,combinations=combos)
    (REPO_ROOT/'docs'/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('TOP',json.dumps(combos[:8],indent=2),flush=True)


if __name__=='__main__':main()
