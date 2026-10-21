"""Train finer disjoint displacement tiles using train-observed port pairs only.

Generated distances are cached once; neither fitting nor weights use public delay
labels. Unobserved training pairs retain the baseline at deployment.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='6'
import argparse
import json
from pathlib import Path
import time
import numpy as np
from displacement_analyze import OUT
from displacement_assets import sha
from displacement_generated_refit import load_basis
from displacement_local_verify import load_base
from displacement_residual import indices
from project_paths import REPO_ROOT


def compress(data,raw,gap,port,rank):
    w=(1000/np.maximum(raw.astype(np.float32)+gap,200))**2
    rw=port*np.sqrt(w.mean(1));rw/=np.sqrt(np.mean(rw**2))
    cw=np.sqrt(w.mean(0));cw/=np.sqrt(np.mean(cw**2))
    data=(data*rw[:,None])*cw
    width=min(rank+24,min(data.shape));rng=np.random.default_rng(20261027)
    q,_=np.linalg.qr(rng.standard_normal((data.shape[1],width),dtype=np.float32))
    for _ in range(3):
        left,_=np.linalg.qr(data@q);q,_=np.linalg.qr(data.T@left)
    _,singular,basis=np.linalg.svd(left.T@data,full_matrices=False);basis=basis[:rank]
    coeff=(data@basis.T)/rw[:,None];physical=basis/cw
    vs=np.maximum(np.max(np.abs(physical),axis=1)/32767,np.finfo(np.float32).tiny)
    vq=np.rint(physical/vs[:,None]).clip(-32767,32767).astype('<i2')
    return coeff,vq,vs,singular


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',default='stage11_tile')
    ap.add_argument('--rank',type=int,default=96);ap.add_argument('--kind',choices=('residual','direct','both'),default='both')
    ap.add_argument('--raw',type=Path,default=Path('F:/EDA/pango_delay_handover_v3/build'))
    args=ap.parse_args();started=time.perf_counter();h,_,v,_,_,_,ids=load_basis();u,_,m=load_base()
    assert sha(args.raw/'arch.txt')==sha(OUT/'oracle/oracle_arch.txt')
    g=np.load(OUT/'geometry.npz');tr=indices(g)[0];count=np.bincount(g['si'][tr]*288+g['ti'][tr],minlength=59904)
    pairs=np.flatnonzero(count);port=np.sqrt((count[pairs]+2)/(count[pairs].mean()+2)).astype(np.float32)
    xx,yy=np.meshgrid(np.arange(-32,33),np.arange(-160,161),indexing='ij')
    loc=((xx+120)*1105+yy+552).ravel();xx=xx.ravel();yy=yy.ravel()
    scratch=OUT/(args.name+'_raw_scratch.npy')
    raw=np.lib.format.open_memmap(scratch,mode='w+',dtype='<i2',shape=(len(pairs),len(loc)));stats=[]
    for si,pid in enumerate(ids[:208]):
        path=args.raw/f'all_C_{pid}.bin';st=path.stat();assert st.st_size==288*266305*2
        lo,hi=np.searchsorted(pairs,[si*288,(si+1)*288])
        if hi>lo:
            mm=np.memmap(path,'<i2','r',shape=(288,266305))
            raw[lo:hi]=mm[np.ix_(pairs[lo:hi]-si*288,loc)]
        assert path.stat().st_mtime_ns==st.st_mtime_ns
        stats.append(dict(source=int(pid),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
        if (si+1)%32==0:print('tile generated',si+1,'elapsed',round(time.perf_counter()-started,1),flush=True)
    raw.flush();reports=[];xedges=[-32,-8,9,33];yedges=[-160,-48,0,49,161]
    for xi,(x0,x1) in enumerate(zip(xedges,xedges[1:])):
        for yi,(y0,y1) in enumerate(zip(yedges,yedges[1:])):
            use=np.flatnonzero((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1));locations=loc[use]
            actual=raw[:,use].copy();gap=(2.5*np.abs(xx[use])+.08*np.abs(yy[use])).astype(np.float32)
            base=u[pairs]@v[locations].astype(np.float32).T+m[pairs,None]
            for kind in (('residual','direct') if args.kind=='both' else [args.kind]):
                name=f'{args.name}_{kind}_{xi}{yi}';data=actual.astype(np.float32)
                if kind=='residual':data-=base
                coeff,vq,vs,singular=compress(data,actual,gap,port,args.rank)
                full=np.zeros((59904,args.rank),np.float32);full[pairs]=coeff
                target=OUT/(name+'_factors.npz')
                np.savez_compressed(target,coefficients=full,v=vq,scales=vs,locations=locations)
                report=dict(name=name,rank=args.rank,rows=59904,fit_pairs=len(pairs),columns=len(use),
                            direct=kind=='direct',refit_base=True,public_delay_labels_used=False,
                            geometry_training_rows_only=True,sparse_pair_geometry_only=True,
                            bounds_inclusive=[x0,x1-1,y0,y1-1],architecture_sha256=sha(args.raw/'arch.txt'),
                            source_stats=stats,relative_weight='generated inverse square delay row/column marginals times train port frequency',
                            singular_values=singular.tolist(),factors_sha256=sha(target),scratch_removed=False,
                            elapsed_seconds=time.perf_counter()-started)
                reports.append(report);print('tile done',name,'elapsed',round(time.perf_counter()-started,1),flush=True)
    raw._mmap.close();del raw
    assert scratch.parent.resolve()==OUT.resolve();scratch.unlink()
    for report in reports:
        report['scratch_removed']=True
        (REPO_ROOT/'docs'/(report['name']+'_training.json')).write_text(json.dumps(report,indent=2)+'\n')
    (REPO_ROOT/'docs'/(args.name+'_training.json')).write_text(json.dumps(dict(name=args.name,models=[r['name'] for r in reports],
          elapsed_seconds=time.perf_counter()-started,fit_pairs=len(pairs),public_delay_labels_used=False,scratch_removed=True),indent=2)+'\n')


if __name__=='__main__':main()
