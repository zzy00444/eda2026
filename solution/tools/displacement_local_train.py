"""Compress generated short-displacement residuals with a separate local basis.

No public delay labels are used. Optional port sampling weights use only the
authorized training geometry. Completed external distance matrices are read-only.
"""
from __future__ import annotations
import os
for _var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_var]='6'
import argparse
import json
from pathlib import Path
import struct
import time
import zlib
import numpy as np
from displacement_generated_refit import load_basis
from displacement_analyze import OUT
from displacement_residual import indices
from displacement_assets import sha
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',default='stage10_local')
    ap.add_argument('--raw',type=Path,default=Path('F:/EDA/pango_delay_handover_v3/build'))
    ap.add_argument('--lx',type=int,default=24)
    ap.add_argument('--ly',type=int,default=120)
    ap.add_argument('--rank',type=int,default=160)
    ap.add_argument('--port-weight',action='store_true')
    ap.add_argument('--exclude-lx',type=int,default=-1)
    ap.add_argument('--exclude-ly',type=int,default=-1)
    ap.add_argument('--relative-power',type=float,default=0.,choices=(0.,1.,2.))
    ap.add_argument('--refit-base',action='store_true')
    ap.add_argument('--direct',action='store_true',help='Compress generated distances directly, replacing the base dot in covered regions')
    args=ap.parse_args();started=time.perf_counter()
    h,scale,v,uh,u,m,ids=load_basis();ns,nt,rank=map(int,uh[:3]);rows=ns*nt
    assert sha(args.raw/'arch.txt')==sha(OUT/'oracle/oracle_arch.txt')
    xx,yy=np.meshgrid(np.arange(-args.lx,args.lx+1),np.arange(-args.ly,args.ly+1),indexing='ij')
    use=~((np.abs(xx)<=args.exclude_lx)&(np.abs(yy)<=args.exclude_ly))
    loc=((xx-int(h[4]))*int(h[3])+yy-int(h[5]))[use].ravel();cols=len(loc)
    if args.refit_base:
        from displacement_local_verify import load_base
        ru,_,rm=load_base()
        # load_base U already includes the V scale.
        u=ru/scale;m=rm.astype(np.float32)
    weight=np.ones(rows,np.float32)
    if args.port_weight:
        g=np.load(OUT/'geometry.npz');tr=indices(g)[0]
        count=np.bincount((g['si'][tr]*nt+g['ti'][tr]),minlength=rows)
        weight=np.sqrt((count+2)/(count.mean()+2)).astype(np.float32)
    np.save(OUT/(args.name+'_row_weight.npy'),weight)
    scratch=OUT/(args.name+'_scratch.npy')
    residual=np.lib.format.open_memmap(scratch,mode='w+',dtype=np.float32,shape=(rows,cols))
    localv=v[loc].astype(np.float32)*scale
    stats=[];relative_sum=np.zeros(cols,np.float64)
    gap=2.5*np.abs(loc//int(h[3])+int(h[4]))+.08*np.abs(loc%int(h[3])+int(h[5]))
    for si,pid in enumerate(ids[:ns]):
        path=args.raw/f'all_C_{pid}.bin';st=path.stat();assert st.st_size==nt*int(h[1])*2
        raw=np.memmap(path,'<i2','r',shape=(nt,int(h[1])))[:,loc].astype(np.float32)
        sl=slice(si*nt,(si+1)*nt)
        residual[sl]=(raw if args.direct else raw-(u[sl]@localv.T+m[sl,None]))*weight[sl,None]
        if args.relative_power:
            relative_sum+=np.sum((1000/np.maximum(raw+gap,200))**args.relative_power,axis=0)
        assert path.stat().st_mtime_ns==st.st_mtime_ns
        stats.append(dict(source=int(pid),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
        if (si+1)%16==0:print('local generated',si+1,'elapsed',round(time.perf_counter()-started,1),flush=True)
    column_weight=np.sqrt(relative_sum/rows).astype(np.float32) if args.relative_power else np.ones(cols,np.float32)
    column_weight/=np.sqrt(np.mean(column_weight**2))
    for lo in range(0,rows,1024):residual[lo:lo+1024]*=column_weight
    residual.flush();width=args.rank+24;rng=np.random.default_rng(20261026)
    q,_=np.linalg.qr(rng.standard_normal((cols,width),dtype=np.float32))
    for iteration in range(3):
        left=np.empty((rows,width),np.float32)
        for lo in range(0,rows,1024):left[lo:lo+1024]=residual[lo:lo+1024]@q
        left,_=np.linalg.qr(left)
        z=np.zeros((cols,width),np.float32)
        for lo in range(0,rows,1024):z+=residual[lo:lo+1024].T@left[lo:lo+1024]
        q,_=np.linalg.qr(z)
        print('local power',iteration+1,'elapsed',round(time.perf_counter()-started,1),flush=True)
    projected=np.zeros((width,cols),np.float32)
    for lo in range(0,rows,1024):projected+=left[lo:lo+1024].T@residual[lo:lo+1024]
    _,singular,basis=np.linalg.svd(projected,full_matrices=False);basis=basis[:args.rank]
    physical=basis/column_weight
    vs=np.maximum(np.max(np.abs(physical),axis=1)/32767,np.finfo(np.float32).tiny)
    vq=np.rint(physical/vs[:,None]).clip(-32767,32767).astype('<i2')
    decoded=vq.astype(np.float32)*vs[:,None]
    coeff=np.empty((rows,args.rank),np.float32)
    projector=decoded*column_weight**2
    # Quantized physical basis needs its weighted Gram inverse, especially for prefixes.
    gram=(projector@decoded.T).astype(np.float64)
    projector=np.linalg.solve(gram,projector).astype(np.float32)
    for lo in range(0,rows,1024):coeff[lo:lo+1024]=((residual[lo:lo+1024]/column_weight)@projector.T)/weight[lo:lo+1024,None]
    np.savez_compressed(OUT/(args.name+'_factors.npz'),coefficients=coeff,v=vq,scales=vs,locations=loc,column_weight=column_weight)
    report=dict(name=args.name,lx=args.lx,ly=args.ly,rank=args.rank,rows=rows,columns=cols,
                port_weight=args.port_weight,public_delay_labels_used=False,geometry_training_rows_only=True,
                exclude_rectangle=[args.exclude_lx,args.exclude_ly],relative_power=args.relative_power,
                relative_weight='mean over generated pairs of inverse delay power; separable displacement proxy, not entrywise optimum',
                refit_base=args.refit_base,direct=args.direct,
                architecture_sha256=sha(args.raw/'arch.txt'),source_stats=stats,
                singular_values=singular.tolist(),factors_sha256=sha(OUT/(args.name+'_factors.npz')),
                elapsed_seconds=time.perf_counter()-started,scratch_bytes=scratch.stat().st_size)
    del residual;assert scratch.parent.resolve()==OUT.resolve();scratch.unlink()
    report['scratch_removed']=True
    (REPO_ROOT/'docs'/(args.name+'_training.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
