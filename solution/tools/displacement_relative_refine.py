"""Refine a fixed-rank local model with entrywise generated relative error weights.

Alternating preconditioned gradient directions use exact quadratic line searches
for every row/column. All weights come from generated distances and train geometry.
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


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('factors');ap.add_argument('--name',required=True)
    ap.add_argument('--rank',type=int,required=True);ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--raw',type=Path,default=Path('F:/EDA/pango_delay_handover_v3/build'))
    args=ap.parse_args();started=time.perf_counter();path=OUT/(args.factors+'_factors.npz');f=np.load(path)
    h,sc,v,uh,_,_,ids=load_basis();u,_,m=load_base();loc=f['locations'];rows=len(u);cols=len(loc);rank=args.rank
    assert sha(args.raw/'arch.txt')==sha(OUT/'oracle/oracle_arch.txt') and rank<=f['coefficients'].shape[1]
    g=np.load(OUT/'geometry.npz');tr=indices(g)[0]
    counts=np.bincount(g['si'][tr]*288+g['ti'][tr],minlength=rows)
    port=np.sqrt((counts+2)/(counts.mean()+2)).astype(np.float32)
    a=f['coefficients'][:,:rank].copy()*port[:,None]
    b=f['v'][:rank].astype(np.float32)*f['scales'][:rank,None]
    scratch=[OUT/(args.name+'_residual_scratch.npy'),OUT/(args.name+'_weight_scratch.npy')]
    residual=np.lib.format.open_memmap(scratch[0],mode='w+',dtype=np.float32,shape=(rows,cols))
    weights=np.lib.format.open_memmap(scratch[1],mode='w+',dtype=np.float32,shape=(rows,cols))
    gap=2.5*np.abs(loc//int(h[3])+int(h[4]))+.08*np.abs(loc%int(h[3])+int(h[5]))
    lv=v[loc].astype(np.float32);stats=[]
    for si,pid in enumerate(ids[:208]):
        p=args.raw/f'all_C_{pid}.bin';st=p.stat();assert st.st_size==288*int(h[1])*2
        raw=np.memmap(p,'<i2','r',shape=(288,int(h[1])))[:,loc].astype(np.float32);sl=slice(si*288,(si+1)*288)
        residual[sl]=(raw-(u[sl]@lv.T+m[sl,None]))*port[sl,None]
        weights[sl]=(1000/np.maximum(raw+gap,200))**2
        assert p.stat().st_mtime_ns==st.st_mtime_ns
        stats.append(dict(source=int(pid),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
        if (si+1)%32==0:print('weighted generated',si+1,'elapsed',round(time.perf_counter()-started,1),flush=True)
    residual.flush();weights.flush();cw=np.zeros(cols,np.float64);rw=np.empty(rows,np.float32)
    for lo in range(0,rows,512):
        w=weights[lo:lo+512];cw+=w.sum(0,dtype=np.float64);rw[lo:lo+len(w)]=w.mean(1)
    cw=(cw/rows).astype(np.float32);history=[]
    for step in range(args.steps+1):
        loss=0.;grad=np.zeros_like(b)
        if step:
            gram=(b*cw)@b.T;ridge=max(float(np.trace(gram)/rank)*1e-6,1e-8)
            inv=np.linalg.inv(gram.astype(np.float64)+np.eye(rank)*ridge).astype(np.float32)
        for lo in range(0,rows,512):
            aa=a[lo:lo+512];r=residual[lo:lo+len(aa)];w=weights[lo:lo+len(aa)]
            error=aa@b-r
            if step:
                direction=-((error*w)@b.T)@inv;change=direction@b
                alpha=-np.sum(error*change*w,axis=1)/np.maximum(np.sum(change*change*w,axis=1),1e-20)
                aa+=alpha[:,None]*direction;error=aa@b-r
                grad+=aa.T@(error*w)
            loss+=float(np.sum(error.astype(np.float64)**2*w))
        if step:
            gram=(a.T*rw)@a;ridge=max(float(np.trace(gram)/rank)*1e-6,1e-8)
            direction=-np.linalg.solve(gram.astype(np.float64)+np.eye(rank)*ridge,grad).astype(np.float32)
            numerator=np.zeros(cols,np.float64);denominator=numerator.copy()
            for lo in range(0,rows,512):
                aa=a[lo:lo+512];r=residual[lo:lo+len(aa)];w=weights[lo:lo+len(aa)]
                error=aa@b-r;change=aa@direction
                numerator-=np.sum(error*change*w,axis=0,dtype=np.float64)
                denominator+=np.sum(change*change*w,axis=0,dtype=np.float64)
            alpha=(numerator/np.maximum(denominator,1e-20)).astype(np.float32)
            b+=direction*alpha
            loss_after=0.
            for lo in range(0,rows,512):
                error=a[lo:lo+512]@b-residual[lo:lo+512]
                loss_after+=float(np.sum(error.astype(np.float64)**2*weights[lo:lo+512]))
            loss=loss_after
        value=dict(step=step,relative_squared_loss=loss/(rows*cols),elapsed=time.perf_counter()-started)
        history.append(value);print(value,flush=True)
        (OUT/(args.name+'_progress.json')).write_text(json.dumps(history,indent=2)+'\n')
        if step:assert loss<=history[-2]['relative_squared_loss']*(rows*cols)*1.00001
    # QR/SVD reorders the factors without changing their rank-r product.
    qa,ra=np.linalg.qr(a);qb,rb=np.linalg.qr((b*np.sqrt(cw)).T)
    left,singular,right=np.linalg.svd(ra@rb.T,full_matrices=False)
    a=(qa@left)*singular;b=(right@qb.T)/np.sqrt(cw)
    vs=np.maximum(np.max(np.abs(b),axis=1)/32767,np.finfo(np.float32).tiny)
    vq=np.rint(b/vs[:,None]).clip(-32767,32767).astype('<i2')
    target=OUT/(args.name+'_factors.npz')
    np.savez_compressed(target,coefficients=(a/port[:,None]).astype(np.float32),v=vq,scales=vs,locations=loc)
    report=dict(name=args.name,initial_factors=args.factors,initial_sha256=sha(path),factors_sha256=sha(target),
                rank=rank,rows=rows,columns=cols,public_delay_labels_used=False,geometry_training_rows_only=True,
                entrywise_relative_weight='train pair frequency * residual^2 / max(generated_distance+gap_proxy,200)^2',
                gap_proxy='2.5*abs(dx)+0.08*abs(dy)',refit_base=True,history=history,
                architecture_sha256=sha(args.raw/'arch.txt'),source_stats=stats,steps=args.steps,
                elapsed_seconds=time.perf_counter()-started)
    # Explicitly close maps: the last block views can otherwise keep Windows handles alive.
    residual._mmap.close();weights._mmap.close()
    del residual,weights
    for p in scratch:assert p.parent.resolve()==OUT.resolve();p.unlink()
    report['scratch_removed']=True
    (REPO_ROOT/'docs'/(args.name+'_training.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
