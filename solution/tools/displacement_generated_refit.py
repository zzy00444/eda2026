"""Refit the existing basis from architecture-generated distances, without public labels.

Reads the external project's completed source tables without modifying them.
Only small coefficients and configuration evidence are saved in this repository.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
for _key in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='6'
import numpy as np
from displacement_analyze import OUT, points
from displacement_assets import sha
from project_paths import REPO_ROOT


def load_basis():
    with (OUT/'oracle/oracle_vt.bin').open('rb') as f:
        h=np.fromfile(f,'<i4',6);s=np.fromfile(f,'<f4',int(h[0]))
    v=np.memmap(OUT/'oracle/oracle_vt.bin','<i2','r',offset=24+4*int(h[0]),shape=(int(h[1]),int(h[0])))
    with (OUT/'oracle/oracle_U.bin').open('rb') as f:
        uh=np.fromfile(f,'<i4',6);u=np.fromfile(f,'<f4').reshape(-1,int(h[0]))
    m=np.fromfile(OUT/'oracle/oracle_M.bin','<i2',offset=12).astype(np.float32)
    ids=np.loadtxt(OUT/'oracle/oracle_ids.txt',dtype=np.int32)
    return h,s,v,uh,u,m,ids


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',type=Path,default=Path('F:/EDA/pango_delay_handover_v3/build'))
    ap.add_argument('--samples',type=int,default=4096)
    ap.add_argument('--fit',action='store_true')
    args=ap.parse_args()
    h,s,v,uh,u,m,ids=load_basis();ns,nt,rank=map(int,uh[:3]);nc=int(h[1])
    assert sha(args.raw/'arch.txt')==sha(OUT/'oracle/oracle_arch.txt'),'Architecture mismatch'
    for pid in ids[:ns]:assert (args.raw/f'all_C_{pid}.bin').stat().st_size==nt*nc*2
    rng=np.random.default_rng(20261024);sample=rng.choice(nc,args.samples+2048,replace=False)
    ti,vi=np.sort(sample[:args.samples]),np.sort(sample[args.samples:])
    norms=np.maximum(np.sqrt(np.mean((v.astype(np.float32)*s)**2,axis=0)),1e-8)
    xt=np.column_stack((v[ti].astype(np.float32)*s/norms,np.ones(len(ti),np.float32)))
    xv=np.column_stack((v[vi].astype(np.float32)*s/norms,np.ones(len(vi),np.float32)))
    coeff=np.column_stack((u*norms,m));pilot=np.unique(np.linspace(0,ns-1,12,dtype=int))
    def raw(si,ix):
        return np.memmap(args.raw/f'all_C_{ids[si]}.bin','<i2','r',shape=(nt,nc))[:,ix].astype(np.float32)
    started=time.perf_counter();dt=np.concatenate([raw(si,ti) for si in pilot]);dv=np.concatenate([raw(si,vi) for si in pilot]);bc=np.concatenate([coeff[si*nt:(si+1)*nt] for si in pilot])
    gap=lambda ix:2.5*np.abs(ix//int(h[3])+int(h[4]))+.08*np.abs(ix%int(h[3])+int(h[5]))
    gv=gap(vi);proxy=np.maximum(dt.mean(0)+gap(ti),200)
    def score(p):return float((100*(1-np.tanh(4*np.abs(p-dv)/np.maximum(dv+gv,1)))).mean())
    baseline=score(bc@xv.T);print('baseline generated',baseline,flush=True)
    curve=[];best=None
    for power in [0,1,2]:
        weights=(np.median(proxy)/proxy)**power;weights/=weights.mean()
        gram=(xt.T*weights)@xt/len(xt)
        rhs=((dt-bc@xt.T)*weights)@xt/len(xt)
        for ridge in [.0001,.01,.1,1.]:
            penalty=np.eye(rank+1)*ridge;penalty[-1,-1]*=.1
            inv=np.linalg.inv(gram.astype(np.float64)+penalty).astype(np.float32)
            pred=(bc+rhs@inv)@xv.T
            sc=score(pred);row=dict(power=power,ridge=ridge,generated_accuracy=sc,gain=sc-baseline)
            print(json.dumps(row),flush=True);curve.append(row)
            if best is None or sc>best[0]:best=(sc,power,ridge,(xt*weights[:,None])@inv/len(xt))
    report=dict(baseline_generated=baseline,curve=curve,selected=dict(power=best[1],ridge=best[2]),
                samples=args.samples,validation_displacements=len(vi),pilot_sources=ids[pilot].tolist(),
                architecture_sha256=sha(args.raw/'arch.txt'),public_answers_used=False,
                external_source=str(args.raw),source_tables_modified=False)
    if args.fit:
        output=coeff.copy();projector=best[3];stats=[]
        for si in range(ns):
            path=args.raw/f'all_C_{ids[si]}.bin';before=path.stat()
            data=raw(si,ti);sl=slice(si*nt,(si+1)*nt)
            output[sl]+=(data-coeff[sl]@xt.T)@projector
            after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            stats.append(dict(source=int(ids[si]),bytes=before.st_size,mtime_ns=before.st_mtime_ns))
            if (si+1)%16==0:print('fit',si+1,'elapsed',round(time.perf_counter()-started,1),flush=True)
        newu=(output[:,:rank]/norms).astype('<f4')
        limit=32767.;us=np.maximum(np.max(np.abs(newu),axis=0)/limit,np.finfo(np.float32).tiny)
        uq=np.rint(newu/us).clip(-32767,32767).astype('<i2');uh[5]=16
        with (OUT/'stage10_generated_U.bin').open('wb') as f:uh.tofile(f);us.astype('<f4').tofile(f);uq.tofile(f)
        nm=np.floor(output[:,-1]+.5);assert np.max(np.abs(nm))<32767
        with (OUT/'stage10_generated_M.bin').open('wb') as f:np.array([ns,nt,0],'<i4').tofile(f);nm.astype('<i2').tofile(f)
        report.update(U_sha256=sha(OUT/'stage10_generated_U.bin'),M_sha256=sha(OUT/'stage10_generated_M.bin'),source_stats=stats)
    report['elapsed_seconds']=time.perf_counter()-started
    (REPO_ROOT/'docs/stage10_generated_refit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
