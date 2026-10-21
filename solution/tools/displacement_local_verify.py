"""Reconstruct both branches, float32 FMA dots, local factors and wall residual.

The reference reads the frozen quantized asset, never the deployed base output.
It also checks the unchanged base against the stage-8 public regression output.
"""
import os
for _var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_var]='2'
import argparse
import json
import struct
import lightgbm as lgb
import numpy as np
import pandas as pd
from displacement_analyze import OUT, describe, points
from displacement_assets import sha
from displacement_residual import architecture, make_features, indices
from displacement_generalization import mean_delta_interval
from experiment import pin_parts
from project_paths import REPO_ROOT


def round_away(a):
    return np.where(a>=0,np.floor(a+.5),np.ceil(a-.5)).astype(np.int64)


def fma_dot(u,v):
    """Exactly reproduce dotAvx2: double intermediates represent each FMA."""
    n,rank=u.shape;a=np.zeros((n,8),np.float32);b=a.copy();r=0
    while r+16<=rank:
        a=(u[:,r:r+8].astype(np.float64)*v[:,r:r+8]+a).astype(np.float32)
        b=(u[:,r+8:r+16].astype(np.float64)*v[:,r+8:r+16]+b).astype(np.float32)
        r+=16
    while r+8<=rank:
        a=(u[:,r:r+8].astype(np.float64)*v[:,r:r+8]+a).astype(np.float32);r+=8
    a=a+b;s=a[:,:4]+a[:,4:];s=s[:,:2]+s[:,2:];out=s[:,0]+s[:,1]
    while r<rank:
        out=out+(u[:,r]*v[:,r]).astype(np.float32);r+=1
    return out


def load_base():
    with (OUT/'U_refit_q16.bin').open('rb') as stream:
        h=np.fromfile(stream,'<i4',6);sc=np.fromfile(stream,'<f4',int(h[2]))
        u=np.fromfile(stream,'<i2').reshape(-1,int(h[2])).astype(np.float32)*sc
    with (OUT/'oracle/oracle_vt.bin').open('rb') as stream:
        vh=np.fromfile(stream,'<i4',6);sc=np.fromfile(stream,'<f4',int(vh[0]))
    u*=sc
    v=np.memmap(OUT/'oracle/oracle_vt.bin','<i2','r',offset=24+4*int(vh[0]),shape=(int(vh[1]),int(vh[0])))
    m=np.fromfile(OUT/'M_refit.bin','<i2',offset=12)
    return u,v,m


def load_local(path):
    with path.open('rb') as stream:
        magic,rows,rank,cols,nc,bits,reserved=struct.unpack('<8s6I',stream.read(32))
        if magic in (b'PGLOC002',b'PGLOC003',b'PGLOC004',b'PGLOC005'):
            assert bits in (0,1) and not reserved
            mapping=np.fromfile(stream,'<i4',nc);parts=[];begin=0
            for _ in range(rank):
                pr,pc,pb,stored=struct.unpack('<4I',stream.read(16))
                direct=bool(pr>>31) if magic==b'PGLOC005' else magic==b'PGLOC004'
                pr&=0x7fffffff
                pmapping=np.fromfile(stream,'<i4',rows) if magic!=b'PGLOC002' else None
                assert stored>0 if pmapping is not None else stored==0
                us=np.fromfile(stream,'<f4',pr);vs=np.fromfile(stream,'<f4',pr)
                uq=np.fromfile(stream,'i1' if pb==8 else '<i2',(stored or rows)*pr).reshape(stored or rows,pr)
                v=np.fromfile(stream,'<i2',pc*pr).reshape(pc,pr)
                if bits:
                    packed=v.view(np.uint8).reshape(2,pr,pc).transpose(1,2,0).copy().view('<i2').reshape(pr,pc).T.copy()
                    for col in range(12,pc):packed[col]=(packed[col].astype(np.int32)+packed[col-12]).astype('<i2')
                    v=packed
                pm=np.where((mapping>=begin)&(mapping<begin+pc),mapping-begin,-1)
                parts.append((pm,(uq.astype(np.float32)*us)*vs,v,pmapping,direct));begin+=pc
            assert begin==cols and stream.read()==b''
            return parts
        assert magic==b'PGLOC001' and not reserved
        mapping=np.fromfile(stream,'<i4',nc)
        us=np.fromfile(stream,'<f4',rank);vs=np.fromfile(stream,'<f4',rank)
        uq=np.fromfile(stream,'i1' if bits==8 else '<i2',rows*rank).reshape(rows,rank)
        u=(uq.astype(np.float32)*us)*vs
        v=np.fromfile(stream,'<i2',cols*rank).reshape(cols,rank)
        assert stream.read()==b''
    return mapping,u,v


def predict(g,base_assets,local_assets=None,*,local_bias=None,calibration=None,return_raw=False):
    u,v,m=base_assets
    _,_,_,ndx,ndy,vp,hp,dd=architecture()
    sx,sy,tx,ty,sp,tp=(g[k].astype(np.int32) for k in ('sx','sy','tx','ty','source_id','target_id'))
    si,ti=g['si'],g['ti'];pair=np.maximum(si*288+ti,0)
    dx,dy=tx-sx,ty-sy
    cross=((sx<=75)&(tx>=90))|((tx<=75)&(sx>=90))
    # A row is a door precisely when the architecture loader marks it open.
    doorpre=np.r_[0,np.cumsum(dd==0)]
    door=(doorpre[np.maximum(sy,ty)+1]-doorpre[np.minimum(sy,ty)])>0
    row0=((dx-np.where(cross,np.where(dx>0,14,-14),0)+120)*1105+dy+552)
    row1=(dx+120)*1105+dy+552
    lx=sx+ndx[sp];ly=sy+ndy[sp]
    good=(si>=0)&(ti>=0)&(lx>=0)&(lx<120)&(ly>=0)&(ly<550)
    # Invalid landings do not enter the baseline solver; clip only for safe indexing.
    lx=np.clip(lx,0,119);ly=np.clip(ly,0,549)
    netwall=(ndx[sp]!=0)&(lx>=76)&(lx<=89)&((doorpre[np.maximum(sy,ly)+1]-doorpre[np.minimum(sy,ly)])==0)
    gap=vp[np.maximum(sx,lx)]-vp[np.minimum(sx,lx)]+vp[np.maximum(lx,tx)]-vp[np.minimum(lx,tx)]
    gap+=hp[np.maximum(sy,ly)]-hp[np.minimum(sy,ly)]+hp[np.maximum(ly,ty)]-hp[np.minimum(ly,ty)]
    def branch(rows):
        out=np.empty(len(sx),np.int64)
        for start in range(0,len(sx),4096):
            end=min(len(sx),start+4096);p=pair[start:end];r=rows[start:end]
            out[start:end]=round_away(m[p].astype(np.float64)+fma_dot(u[p],v[r]).astype(np.float64))
            if local_assets is not None:
                for part in (local_assets if isinstance(local_assets,list) else [local_assets]):
                    mapping,lu,lv=part[:3];pmapping=part[3] if len(part)>3 else None
                    col=mapping[r];mask=col>=0
                    pp=p if pmapping is None else pmapping[p]
                    mask&=pp>=0
                    if not np.any(mask):continue
                    d=round_away(fma_dot(lu[pp[mask]],lv[col[mask]]).astype(np.float64))
                    if len(part)>4 and part[4]:out[start:end][mask]=d
                    else:out[start:end][mask]+=d
                    if local_bias is not None:
                        out[start:end][mask]+=local_bias[p[mask]]
        return out
    a=branch(row0)+cross*292;b=branch(row1)
    raw=np.where(cross&door,np.minimum(a,b),a)+gap+netwall*292
    raw[~good]=0;raw[(sx==tx)&(sy==ty)&(sp==tp)]=0
    if return_raw:
        return raw.astype(np.int32),good
    cal=np.fromfile(OUT/'calibration_train.bin','<f4') if calibration is None else calibration
    bins=np.searchsorted([700,1000,2000,4000],raw,side='right')
    pred=np.where(raw>0,round_away(raw.astype(np.float64)*cal[np.maximum(ti,0)*5+bins]),0)
    return np.maximum(pred,0).astype(np.int32),good


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name');ap.add_argument('--local-file',required=True)
    ap.add_argument('--wall-model',required=True);ap.add_argument('--trees',type=int,required=True)
    args=ap.parse_args();g=np.load(OUT/'geometry.npz');base_assets=load_base()
    base,_=predict(g,base_assets)
    known=np.load(OUT/'displacement_cal16_prediction.npy')
    assert np.array_equal(base,known),('stage8 reference mismatch',np.count_nonzero(base!=known))
    local_path=OUT/args.local_file
    ref,good=predict(g,base_assets,load_local(local_path))
    local_ref=ref.copy();x=make_features(g,ref);ix=np.flatnonzero((x.landing_wall.to_numpy()>0)&good)
    model_path=OUT/(args.wall_model+'.txt');model=lgb.Booster(model_file=str(model_path))
    ref[ix]=np.maximum(0,np.floor(ref[ix]+model.predict(x.iloc[ix],num_iteration=args.trees,num_threads=2)+.5)).astype(np.int32)
    actual=pd.read_csv(OUT/(args.name+'_result.csv'),usecols=['delay']).delay.to_numpy(np.int32)
    mismatch=np.flatnonzero(ref!=actual)
    print('reference mismatches',len(mismatch),mismatch[:10],flush=True)
    assert len(mismatch)==0
    np.save(OUT/(args.name+'_independent_reference.npy'),ref)
    np.save(OUT/(args.name+'_prediction.npy'),actual)
    _,va,te,held=indices(g);position=g['sx'].astype(np.int64)*550+g['sy']
    report=dict(candidate=args.name,binary_sha256=sha(REPO_ROOT/'solution/bin'/('estimate_'+args.name)),
                local_asset_sha256=sha(local_path),wall_model_sha256=sha(model_path),wall_trees=args.trees,
                output_sha256=sha(OUT/(args.name+'_result.csv')),
                reference_sha256=sha(OUT/(args.name+'_independent_reference.npy')),
                rows=len(ref),python_cpp_mismatch_count=len(mismatch),stage8_reference_mismatch_count=0,
                both_branches_reconstructed=True,float32_fma_emulated=True,
                local_only_valid=describe(g['y'][va],local_ref[va]),
                caveat='Historical regression slices; only new wall parameters and local geometry weights exclude held source coordinates',
                splits={})
    for key,ix in [('valid',va),('test',te),('position_assess',held)]:
        report['splits'][key]=dict(updated=describe(g['y'][ix],ref[ix]),baseline=describe(g['y'][ix],known[ix]),
                                  paired_delta=mean_delta_interval(points(g['y'][ix],ref[ix])-points(g['y'][ix],known[ix]),position[ix]))
    oracle_path=OUT/(args.name+'_oracle.csv')
    if oracle_path.exists():
        truth=pd.read_csv(OUT/'oracle_answers.csv');deployed=pd.read_csv(oracle_path)
        assert deployed.From.equals(truth.From) and deployed.To.equals(truth.To)
        names=architecture()[0];nm={n:i for i,n in enumerate(names)}
        ids=np.loadtxt(OUT/'oracle/oracle_ids.txt',dtype=np.int32)
        sm={int(p):i for i,p in enumerate(ids[:208])};tm={int(p):i for i,p in enumerate(ids[208:])}
        og={}
        og['sx'],og['sy'],source=pin_parts(truth.From);og['tx'],og['ty'],target=pin_parts(truth.To)
        og['source_id']=source.map(nm).to_numpy(np.int32);og['target_id']=target.map(nm).to_numpy(np.int32)
        og['si']=np.array([sm.get(int(p),-1) for p in og['source_id']]);og['ti']=np.array([tm.get(int(p),-1) for p in og['target_id']])
        op,good=predict(og,base_assets,load_local(local_path));ox=make_features(og,op)
        wi=np.flatnonzero((ox.landing_wall.to_numpy()>0)&good)
        if len(wi):
            op[wi]=np.maximum(0,np.floor(op[wi]+model.predict(ox.iloc[wi],num_iteration=args.trees,num_threads=2)+.5)).astype(np.int32)
        assert np.array_equal(op,deployed.delay.to_numpy())
        report['oracle']=dict(truth_sha256=sha(OUT/'oracle_answers.csv'),request_sha256=sha(OUT/'oracle_request.csv'),
                              spec_sha256=sha(OUT/'oracle_spec.json'),output_sha256=sha(oracle_path),python_cpp_mismatch_count=0,
                              official_eight=describe(truth.delay.to_numpy()[:8],op[:8]),reserved_48=describe(truth.delay.to_numpy()[8:],op[8:]))
    (REPO_ROOT/'docs'/(args.name+'_reference.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
