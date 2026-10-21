"""Export disjoint displacement regions with individual quantized ranks."""
import argparse
import json
import struct
import numpy as np
from displacement_analyze import OUT
from displacement_assets import sha
from displacement_local_screen import quantize
from displacement_residual import indices
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('name')
    ap.add_argument('--part',action='append',required=True,help='factor_prefix:rank[:bits]')
    ap.add_argument('--sparse-pairs',action='store_true',help='Store only pairs observed in training geometry; others retain baseline')
    ap.add_argument('--direct',action='store_true',help='Replace covered base distances; requires direct factors')
    ap.add_argument('--mixed',action='store_true',help='Allow direct and residual tiles together; mode comes from each training record')
    ap.add_argument('--delta-v',action='store_true',help='Losslessly encode V as lag-12 deltas in two byte planes')
    args=ap.parse_args();mapping=np.full(266305,-1,'<i4');parts=[];offset=0;records=[]
    pairmap=None;selected=np.arange(59904)
    if args.sparse_pairs:
        g=np.load(OUT/'geometry.npz');tr=indices(g)[0]
        selected=np.unique(g['si'][tr]*288+g['ti'][tr])
        pairmap=np.full(59904,-1,'<i4');pairmap[selected]=np.arange(len(selected))
    elif args.direct or args.mixed:pairmap=np.arange(59904,dtype='<i4')
    for spec in args.part:
        tokens=spec.split(':');name=tokens[0];rank=int(tokens[1]);bits=int(tokens[2]) if len(tokens)>2 else 8
        assert bits in (8,16) and 0<rank<=512
        path=OUT/(name+'_factors.npz');f=np.load(path)
        training=json.loads((REPO_ROOT/'docs'/(name+'_training.json')).read_text())
        direct=bool(training.get('direct',False))
        if not args.mixed:assert direct==args.direct
        assert rank<=f['coefficients'].shape[1] and np.all(mapping[f['locations']]<0)
        uq,us=quantize(f['coefficients'][selected,:rank],bits);vs=f['scales'][:rank];v=f['v'][:rank].T.copy()
        mapping[f['locations']]=np.arange(len(v))+offset;offset+=len(v)
        parts.append((rank,len(v),bits,us,vs,uq,v,direct))
        records.append(dict(factors=name,factors_sha256=sha(path),rank=rank,bits=bits,columns=len(v),stored_pairs=len(selected),direct=direct))
    target=OUT/(args.name+'.local')
    with target.open('wb') as stream:
        magic=b'PGLOC005' if args.mixed else (b'PGLOC004' if args.direct else (b'PGLOC003' if pairmap is not None else b'PGLOC002'))
        stream.write(struct.pack('<8s6I',magic,59904,len(parts),offset,len(mapping),int(args.delta_v),0));mapping.tofile(stream)
        for rank,cols,bits,us,vs,uq,v,direct in parts:
            stream.write(struct.pack('<4I',rank|((int(direct)<<31) if args.mixed else 0),cols,bits,len(selected) if pairmap is not None else 0))
            if pairmap is not None:pairmap.tofile(stream)
            for arr in (us.astype('<f4'),vs.astype('<f4'),uq):arr.tofile(stream)
            if args.delta_v:
                d=v.astype('<i2').copy();d[12:]=(v[12:].astype(np.int32)-v[:-12]).astype('<i2')
                d.T.copy().view(np.uint8).reshape(rank,cols,2).transpose(2,0,1).copy().tofile(stream)
            else:v.astype('<i2').tofile(stream)
    report=dict(candidate=args.name,format=magic.decode(),parts=records,asset_sha256=sha(target),
                asset_bytes=target.stat().st_size,public_delay_labels_used=False,
                sparse_pairs_training_geometry_only=args.sparse_pairs,
                direct=args.direct,
                mixed=args.mixed,
                delta_v=args.delta_v,
                correction='replace covered baseline before min' if args.direct else 'one disjoint region per branch; separately round local and baseline dots before min')
    (REPO_ROOT/'docs'/(args.name+'_local_export.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
