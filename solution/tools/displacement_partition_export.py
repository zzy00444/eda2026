"""Export disjoint displacement regions with individual quantized ranks."""
import argparse
import json
import struct
import numpy as np
from displacement_analyze import OUT
from displacement_assets import sha
from displacement_local_screen import quantize
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('name')
    ap.add_argument('--part',action='append',required=True,help='factor_prefix:rank[:bits]')
    args=ap.parse_args();mapping=np.full(266305,-1,'<i4');parts=[];offset=0;records=[]
    for spec in args.part:
        tokens=spec.split(':');name=tokens[0];rank=int(tokens[1]);bits=int(tokens[2]) if len(tokens)>2 else 8
        assert bits in (8,16) and 0<rank<=512
        path=OUT/(name+'_factors.npz');f=np.load(path)
        assert rank<=f['coefficients'].shape[1] and np.all(mapping[f['locations']]<0)
        uq,us=quantize(f['coefficients'][:,:rank],bits);vs=f['scales'][:rank];v=f['v'][:rank].T.copy()
        mapping[f['locations']]=np.arange(len(v))+offset;offset+=len(v)
        parts.append((rank,len(v),bits,us,vs,uq,v))
        records.append(dict(factors=name,factors_sha256=sha(path),rank=rank,bits=bits,columns=len(v)))
    target=OUT/(args.name+'.local')
    with target.open('wb') as stream:
        stream.write(struct.pack('<8s6I',b'PGLOC002',59904,len(parts),offset,len(mapping),0,0));mapping.tofile(stream)
        for rank,cols,bits,us,vs,uq,v in parts:
            stream.write(struct.pack('<4I',rank,cols,bits,0))
            for arr in (us.astype('<f4'),vs.astype('<f4'),uq,v.astype('<i2')):arr.tofile(stream)
    report=dict(candidate=args.name,format='PGLOC002',parts=records,asset_sha256=sha(target),
                asset_bytes=target.stat().st_size,public_delay_labels_used=False,
                correction='one disjoint region per branch; separately round local and baseline dots before min')
    (REPO_ROOT/'docs'/(args.name+'_local_export.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
