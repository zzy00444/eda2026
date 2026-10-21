"""Freeze a quantized prefix of the architecture-generated local factors."""
import argparse
import json
import struct
import numpy as np
from displacement_analyze import OUT
from displacement_assets import sha
from displacement_local_screen import quantize
from project_paths import REPO_ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name')
    ap.add_argument('--factors', default='stage10_local_weighted')
    ap.add_argument('--rank', type=int, default=96)
    ap.add_argument('--bits', type=int, choices=(8, 16), default=8)
    args = ap.parse_args()
    path = OUT/(args.factors+'_factors.npz')
    f = np.load(path)
    assert 0 < args.rank <= f['coefficients'].shape[1]
    uq, us = quantize(f['coefficients'][:, :args.rank], args.bits)
    vs = f['scales'][:args.rank]
    v = f['v'][:args.rank].T.copy()
    mapping = np.full(266305, -1, '<i4')
    mapping[f['locations']] = np.arange(len(v))
    target = OUT/(args.name+'.local')
    with target.open('wb') as stream:
        stream.write(struct.pack('<8s6I', b'PGLOC001', len(uq), args.rank, len(v), len(mapping), args.bits, 0))
        for array in (mapping, us.astype('<f4'), vs.astype('<f4'), uq, v.astype('<i2')):
            array.tofile(stream)
    report = dict(candidate=args.name, factors=args.factors, factors_sha256=sha(path),
                  rank=args.rank, bits=args.bits, columns=len(v), pairs=len(uq),
                  asset_sha256=sha(target), asset_bytes=target.stat().st_size,
                  correction='round(local dot) added to round(base M+dot), before branch minimum and calibration',
                  distance_gate='V row belongs to frozen local rectangle; both candidates evaluated',
                  public_delay_labels_used=False)
    (REPO_ROOT/'docs'/(args.name+'_local_export.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
