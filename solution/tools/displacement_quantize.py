"""Quantize U with one scale per SVD component; decode once during loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from displacement_assets import sha
from project_paths import BUILD, REPO_ROOT


def quantize(source, output, bits):
    with source.open('rb') as stream:
        header = np.fromfile(stream, dtype='<i4', count=6)
        assert header[5] == 0
        raw = np.fromfile(stream, dtype='<f4').reshape(-1, int(header[2]))
    limit = (1 << (bits - 1)) - 1
    scales = np.maximum(np.abs(raw).max(axis=0) / limit, np.finfo(np.float32).tiny)
    dtype = '<i2' if bits == 16 else 'i1'
    quantized = np.rint(raw / scales).clip(-limit, limit).astype(dtype)
    restored = quantized.astype(np.float32) * scales
    header[5] = bits
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        header.tofile(stream)
        scales.astype('<f4').tofile(stream)
        quantized.tofile(stream)
    return {'bits': bits, 'scaling': 'per-component maxabs, float32 scale',
            'source_sha256': sha(source), 'output_sha256': sha(output),
            'bytes': output.stat().st_size,
            'raw_bytes_saved': source.stat().st_size - output.stat().st_size,
            'U_max_abs_reconstruction_error': float(np.abs(raw - restored).max()),
            'U_mean_abs_reconstruction_error': float(np.abs(raw - restored).mean())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BUILD / 'displacement/oracle/oracle_U.bin')
    parser.add_argument('--bits', type=int, choices=[8, 16], default=16)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or BUILD / f'displacement/U_q{args.bits}.bin'
    record = quantize(args.source, target, args.bits)
    report_name = f'{target.stem}_quantization' if args.output else f'displacement_quantization_{args.bits}'
    (REPO_ROOT / f'docs/{report_name}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
