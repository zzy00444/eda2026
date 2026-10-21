"""Import, fingerprint and embed the externally supplied low-rank model.

Python standard library only, so packing also runs on the WSL host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

TABLES = ('oracle_vt.bin', 'oracle_U.bin', 'oracle_M.bin', 'oracle_ids.txt', 'oracle_arch.txt')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pack(raw: Path, tables: Path, output: Path, u: Path | None = None,
         m: Path | None = None, calibration: Path | None = None) -> dict:
    paths = [tables / name for name in TABLES]
    if u is not None:
        paths[1] = u
    if m is not None:
        paths[2] = m
    with paths[1].open('rb') as stream:
        n_src, n_tgt = struct.unpack('<2i', stream.read(8))
    offsets, sizes = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as target:
        with raw.open('rb') as source:
            shutil.copyfileobj(source, target)
        for path in paths:
            offsets.append(target.tell())
            sizes.append(path.stat().st_size)
            with path.open('rb') as source:
                shutil.copyfileobj(source, target)
        data = calibration.read_bytes() if calibration else struct.pack('<f', 1.) * (n_tgt * 5)
        if len(data) != n_tgt * 5 * 4:
            raise ValueError('Calibration size mismatch')
        offsets.append(target.tell())
        sizes.append(len(data))
        target.write(data)
        target.write(b'PANGDL02' + struct.pack('<6Q6Q2I', *offsets, *sizes, 6, 2))
    output.chmod(0o755)
    return {'binary_sha256': sha(output), 'binary_bytes': output.stat().st_size,
            'table_sha256': {name: sha(path) for name, path in zip(TABLES, paths)},
            'calibration': str(calibration) if calibration else 'unity',
            'sources': n_src, 'targets': n_tgt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'build/displacement/oracle')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in TABLES:
        source, target = args.source / name, args.output / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() == target.resolve():
            raise ValueError('Import destination must differ from the source')
        if not target.exists() or sha(source) != sha(target):
            shutil.copy2(source, target)
        files[name] = {'bytes': target.stat().st_size, 'sha256': sha(target)}
    record = {'origin': str(args.source.resolve()), 'files': files,
              'model_labels': 'Architecture-generated baseline, no public calibration',
              'upstream_sources': {name: sha(args.source / 'src' / name)
                                   for name in ('estimate_sub.cpp', 'exact.cpp')}}
    (args.output.parent / 'import.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
