"""Package one isolated ELF, preserving the standalone submission contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from displacement_assets import sha
from project_paths import REPO_ROOT, SOLUTION_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    binary = SOLUTION_ROOT / 'bin' / ('estimate_' + args.name)
    target = SOLUTION_ROOT / (args.name + '_submission.zip')
    with binary.open('rb') as stream:
        assert stream.read(4) == b'\x7fELF'
    info = zipfile.ZipInfo('bin/estimate', date_time=(2026, 10, 21, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100755 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(info, binary.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    assert target.stat().st_size < 100_000_000
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ['bin/estimate']
        assert archive.getinfo('bin/estimate').external_attr >> 16 == 0o100755
        import hashlib
        assert hashlib.sha256(archive.read('bin/estimate')).hexdigest() == sha(binary)
    record = {'candidate': args.name, 'binary_bytes': binary.stat().st_size,
              'zip_bytes': target.stat().st_size, 'binary_sha256': sha(binary),
              'zip_sha256': sha(target), 'contents': ['bin/estimate'], 'unix_mode': '100755'}
    (REPO_ROOT / 'docs' / (args.name + '_package.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
