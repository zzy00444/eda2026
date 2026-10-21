"""Build and execute isolated displacement candidates through WSL Docker.

Run with WSL Python; training/scoring use the Windows project virtualenv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

from displacement_assets import pack, sha

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / 'solution/build/displacement'
WORK = Path.home() / 'eda2026-displacement'
IMAGE = 'fpga-ziguang:latest'


def docker(command, *, readonly_repo=False):
    args = ['docker', 'run', '--rm', '--cpus=1', '--memory=2000000000',
            '--memory-swap=2000000000', '--user', f'{os.getuid()}:{os.getgid()}',
            '--mount', f'type=bind,source={WORK},target=/work', '-w', '/work']
    if readonly_repo:
        args += ['--mount', f'type=bind,source={REPO},target=/workspace,readonly']
    subprocess.run(args + [IMAGE] + list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('build')
    build.add_argument('name')
    build.add_argument('--u', type=Path)
    build.add_argument('--m', type=Path)
    build.add_argument('--calibration', type=Path)
    build.add_argument('--define', action='append', default=[])
    build.add_argument('--residual-header', type=Path)
    build.add_argument('--bound-file', type=Path)
    build.add_argument('--local-file', type=Path)
    build.add_argument('--wall-header', type=Path)
    run = sub.add_parser('run')
    run.add_argument('names', nargs='+')
    run.add_argument('--request', type=Path, default=REPO / 'solution/build/local_runs/request_1m.csv')
    run.add_argument('--trace', action='store_true')
    run.add_argument('--suffix', default='result')
    oracle = sub.add_parser('oracle')
    oracle.add_argument('--queries', type=Path, required=True)
    bench = sub.add_parser('benchmark')
    bench.add_argument('names', nargs='+')
    bench.add_argument('--runs', type=int, default=5)
    bench.add_argument('--tag', default='displacement_final')
    validate = sub.add_parser('validate')
    validate.add_argument('name')
    clean = sub.add_parser('clean')
    clean.add_argument('names', nargs='+')
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    if args.action == 'build':
        if not args.name.replace('_', '').isalnum():
            raise ValueError('Use an alphanumeric candidate name')
        source = REPO / 'solution/src/displacement.cpp'
        shutil.copy2(source, WORK / 'displacement.cpp')
        raw = WORK / (args.name + '.raw')
        flags = ['-std=c++17', '-O3', '-DNDEBUG'] + ['-D' + item for item in args.define]
        libraries = []
        if args.wall_header:
            if args.residual_header:
                raise ValueError('Wall-only and full residual runtimes are alternative builds')
            shutil.copy2(args.wall_header, WORK / 'displacement_wall_model.hpp')
            shutil.copy2(REPO / 'solution/src/displacement_wall_runtime.hpp', WORK / 'displacement_wall_runtime.hpp')
            flags += ['-DEST_WALL']
        if args.local_file:
            if args.residual_header:
                raise ValueError('Local and envelope descriptors cannot share the same slot')
            shutil.copy2(REPO / 'solution/src/displacement_local_runtime.hpp', WORK / 'displacement_local_runtime.hpp')
            flags += ['-DEST_LOCAL']
        if args.residual_header:
            if not args.bound_file:
                raise ValueError('Residual build requires --bound-file')
            shutil.copy2(args.residual_header, WORK / 'displacement_residual_model.hpp')
            shutil.copy2(REPO / 'solution/src/displacement_residual_runtime.hpp', WORK / 'displacement_residual_runtime.hpp')
            flags += ['-DEST_RESIDUAL']
            libraries += ['-lz']
        docker(['g++', *flags, '/work/displacement.cpp', *libraries, '-o', '/work/' + raw.name])
        if args.local_file:
            with raw.open('ab') as target:
                begin = target.tell()
                local = args.local_file.read_bytes()
                target.write(local)
                target.write(b'PGLCF001'+struct.pack('<3Q', begin, len(local), 0))
        if args.residual_header:
            with raw.open('ab') as target:
                begin = target.tell()
                compressed = args.bound_file.read_bytes()
                encoding = json.loads((REPO / 'docs/stage9_bound_encoding.json').read_text())
                assert sha(args.bound_file) == encoding['compressed_sha256']
                target.write(compressed)
                target.write(b'PGBND001'+struct.pack('<3Q',begin,len(compressed),encoding['decoded_stream_bytes']))
        binary = WORK / args.name
        record = pack(raw, BUILD / 'oracle', binary, args.u, args.m, args.calibration)
        record.update(source_sha256=sha(source), compiler_flags=flags, image=IMAGE)
        if args.wall_header:
            record.update(wall_header_sha256=sha(args.wall_header),
                          wall_runtime_sha256=sha(REPO / 'solution/src/displacement_wall_runtime.hpp'))
        if args.local_file:
            record.update(local_asset_sha256=sha(args.local_file),
                          local_runtime_sha256=sha(REPO / 'solution/src/displacement_local_runtime.hpp'))
        if args.residual_header:
            record.update(residual_header_sha256=sha(args.residual_header), bound_sha256=sha(args.bound_file),
                          residual_runtime_sha256=sha(REPO / 'solution/src/displacement_residual_runtime.hpp'), libraries=libraries)
        target = REPO / 'solution/bin' / ('estimate_' + args.name)
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(binary, target)
        (BUILD / (args.name + '_build.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2), flush=True)
    elif args.action == 'run':
        request = WORK / 'request.csv'
        shutil.copy2(args.request, request)
        for name in args.names:
            binary = WORK / name
            if not binary.exists():
                shutil.copy2(REPO / 'solution/bin' / ('estimate_' + name), binary)
                binary.chmod(0o755)
            output = name + '_' + args.suffix + '.csv'
            command = ['/work/' + name, '-in', '/work/request.csv', '-out', '/work/' + output]
            if args.trace:
                command += ['-trace', '/work/' + name + '_trace.bin']
            docker(command)
            shutil.copy2(WORK / output, BUILD / output)
            if args.trace:
                shutil.copy2(WORK / (name + '_trace.bin'), BUILD / (name + '_trace.bin'))
            print(name, 'result_sha256', sha(WORK / output), flush=True)
    elif args.action == 'oracle':
        shutil.copy2(REPO / 'solution/src/displacement_oracle.cpp', WORK / 'oracle.cpp')
        docker(['g++', '-std=c++17', '-O3', '-DNDEBUG', '/work/oracle.cpp', '-o', '/work/oracle'])
        shutil.copy2(BUILD / 'oracle/oracle_arch.txt', WORK / 'arch.txt')
        shutil.copy2(args.queries, WORK / 'oracle_queries.csv')
        # Save exact solver stdout for scoring; the source reports mismatches on stderr.
        with (BUILD / 'oracle_result.csv').open('w') as output:
            command = ['docker', 'run', '--rm', '--cpus=1', '--memory=2000000000',
                       '--memory-swap=2000000000', '--mount', f'type=bind,source={WORK},target=/work',
                       IMAGE, '/work/oracle', '/work/arch.txt', '/work/oracle_queries.csv']
            subprocess.run(command, stdout=output, check=True)
    elif args.action == 'benchmark':
        if args.runs < 1:
            raise ValueError('At least one run is required')
        request = WORK / 'request.csv'
        shutil.copy2(REPO / 'solution/build/local_runs/request_1m.csv', request)
        expanded = WORK / 'request_10m.csv'
        with request.open('rb') as source, expanded.open('wb') as target:
            header = source.readline()
            start = source.tell()
            target.write(header)
            for _ in range(10):
                source.seek(start)
                shutil.copyfileobj(source, target, 4 << 20)
        expected, data = {}, {}
        try:
            for name in args.names:
                binary = WORK / name
                original = REPO / 'solution/bin' / ('estimate_' + name)
                if not binary.exists() or sha(binary) != sha(original):
                    shutil.copy2(original, binary)
                    binary.chmod(0o755)
                reference = WORK / (name + '_result.csv')
                # Always regenerate the reference for this exact binary.
                docker(['/work/' + name, '-in', '/work/request.csv', '-out', '/work/' + reference.name])
                digest = hashlib.sha256()
                with reference.open('rb') as stream:
                    digest.update(stream.readline())
                    begin = stream.tell()
                    for _ in range(10):
                        stream.seek(begin)
                        for block in iter(lambda: stream.read(4 << 20), b''):
                            digest.update(block)
                expected[name] = digest.hexdigest()
                data[name] = {'binary_sha256': sha(binary), 'reference_1m_sha256': sha(reference), 'runs': []}
            for iteration in range(args.runs):
                for name in (args.names if iteration % 2 == 0 else args.names[::-1]):
                    report = WORK / 'benchmark_once.json'
                    output = WORK / 'benchmark_output.csv'
                    docker(['python3', '/workspace/solution/tools/benchmark.py',
                            '--input', '/work/request_10m.csv', '--output', '/work/' + output.name,
                            '--report', '/work/' + report.name, '--executable', '/work/' + name],
                           readonly_repo=True)
                    item = json.loads(report.read_text())[0]
                    assert item['output_sha256'] == expected[name], name + ' repeated output mismatch'
                    assert item['max_observed_threads'] == 1
                    assert item['container_peak_bytes'] < 2_000_000_000
                    item.update(run=iteration+1, rows=10_000_000, repeated_output_verified=True)
                    data[name]['runs'].append(item)
                    output.unlink()
                    print(name, iteration+1, item['cpu_seconds'], flush=True)
            for item in data.values():
                item['mean_10m_cpu_seconds'] = sum(run['cpu_seconds'] for run in item['runs']) / args.runs
                item['projected_1b_cpu_seconds'] = item['mean_10m_cpu_seconds'] * 100
                item['max_container_bytes'] = max(run['container_peak_bytes'] for run in item['runs'])
            result = {'image': IMAGE, 'cpu_limit': 1, 'memory_limit_bytes': 2_000_000_000,
                      'runs': args.runs, 'timer': 'child user+sys; fresh container per candidate/run',
                      'order': 'alternating forward/reverse', 'rows': 10_000_000,
                      'cpu_model': next(line.split(':', 1)[1].strip() for line in
                                        Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                      'official_score': False, 'full_1b_completed': False, 'candidates': data}
            (REPO / 'docs' / (args.tag + '_benchmark.json')).write_text(json.dumps(result, indent=2) + '\n')
        finally:
            expanded.unlink(missing_ok=True)
            (WORK / 'benchmark_output.csv').unlink(missing_ok=True)
    elif args.action == 'validate':
        name = args.name
        binary = REPO / 'solution/bin' / ('estimate_' + name)
        candidate = WORK / name
        if not candidate.exists() or sha(candidate) != sha(binary):
            shutil.copy2(binary, candidate)
            candidate.chmod(0o755)
        binary_sha = sha(candidate)
        package = REPO / 'solution' / (name + '_submission.zip')
        package_sha = sha(package)
        docker(['python3', '/workspace/solution/tools/integration.py', '--executable', '/work/' + name,
                '--report', '/work/' + name + '_integration.json'], readonly_repo=True)
        docker(['python3', '/workspace/solution/tools/displacement_contract.py', '--executable', '/work/' + name,
                '--report', '/work/' + name + '_contract.json'], readonly_repo=True)
        docker(['python3', '/workspace/solution/tools/standalone.py',
                '--package', '/workspace/solution/' + name + '_submission.zip',
                '--reference-executable', '/work/' + name,
                '--report', '/work/' + name + '_standalone.json'], readonly_repo=True)
        for suffix in ('integration', 'contract', 'standalone'):
            shutil.copy2(WORK / (name + '_' + suffix + '.json'),
                         REPO / 'docs' / (name + '_' + suffix + '.json'))
        assert sha(candidate) == sha(binary) == binary_sha
        assert sha(package) == package_sha
        record = {'candidate': name, 'binary_sha256': binary_sha, 'zip_sha256': package_sha,
                  'reports': {suffix: sha(REPO / 'docs' / (name + '_' + suffix + '.json'))
                              for suffix in ('integration', 'contract', 'standalone')},
                  'image': IMAGE, 'memory_limit_bytes': 2_000_000_000, 'all_passed': True}
        (REPO / 'docs' / (name + '_validation.json')).write_text(json.dumps(record, indent=2) + '\n')
    elif args.action == 'clean':
        for name in args.names:
            if not name.replace('_', '').isalnum():
                raise ValueError('Invalid candidate name')
            targets = [WORK / name, WORK / (name + '.raw'), WORK / (name + '_result.csv'),
                       REPO / 'solution/bin' / ('estimate_' + name), BUILD / (name + '_result.csv')]
            for path in targets:
                if path.parent not in (WORK, BUILD, REPO / 'solution/bin'):
                    raise ValueError('Unexpected cleanup directory')
                if path.is_file():
                    path.unlink()
                    print('removed', path, flush=True)


if __name__ == '__main__':
    main()
