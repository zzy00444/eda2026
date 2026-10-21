"""Build and execute isolated displacement candidates through WSL Docker.

Run with WSL Python; training/scoring use the Windows project virtualenv.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
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
    run = sub.add_parser('run')
    run.add_argument('names', nargs='+')
    run.add_argument('--request', type=Path, default=REPO / 'solution/build/local_runs/request_1m.csv')
    oracle = sub.add_parser('oracle')
    oracle.add_argument('--queries', type=Path, required=True)
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
        docker(['g++', *flags, '/work/displacement.cpp', '-o', '/work/' + raw.name])
        binary = WORK / args.name
        record = pack(raw, BUILD / 'oracle', binary, args.u, args.m, args.calibration)
        record.update(source_sha256=sha(source), compiler_flags=flags, image=IMAGE)
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
            output = name + '_result.csv'
            docker(['/work/' + name, '-in', '/work/request.csv', '-out', '/work/' + output])
            shutil.copy2(WORK / output, BUILD / output)
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


if __name__ == '__main__':
    main()
