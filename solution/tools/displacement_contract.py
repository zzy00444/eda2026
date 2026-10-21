"""Regression checks for malformed input, hash collisions and buffered I/O."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    arch = Path('/workspace/solution/build/displacement/oracle/oracle_arch.txt').read_text().split()
    count = int(arch[1])
    names = [arch[3 + 3*i] for i in range(count)]
    mask = (1 << 64)-1
    keys = [int.from_bytes(name.encode().ljust(8, b'\0'), 'little') for name in names]
    seed = 0x9E3779B97F4A7C15
    for _ in range(4096):
        slots = {((key*seed)&mask) >> 48 for key in keys}
        if len(slots) == len(keys):
            break
        seed = (seed*0x2545F4914F6CDD1D + 0x9E3779B97F4A7C15)&mask
    else:
        raise AssertionError('Could not reproduce the fast hash seed')
    for i in range(100000):
        unknown = 'BAD' + str(i)
        key = int.from_bytes(unknown.encode().ljust(8, b'\0'), 'little')
        if ((key*seed)&mask) >> 48 in slots:
            break
    else:
        raise AssertionError('Could not construct a collision fixture')
    passed = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        request, output = root / 'in.csv', root / 'out.csv'
        def run(payload, good):
            request.write_bytes(payload)
            child = subprocess.run([args.executable, '-in', str(request), '-out', str(output)], capture_output=True)
            assert (child.returncode == 0) == good, (payload[:100], child.stderr)
            return output.read_bytes() if good else None
        for name, payload in {
            'colliding_unknown_port': f'From,To\nSRB_18_95/{unknown},SRB_38_57/ZDN[7]\n'.encode(),
            'extra_column': b'From,To\nSRB_18_95/A_FQ0,SRB_38_57/ZDN[7],1\n',
            'negative_coordinate': b'From,To\nSRB_-1_95/A_FQ0,SRB_38_57/ZDN[7]\n',
            'out_of_grid': b'From,To\nSRB_120_95/A_FQ0,SRB_38_57/ZDN[7]\n',
            'nondigit_coordinate': b'From,To\nSRB_1x_95/A_FQ0,SRB_38_57/ZDN[7]\n',
            'integer_overflow': b'From,To\nSRB_99999999999999999999_95/A_FQ0,SRB_38_57/ZDN[7]\n',
            'bad_prefix': b'From,To\nBAD_18_95/A_FQ0,SRB_38_57/ZDN[7]\n',
            'overlong_row': b'From,To\nSRB_' + b'0'*1100 + b'18_95/A_FQ0,SRB_38_57/ZDN[7]\n',
        }.items():
            run(payload, False)
            passed.append(name)
        ordinary = run(b'From,To\nSRB_18_95/A_FQ0,SRB_38_57/ZDN[7]\n', True)
        padded = run(b'From,To\nSRB_000000018_000000095/A_FQ0,SRB_000000038_000000057/ZDN[7]', True)
        a = list(csv.DictReader(io.StringIO(ordinary.decode())))[0]
        b = list(csv.DictReader(io.StringIO(padded.decode())))[0]
        assert a['delay'] == b['delay']
        assert b['From'] == 'SRB_000000018_000000095/A_FQ0'
        passed.append('long_zero_padded_coordinate_and_name_preservation')
        assert run(b'From,To', True) == b'From,To,delay\n'
        passed.append('header_without_final_newline')
        request.write_bytes(b'From,To\nSRB_18_95/A_FQ0,SRB_38_57/ZDN[7]\n')
        child = subprocess.run([args.executable, '-in', str(request), '-out', '/dev/full'], capture_output=True)
        assert child.returncode != 0
        passed.append('write_failure_nonzero_exit')
    record = {'checks': passed, 'colliding_unknown_fixture': unknown, 'all_passed': True}
    Path(args.report).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
