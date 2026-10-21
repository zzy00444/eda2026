"""Check evidence against the actual ELF and ZIP before promoting stage 8.

Run on Windows after displacement_run.py benchmark/validate and package.py.
No compilation, fitting, packaging or external publishing occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile

from displacement_assets import sha, TABLES
from project_paths import DATA, REPO_ROOT, SOLUTION_ROOT

DOCS = REPO_ROOT / 'docs'
BUILD = SOLUTION_ROOT / 'build/displacement'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_payload(binary, build):
    """Verify each embedded model segment, including the calibration table."""
    data = binary.read_bytes()
    footer = data[-112:]
    assert data[:4] == b'\x7fELF' and footer[:8] == b'PANGDL02'
    fields = struct.unpack('<6Q6Q2I', footer[8:])
    offsets, sizes = fields[:6], fields[6:12]
    assert fields[12:] == (6, 2)
    actual = {}
    names = [*TABLES, 'calibration_train.bin']
    paths = [BUILD / 'oracle/oracle_vt.bin', BUILD / 'U_refit_q16.bin',
             BUILD / 'M_refit.bin', BUILD / 'oracle/oracle_ids.txt',
             BUILD / 'oracle/oracle_arch.txt', BUILD / 'calibration_train.bin']
    for i, (name, path) in enumerate(zip(names, paths)):
        begin, length = offsets[i], sizes[i]
        assert 0 < begin < begin + length <= len(data)-112
        if i:
            assert offsets[i-1] + sizes[i-1] == begin
        digest = hashlib.sha256(data[begin:begin+length]).hexdigest()
        assert path.stat().st_size == length and sha(path) == digest, name
        if name in build['table_sha256']:
            assert build['table_sha256'][name] == digest
        actual[name] = {'bytes': length, 'sha256': digest}
    assert offsets[-1] + sizes[-1] == len(data)-112
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', choices=['displacement_cal16', 'displacement_cal64'])
    args = parser.parse_args()
    name = args.candidate
    binary = SOLUTION_ROOT / 'bin' / ('estimate_' + name)
    package = SOLUTION_ROOT / (name + '_submission.zip')
    build = read(BUILD / (name + '_build.json'))
    analysis = read(DOCS / (name + '_analysis.json'))
    packaged = read(DOCS / (name + '_package.json'))
    validation = read(DOCS / (name + '_validation.json'))
    benchmark = read(DOCS / 'displacement_final_benchmark.json')
    binary_sha, zip_sha = sha(binary), sha(package)
    assert binary_sha == build['binary_sha256'] == packaged['binary_sha256'] == validation['binary_sha256']
    assert zip_sha == packaged['zip_sha256'] == validation['zip_sha256']
    assert build['source_sha256'] == sha(SOLUTION_ROOT / 'src/displacement.cpp')
    assert binary.stat().st_size == build['binary_bytes'] == packaged['binary_bytes']
    assert package.stat().st_size == packaged['zip_bytes'] < 100_000_000
    with zipfile.ZipFile(package) as archive:
        assert archive.namelist() == ['bin/estimate']
        assert archive.getinfo('bin/estimate').external_attr >> 16 == 0o100755
        assert hashlib.sha256(archive.read('bin/estimate')).hexdigest() == binary_sha
    payload = check_payload(binary, build)
    refit = read(DOCS / 'displacement_refit.json')
    quantization = read(DOCS / 'U_refit_q16_quantization.json')
    calibration = read(DOCS / 'displacement_train_calibration.json')
    assert refit['M_sha256'] == payload['oracle_M.bin']['sha256']
    assert quantization['source_sha256'] == refit['U_sha256']
    assert quantization['output_sha256'] == payload['oracle_U.bin']['sha256']
    assert calibration['output_sha256'] == payload['calibration_train.bin']['sha256']
    assert validation['all_passed']
    checks = {}
    for suffix, digest in validation['reports'].items():
        path = DOCS / (name + '_' + suffix + '.json')
        assert sha(path) == digest
        checks[suffix] = read(path)
    assert len(checks['integration']['interface_checks']) == 11
    assert checks['contract']['all_passed'] and len(checks['contract']['checks']) == 11
    assert checks['standalone']['standalone_zip_test'] == 'passed'
    assert checks['standalone']['binary_sha256'] == binary_sha
    measured = benchmark['candidates'][name]
    assert measured['binary_sha256'] == binary_sha
    assert measured['reference_1m_sha256'] == analysis['output_sha256']
    assert analysis['input_sha256'] == sha(DATA / 'data/delay_estimate_ans.csv')
    assert benchmark['runs'] == 5 and len(measured['runs']) == 5
    assert benchmark['cpu_limit'] == 1 and benchmark['memory_limit_bytes'] == 2_000_000_000
    digests = set()
    for run in measured['runs']:
        assert run['rows'] == 10_000_000 and run['repeated_output_verified']
        assert run['max_observed_threads'] == 1 and run['container_peak_bytes'] < 2_000_000_000
        digests.add(run['output_sha256'])
    assert len(digests) == 1
    assert analysis['valid']['rows'] == analysis['test']['rows'] == 50_000
    assert analysis['full_public']['rows'] == 1_000_000
    comparison = {}
    for candidate in ('stage7_fold', 'displacement_baseline', 'displacement_cal16', 'displacement_cal64'):
        candidate_runs = benchmark['candidates'][candidate]['runs']
        assert len(candidate_runs) == 5
        cpu = sum(item['cpu_seconds'] for item in candidate_runs) / 5
        if candidate == 'stage7_fold':
            score = read(SOLUTION_ROOT / 'build/stage7_fold_metrics.json')['test']['score']
        else:
            # cal16 and cal64 have identical outputs, checked below.
            score = read(DOCS / (('displacement_baseline' if candidate == 'displacement_baseline'
                                 else name) + '_analysis.json'))['test']['accuracy']
        comparison[candidate] = {'test_accuracy': score, 'mean_10m_cpu_seconds': cpu,
                                 'projected_1b_cpu_seconds': cpu * 100,
                                 'projected_total': .8*score + 5 + 15*max(0., 1-cpu*100/1800)}
    assert benchmark['candidates']['displacement_cal16']['reference_1m_sha256'] == \
        benchmark['candidates']['displacement_cal64']['reference_1m_sha256']
    current = comparison[name]
    assert current['projected_1b_cpu_seconds'] < 1800
    assert current['test_accuracy'] > comparison['displacement_baseline']['test_accuracy']
    assert current['projected_total'] > comparison['stage7_fold']['projected_total']
    evidence = ['displacement_final_benchmark.json', name + '_analysis.json', name + '_package.json',
                name + '_validation.json', *[name + '_' + suffix + '.json' for suffix in checks],
                'displacement_import.json', 'displacement_refit.json', 'U_refit_q16_quantization.json',
                'displacement_train_calibration.json',
                'displacement_generalization.json']
    manifest = {'stage': 8, 'date': '2026-10-21', 'candidate': name,
                'binary': 'solution/bin/estimate', 'submission': 'solution/submission.zip',
                'design': 'full-displacement rank128, U-int16, train-only M and calibration',
                'validation_accuracy': analysis['valid']['accuracy'], **current,
                'full_public_accuracy': analysis['full_public']['accuracy'],
                'max_process_rss_kib': max(r['process_peak_rss_kib'] for r in measured['runs']),
                'max_container_bytes': measured['max_container_bytes'], 'max_threads': 1,
                'output_1m_sha256': analysis['output_sha256'], 'output_10m_sha256': digests.pop(),
                'package': packaged, 'build': build, 'embedded_assets': payload,
                'comparison_same_batch': comparison,
                'cpu_10m_range': [min(r['cpu_seconds'] for r in measured['runs']),
                                  max(r['cpu_seconds'] for r in measured['runs'])],
                'selection_reason': 'cal16: higher accuracy, smaller ZIP and complete input checks; '
                                    'projected total tied with original baseline within timing noise',
                'total_delta_vs_original_baseline': current['projected_total'] -
                    comparison['displacement_baseline']['projected_total'],
                'evidence_sha256': {item: sha(DOCS / item) for item in evidence},
                'source_sha256': {str(path.relative_to(REPO_ROOT)): sha(path)
                                  for path in [SOLUTION_ROOT / 'src/displacement.cpp',
                                               *[SOLUTION_ROOT / 'tools' / item for item in
                                                 ('displacement_assets.py', 'displacement_run.py',
                                                  'displacement_refit.py', 'displacement_quantize.py',
                                                  'displacement_calibrate.py', 'displacement_analyze.py',
                                                  'displacement_contract.py', 'release_stage8.py')]]},
                'official_score': False, 'full_1b_completed': False,
                'limitations': ['Historical rng7 regression slices, not a blind test',
                                '48 reserved oracle queries slightly regress in relative score',
                                'CPU timing extrapolated from 10M, not an official result']}
    # All gates passed. Keep stage7 backups and promote the already tested files.
    shutil.copy2(binary, SOLUTION_ROOT / 'bin/estimate')
    shutil.copy2(package, SOLUTION_ROOT / 'submission.zip')
    assert sha(SOLUTION_ROOT / 'bin/estimate') == binary_sha
    assert sha(SOLUTION_ROOT / 'submission.zip') == zip_sha
    (DOCS / 'stage8_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(BUILD / (name + '_build.json'), DOCS / (name + '_build.json'))
    print(json.dumps({key: manifest[key] for key in ('candidate', 'test_accuracy',
                     'mean_10m_cpu_seconds', 'projected_total', 'package')}, indent=2))


if __name__ == '__main__':
    main()
