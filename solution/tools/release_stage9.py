"""Verify stage-9 evidence and promote the accuracy candidate locally."""
import argparse
import hashlib
import json
import shutil
import struct
import zipfile
import zlib

import numpy as np

from displacement_assets import sha
from project_paths import DATA, REPO_ROOT, SOLUTION_ROOT
from release_stage8 import check_payload

DOCS = REPO_ROOT / 'docs'
BUILD = SOLUTION_ROOT / 'build/displacement'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', choices=['stage9_fast16', 'stage9_accurate64'])
    parser.add_argument('--retain', action='store_true', help='Verify and retain an alternative without changing the default')
    args = parser.parse_args()
    name = args.candidate
    binary = SOLUTION_ROOT / 'bin' / ('estimate_'+name)
    package = SOLUTION_ROOT / (name+'_submission.zip')
    build = read(BUILD / (name+'_build.json'))
    analysis = read(DOCS / (name+'_analysis.json'))
    exported = read(DOCS / (name+'_export.json'))
    frozen = read(BUILD / (name+'_frozen.json'))
    packaged = read(DOCS / (name+'_package.json'))
    validation = read(DOCS / (name+'_validation.json'))
    benchmark = read(DOCS / 'stage9_final_benchmark.json')
    general = read(DOCS / 'stage9_generalization.json')
    verified = general['candidates'][name]
    encoding = read(DOCS / 'stage9_bound_encoding.json')
    binary_sha, zip_sha = sha(binary), sha(package)
    assert binary_sha == build['binary_sha256'] == packaged['binary_sha256'] == validation['binary_sha256'] == verified['binary_sha256']
    assert zip_sha == packaged['zip_sha256'] == validation['zip_sha256']
    assert build['source_sha256'] == sha(SOLUTION_ROOT / 'src/displacement.cpp')
    assert build['residual_runtime_sha256'] == sha(SOLUTION_ROOT / 'src/displacement_residual_runtime.hpp')
    assert build['residual_header_sha256'] == exported['header_sha256'] == sha(BUILD / (name+'.hpp'))
    for key, value in frozen.items():
        assert exported[key] == value, key
    assert exported['model_sha256'] == sha(BUILD / (exported['model']+'.txt'))
    training = read(DOCS / (exported['model']+'.json'))
    assert not training['test_labels_used'] and training['fit_rows'] == general['fit_rows'] == 719583
    assert training['model_sha256'] == exported['model_sha256']
    assert exported['bound_clamp'] and exported['direct_cpp_branches'] and exported['integer_features']
    assert '-DEST_RESIDUAL_BATCH' in build['compiler_flags']
    assert '-DEST_RESIDUAL' in build['compiler_flags'] and build['libraries'] == ['-lz']
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
    with binary.open('rb') as stream:
        stream.seek(-112, 2)
        first = struct.unpack('<Q', stream.read(16)[8:])[0]
        stream.seek(first-32)
        descriptor = stream.read(32)
        assert descriptor[:8] == b'PGBND001'
        offset, size, decoded_size = struct.unpack('<3Q', descriptor[8:])
        assert offset+size == first-32 and size == encoding['compressed_bytes']
        assert decoded_size == encoding['decoded_stream_bytes']
        stream.seek(offset)
        compressed = stream.read(size)
    assert hashlib.sha256(compressed).hexdigest() == encoding['compressed_sha256'] == build['bound_sha256'] == exported['bound_sha256'] == sha(BUILD / 'stage9_bound.z')
    assert len(zlib.decompress(compressed)) == decoded_size and encoding['all_values_roundtrip_verified']
    assert encoding['source_table_sha256'] == sha(SOLUTION_ROOT / 'build/architecture_table.npy')
    assert encoding['source_meta_sha256'] == sha(SOLUTION_ROOT / 'build/architecture_meta.json')
    assert validation['all_passed']
    checks = {}
    for suffix, digest in validation['reports'].items():
        path = DOCS / (name+'_'+suffix+'.json')
        assert sha(path) == digest
        checks[suffix] = read(path)
    assert len(checks['integration']['interface_checks']) == 11
    assert checks['contract']['all_passed'] and len(checks['contract']['checks']) == 11
    assert checks['standalone']['standalone_zip_test'] == 'passed'
    assert checks['standalone']['binary_sha256'] == binary_sha
    assert general['position_assess_rows'] == 180375
    assert verified['rows'] == 1_000_000 and verified['python_cpp_mismatch_count'] == verified['max_absolute_difference'] == 0
    assert verified['reference_sha256'] == sha(BUILD / (name+'_reference.npy'))
    assert verified['prediction_sha256'] == sha(BUILD / (name+'_prediction.npy'))
    assert np.array_equal(np.load(BUILD / (name+'_reference.npy')), np.load(BUILD / (name+'_prediction.npy')))
    assert verified['export_report_sha256'] == sha(DOCS / (name+'_export.json'))
    assert verified['output_sha256'] == analysis['output_sha256'] == sha(BUILD / (name+'_result.csv'))
    for key in ('oracle_truth', 'oracle_request', 'oracle_spec'):
        file_name = {'oracle_truth': 'oracle_answers.csv', 'oracle_request': 'oracle_request.csv', 'oracle_spec': 'oracle_spec.json'}[key]
        assert general[key+'_sha256'] == sha(BUILD / file_name)
    assert verified['oracle_output_sha256'] == sha(BUILD / (name+'_oracle.csv'))
    assert analysis['input_sha256'] == sha(DATA / 'data/delay_estimate_ans.csv')
    assert analysis['full_public']['rows'] == 1_000_000
    for key in ('valid', 'test'):
        assert analysis[key]['rows'] == 50_000 and analysis[key]['accuracy'] > 98
        assert analysis[key] == exported[key]
    assert verified['position_assess'] == exported['position_assess']
    assert verified['position_assess']['accuracy'] > 98
    assert benchmark['runs'] == 5 and benchmark['cpu_limit'] == 1 and benchmark['memory_limit_bytes'] == 2_000_000_000
    comparison = {}
    for candidate, measured in benchmark['candidates'].items():
        assert len(measured['runs']) == 5
        candidate_binary = SOLUTION_ROOT / 'bin' / ('estimate_'+candidate)
        candidate_analysis = read(DOCS / (candidate+'_analysis.json'))
        assert measured['binary_sha256'] == sha(candidate_binary)
        assert measured['reference_1m_sha256'] == candidate_analysis['output_sha256']
        digests = set()
        for run in measured['runs']:
            assert run['rows'] == 10_000_000 and run['repeated_output_verified']
            assert run['max_observed_threads'] == 1 and run['container_peak_bytes'] < 2_000_000_000
            digests.add(run['output_sha256'])
        assert len(digests) == 1
        cpu = sum(r['cpu_seconds'] for r in measured['runs'])/5
        accuracy = candidate_analysis['test']['accuracy']
        comparison[candidate] = dict(test_accuracy=accuracy, mean_10m_cpu_seconds=cpu,
                                     projected_1b_cpu_seconds=cpu*100,
                                     projected_total=.8*accuracy+5+15*max(0.,1-cpu*100/1800),
                                     output_10m_sha256=next(iter(digests)))
    measured = benchmark['candidates'][name]
    current = comparison[name]
    # User explicitly requires the retained accuracy version below the 30-minute
    # zero-time-score threshold, with every local extrapolated run below it.
    assert current['projected_1b_cpu_seconds'] < 1800
    assert max(r['cpu_seconds'] for r in measured['runs'])*100 < 1800
    stage8 = read(DOCS / 'stage8_manifest.json')
    assert sha(SOLUTION_ROOT / 'bin/estimate_displacement_cal16') == stage8['package']['binary_sha256']
    assert sha(SOLUTION_ROOT / 'displacement_cal16_submission.zip') == stage8['package']['zip_sha256']
    evidence = ['stage9_final_benchmark.json', 'stage9_generalization.json', 'stage9_bound_encoding.json',
                'stage8_manifest.json', name+'_analysis.json', name+'_export.json', name+'_package.json',
                name+'_validation.json', *[name+'_'+suffix+'.json' for suffix in checks],
                exported['model']+'.json', 'residual_selection.json', 'displacement_refit.json',
                'U_refit_q16_quantization.json', 'displacement_train_calibration.json']
    source_paths = [SOLUTION_ROOT / 'src/displacement.cpp', SOLUTION_ROOT / 'src/displacement_residual_runtime.hpp']
    source_paths += [SOLUTION_ROOT / 'tools' / item for item in (
        'displacement_residual.py', 'displacement_residual_export.py', 'displacement_residual_verify.py',
        'displacement_bound_pack.py', 'displacement_residual_select.py', 'displacement_assets.py',
        'displacement_run.py', 'displacement_analyze.py', 'displacement_package.py',
        'displacement_contract.py', 'benchmark.py', 'integration.py', 'standalone.py', 'architecture.py',
        'release_stage8.py', 'release_stage9.py')]
    manifest = dict(stage=9, date='2026-10-21', candidate=name,
                    binary=('solution/bin/estimate_'+name if args.retain else 'solution/bin/estimate'),
                    submission=('solution/'+name+'_submission.zip' if args.retain else 'solution/submission.zip'),
                    retained_alternative=args.retain, cpu_budget_seconds=1800,
                    design='stage8 + exact periodic envelope + frozen weighted-L1 position residual; direct C++ trees',
                    validation_accuracy=analysis['valid']['accuracy'], **current,
                    full_public_accuracy=analysis['full_public']['accuracy'],
                    position_assess=verified['position_assess'], reserved_48=verified['reserved_48'],
                    output_1m_sha256=analysis['output_sha256'], python_cpp_mismatch_count=0,
                    max_process_rss_kib=max(r['process_peak_rss_kib'] for r in measured['runs']),
                    max_container_bytes=measured['max_container_bytes'], max_threads=1,
                    cpu_10m_range=[min(r['cpu_seconds'] for r in measured['runs']), max(r['cpu_seconds'] for r in measured['runs'])],
                    package=packaged, build=build, embedded_assets=payload,
                    bound_encoding=encoding, residual=exported,
                    comparison_same_batch=comparison,
                    selection_reason=('Choose the faster validated candidate above 98; retain accurate64 for extra accuracy and stage8 for speed and higher projected total'
                                      if name == 'stage9_fast16' else
                                      'User prioritizes maximum accuracy above 98; retain fast16 and stage8 as faster alternatives'),
                    projected_total_delta_vs_stage8=current['projected_total']-comparison['displacement_cal16']['projected_total'],
                    evidence_sha256={item: sha(DOCS / item) for item in evidence},
                    source_sha256={str(path.relative_to(REPO_ROOT)): sha(path) for path in source_paths},
                    official_score=False, full_1b_completed=False,
                    limitations=['Historical rng7 regression slices, not a blind test',
                                 'Position holdout excludes only newly fitted residual parameters',
                                 'CPU timing extrapolated from 10M; initialized model cost is included in each run',
                                 '48 exact reserved queries do not represent the full hidden distribution',
                                 'Unsupported directions and invalid Net landing retain zero fallback'])
    # All checks precede replacement of canonical artifacts; alternatives remain available.
    if not args.retain:
        shutil.copy2(binary, SOLUTION_ROOT / 'bin/estimate')
        shutil.copy2(package, SOLUTION_ROOT / 'submission.zip')
        assert sha(SOLUTION_ROOT / 'bin/estimate') == binary_sha
        assert sha(SOLUTION_ROOT / 'submission.zip') == zip_sha
    manifest_name = 'stage9_accuracy_manifest.json' if args.retain else 'stage9_manifest.json'
    (DOCS / manifest_name).write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(BUILD / (name+'_build.json'), DOCS / (name+'_build.json'))
    print(json.dumps({key: manifest[key] for key in ('candidate', 'test_accuracy', 'mean_10m_cpu_seconds', 'projected_total', 'package')}, indent=2))


if __name__ == '__main__':
    main()
