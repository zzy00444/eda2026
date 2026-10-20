"""Verify stage-5 evidence, preserve stage 4, and package both accepted variants."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import zipfile

from project_paths import BUILD as B, REPO_ROOT

ROOT = REPO_ROOT
S = ROOT / 'solution'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(name):
    return json.loads((B / name).read_text(encoding='utf-8'))

def package(binary, target):
    data = binary.read_bytes()
    assert data[:4] == b'\x7fELF'
    info = zipfile.ZipInfo('bin/estimate', date_time=(2026,10,20,0,0,0))
    info.create_system = 3
    info.external_attr = 0o100755 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    assert target.stat().st_size < 100_000_000
    with zipfile.ZipFile(target) as z:
        assert z.namelist() == ['bin/estimate']
        assert z.read('bin/estimate') == data
        assert z.getinfo('bin/estimate').external_attr >> 16 == 0o100755
    return {'binary_bytes': len(data), 'zip_bytes': target.stat().st_size,
            'binary_sha256': sha(binary), 'zip_sha256': sha(target),
            'contents': ['bin/estimate'], 'unix_mode': '100755'}

def main():
    variants = {}
    for label in ('score', 'accuracy'):
        generated = 'generated_s5_' + label
        metrics = read(f'stage5_{label}_metrics.json')
        export = read(generated + '/export.json')
        metrics.update({k: export.get(k) for k in ('net_correction', 'net_coefficients', 'packed_ports')})
        metrics['model_sha256'] = sha(B / (export['model'] + '.txt'))
        metrics['generated_header_sha256'] = sha(B / generated / 'model.hpp')
        (B / f'stage5_{label}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        benchmark = read(f'stage5_{generated}_benchmark_10m.json')
        assert metrics['full_parity']['python_cpp_mismatch_rows'] == 0
        assert metrics['zero_samples']['all_correct']
        assert len(benchmark) == 5
        assert len({r['output_sha256'] for r in benchmark}) == 1
        for r in benchmark:
            assert r['rows'] == 10_000_000 and r['repeated_output_verified']
            assert r['max_observed_threads'] == 1 and r['container_peak_bytes'] < 2_000_000_000
        binary = S / 'bin' / f'estimate_stage5_{label}'
        assert binary.exists()
        avg_1b = sum(r['cpu_seconds'] for r in benchmark) / len(benchmark) * 100
        variants[label] = {
            'generated_dir': generated, 'binary': str(binary.relative_to(S)),
            'validation_accuracy': metrics['valid']['score'],
            'test_accuracy': metrics['test']['score'],
            'average_projected_1b_cpu_seconds': avg_1b,
            'average_projected_1b_cpu_minutes': avg_1b / 60,
            'projected_total_from_validation': .8*metrics['valid']['score'] + 15*max(0,1-avg_1b/1800) + 5,
            'projected_total_from_test': .8*metrics['test']['score'] + 15*max(0,1-avg_1b/1800) + 5,
            'cpu_10m_range': [min(r['cpu_seconds'] for r in benchmark), max(r['cpu_seconds'] for r in benchmark)],
            'max_process_rss_kib': max(r['process_peak_rss_kib'] for r in benchmark),
            'max_container_bytes': max(r['container_peak_bytes'] for r in benchmark),
            'benchmark_runs': 5, 'max_threads': 1,
            'output_10m_sha256': benchmark[0]['output_sha256'],
            'full_public_parity_mismatch': 0,
            'net_correction': export['net_correction'],
            'net_coefficients': export['net_coefficients'],
            'primary_model_sha256': metrics['model_sha256'],
            'generated_header_sha256': metrics['generated_header_sha256'],
            'packed_ports': export['packed_ports'],
        }
    control = read('stage5_stage4_control_benchmark_10m.json')
    control_acc = read('cascade10_formal_metrics.json')['test']['score']
    control_cpu = sum(r['cpu_seconds'] for r in control)/len(control)*100
    control_total = .8*control_acc + 15*max(0,1-control_cpu/1800)+5
    assert variants['score']['projected_total_from_validation'] > variants['accuracy']['projected_total_from_validation']
    assert variants['accuracy']['projected_total_from_test'] > control_total

    previous_zip = S / 'stage4_submission.zip'
    if not previous_zip.exists():
        assert sha(S / 'submission.zip') == 'c9fcd4c97298534e2fb6187d4c5e4f9c97ec6e2b81b70a5a3f77cc77b9673960'
        shutil.copy2(S / 'submission.zip', previous_zip)
        shutil.copytree(B / 'generated', B / 'generated_stage4_baseline', dirs_exist_ok=True)
    assert sha(previous_zip) == 'c9fcd4c97298534e2fb6187d4c5e4f9c97ec6e2b81b70a5a3f77cc77b9673960'
    for label in variants:
        target = S / f'stage5_{label}_submission.zip'
        variants[label]['package'] = package(S / variants[label]['binary'], target)
        variants[label]['submission'] = target.name
    shutil.copy2(S / variants['score']['binary'], S / 'bin/estimate')
    shutil.copy2(S / 'stage5_score_submission.zip', S / 'submission.zip')
    for p in (B / variants['score']['generated_dir']).iterdir():
        shutil.copy2(p, B / 'generated' / p.name)
    report = {
        'stage': 5, 'date': '2026-10-20', 'default': 'score', 'variants': variants,
        'same_session_stage4_control': {'test_accuracy': control_acc, 'runs': len(control),
            'average_projected_1b_cpu_seconds': control_cpu,
            'projected_total_from_test': control_total},
        'compiler': 'GCC 11.4.0', 'docker_image': 'fpga-ziguang:latest',
        'compiler_flags': '-O3 -march=x86-64 -ffp-contract=off -fprofile-use=profile -fprofile-correction -DNDEBUG -std=c++17',
        'full_1b_run_completed': False,
        'score_caveat': 'Local held-out accuracy plus 10M timing extrapolation, not an official hidden-set score.',
        'source_hashes': {str(p.relative_to(S)): sha(p) for p in [S/'src/main.cpp',
            S/'tools/export_cpp.py', S/'tools/verify_cpp.py', S/'tools/net_correction_experiment.py',
            S/'tools/build_candidate_wsl.sh', S/'tools/validate_stage5_wsl.sh']},
    }
    (B / 'stage5_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
