"""Verify stage-10 local artifacts, then promote or retain them locally.

Default promotion requires accuracy >=98 and a same-batch score >=stage8.
The additional accuracy alternative follows the user's <1800s CPU budget.
"""
import argparse
import hashlib
import json
import shutil
import struct
import zipfile
import numpy as np
from displacement_assets import sha
from project_paths import REPO_ROOT, SOLUTION_ROOT, DATA
from release_stage8 import check_payload

DOCS=REPO_ROOT/'docs'
BUILD=SOLUTION_ROOT/'build/displacement'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name');ap.add_argument('--retain',action='store_true')
    ap.add_argument('--benchmark',default='stage10_final_benchmark.json')
    args=ap.parse_args();name=args.name
    assert name.replace('_','').isalnum()
    binary=SOLUTION_ROOT/'bin'/('estimate_'+name);package=SOLUTION_ROOT/(name+'_submission.zip')
    build=read(BUILD/(name+'_build.json'));analysis=read(DOCS/(name+'_analysis.json'))
    ref=read(DOCS/(name+'_reference.json'));wall=read(DOCS/(name+'_wall_export.json'))
    training=read(DOCS/(wall['model']+'.json'));packaged=read(DOCS/(name+'_package.json'))
    validation=read(DOCS/(name+'_validation.json'));bench=read(DOCS/args.benchmark)
    local=read(DOCS/(wall['base']+'_local_export.json'))
    local_training=read(DOCS/(local['factors']+'_training.json'))
    binary_sha=sha(binary);zip_sha=sha(package)
    assert binary_sha==build['binary_sha256']==ref['binary_sha256']==packaged['binary_sha256']==validation['binary_sha256']
    assert zip_sha==packaged['zip_sha256']==validation['zip_sha256']
    assert build['source_sha256']==sha(SOLUTION_ROOT/'src/displacement.cpp')
    assert build['local_runtime_sha256']==sha(SOLUTION_ROOT/'src/displacement_local_runtime.hpp')
    assert build['wall_runtime_sha256']==sha(SOLUTION_ROOT/'src/displacement_wall_runtime.hpp')
    assert build['wall_header_sha256']==wall['header_sha256']==sha(BUILD/(name+'.wall.hpp'))
    assert wall['model_sha256']==training['model_sha256']==ref['wall_model_sha256']==sha(BUILD/(wall['model']+'.txt'))
    assert wall['trees']==ref['wall_trees']
    assert not training['test_labels_used'] and not training['architecture_bound_used'] and training['fit_rows']==4233
    assert not local_training['public_delay_labels_used'] and local_training['geometry_training_rows_only']
    assert local_training['scratch_removed'] and local['factors_sha256']==local_training['factors_sha256']
    assert local['factors_sha256']==sha(BUILD/(local['factors']+'_factors.npz'))
    local_path=BUILD/(wall['base']+'.local')
    assert build['local_asset_sha256']==ref['local_asset_sha256']==local['asset_sha256']==sha(local_path)
    assert {'-DEST_LOCAL','-DEST_WALL','-DEST_FAST_ROUND'}<=set(build['compiler_flags'])
    assert '-DEST_TRACE' not in build['compiler_flags']
    assert binary.stat().st_size==build['binary_bytes']==packaged['binary_bytes']
    assert package.stat().st_size==packaged['zip_bytes']<100_000_000
    with zipfile.ZipFile(package) as archive:
        assert archive.namelist()==['bin/estimate']
        assert archive.getinfo('bin/estimate').external_attr>>16==0o100755
        assert hashlib.sha256(archive.read('bin/estimate')).hexdigest()==binary_sha
    payload=check_payload(binary,build)
    with binary.open('rb') as stream:
        stream.seek(-112,2);first=struct.unpack('<Q',stream.read(16)[8:])[0]
        stream.seek(first-32);desc=stream.read(32)
        assert desc[:8]==b'PGLCF001'
        offset,size,reserved=struct.unpack('<3Q',desc[8:])
        assert reserved==0 and offset+size==first-32 and size==local['asset_bytes']
        stream.seek(offset);assert hashlib.sha256(stream.read(size)).hexdigest()==local['asset_sha256']
    assert read(DOCS/'displacement_refit.json')['M_sha256']==payload['oracle_M.bin']['sha256']
    assert read(DOCS/'U_refit_q16_quantization.json')['output_sha256']==payload['oracle_U.bin']['sha256']
    assert read(DOCS/'displacement_train_calibration.json')['output_sha256']==payload['calibration_train.bin']['sha256']
    assert ref['rows']==1_000_000 and ref['python_cpp_mismatch_count']==ref['stage8_reference_mismatch_count']==0
    assert ref['both_branches_reconstructed'] and ref['float32_fma_emulated']
    assert ref['reference_sha256']==sha(BUILD/(name+'_independent_reference.npy'))
    assert np.array_equal(np.load(BUILD/(name+'_independent_reference.npy')),np.load(BUILD/(name+'_prediction.npy')))
    assert ref['output_sha256']==analysis['output_sha256']==sha(BUILD/(name+'_result.csv'))
    assert analysis['input_sha256']==sha(DATA/'data/delay_estimate_ans.csv')
    for key in ('valid','test'):
        assert analysis[key]==wall[key]==ref['splits'][key]['updated']
        assert analysis[key]['rows']==50000 and analysis[key]['accuracy']>=98
    assert wall['position_assess']==ref['splits']['position_assess']['updated']
    assert wall['position_assess']['rows']==180375 and wall['position_assess']['accuracy']>=98
    oracle=ref['oracle'];assert oracle['python_cpp_mismatch_count']==0
    for key,path in [('truth','oracle_answers.csv'),('request','oracle_request.csv'),('spec','oracle_spec.json')]:
        assert oracle[key+'_sha256']==sha(BUILD/path)
    assert oracle['output_sha256']==sha(BUILD/(name+'_oracle.csv'))
    assert validation['all_passed']
    checks={}
    for suffix,digest in validation['reports'].items():
        path=DOCS/(name+'_'+suffix+'.json');assert sha(path)==digest;checks[suffix]=read(path)
    assert len(checks['integration']['interface_checks'])==11
    assert checks['contract']['all_passed'] and len(checks['contract']['checks'])==11
    assert checks['standalone']['standalone_zip_test']=='passed' and checks['standalone']['binary_sha256']==binary_sha
    assert bench['runs']==5 and bench['rows']==10_000_000 and bench['cpu_limit']==1 and bench['memory_limit_bytes']==2_000_000_000
    comparison={}
    for candidate,measured in bench['candidates'].items():
        candidate_binary=SOLUTION_ROOT/'bin'/('estimate_'+candidate)
        candidate_analysis=read(DOCS/(candidate+'_analysis.json'))
        assert measured['binary_sha256']==sha(candidate_binary)
        assert measured['reference_1m_sha256']==candidate_analysis['output_sha256']
        assert len(measured['runs'])==5
        digests=set()
        for run in measured['runs']:
            assert run['rows']==10_000_000 and run['repeated_output_verified']
            assert run['max_observed_threads']==1 and run['container_peak_bytes']<2_000_000_000
            digests.add(run['output_sha256'])
        assert len(digests)==1
        cpu=sum(r['cpu_seconds'] for r in measured['runs'])/5
        acc=candidate_analysis['test']['accuracy']
        comparison[candidate]=dict(test_accuracy=acc,mean_10m_cpu_seconds=cpu,projected_1b_cpu_seconds=cpu*100,
                                   projected_total=.8*acc+20-cpu*100/120)
    current=comparison[name];stage8=comparison['displacement_cal16']
    measured=bench['candidates'][name]
    assert max(r['cpu_seconds'] for r in measured['runs'])*100<1800
    joint=current['projected_total']>=stage8['projected_total']
    if args.retain:
        previous=read(DOCS/'stage9_accuracy_manifest.json')
        assert current['test_accuracy']>=previous['test_accuracy']
        assert analysis['valid']['accuracy']>=previous['validation_accuracy']
        previous_path=DOCS/'stage10_accuracy_manifest.json'
        if previous_path.exists():
            previous=read(previous_path)
            assert current['test_accuracy']>=previous['test_accuracy']
            assert analysis['valid']['accuracy']>=previous['validation_accuracy']
    else:
        assert joint,('same-batch stage8 score gate failed',current,stage8)
    stage8_manifest=read(DOCS/'stage8_manifest.json')
    assert sha(SOLUTION_ROOT/'bin/estimate_displacement_cal16')==stage8_manifest['package']['binary_sha256']
    assert sha(SOLUTION_ROOT/'displacement_cal16_submission.zip')==stage8_manifest['package']['zip_sha256']
    # User requested both the prior releases and an additional highest-accuracy version.
    for old in ('stage9_fast16','stage9_accurate64'):
        old_manifest=read(DOCS/('stage9_manifest.json' if old.endswith('fast16') else 'stage9_accuracy_manifest.json'))
        assert sha(SOLUTION_ROOT/'bin'/('estimate_'+old))==old_manifest['package']['binary_sha256']
        assert sha(SOLUTION_ROOT/(old+'_submission.zip'))==old_manifest['package']['zip_sha256']
    evidence=[args.benchmark,name+'_analysis.json',name+'_reference.json',name+'_wall_export.json',
              name+'_package.json',name+'_validation.json',*[name+'_'+s+'.json' for s in checks],
              wall['model']+'.json',wall['base']+'_local_export.json',local['factors']+'_training.json',
              'stage8_manifest.json','stage9_manifest.json','stage9_accuracy_manifest.json']
    sources=['src/displacement.cpp','src/displacement_local_runtime.hpp','src/displacement_wall_runtime.hpp']
    sources+=['tools/'+p for p in ('displacement_local_train.py','displacement_local_export.py','displacement_local_screen.py',
               'displacement_wall_train.py','displacement_wall_export.py','displacement_local_verify.py','displacement_run.py',
               'displacement_assets.py','displacement_analyze.py','displacement_package.py','release_stage10.py')]
    manifest=dict(stage=10,date='2026-10-21',candidate=name,retained_alternative=args.retain,
                  binary='solution/bin/estimate_'+name if args.retain else 'solution/bin/estimate',
                  submission='solution/'+name+'_submission.zip' if args.retain else 'solution/submission.zip',
                  validation_accuracy=analysis['valid']['accuracy'],**current,
                  full_public_accuracy=analysis['full_public']['accuracy'],position_assess=wall['position_assess'],
                  reserved_48=oracle['reserved_48'],python_cpp_mismatch_count=0,
                  comparison_same_batch=comparison,joint_gate_passed=joint,
                  projected_total_delta_vs_stage8=current['projected_total']-stage8['projected_total'],
                  projected_total_delta_vs_stage8_historical=current['projected_total']-stage8_manifest['projected_total'],
                  max_container_bytes=measured['max_container_bytes'],max_threads=1,cpu_budget_seconds=1800,
                  cpu_10m_range=[min(r['cpu_seconds'] for r in measured['runs']),max(r['cpu_seconds'] for r in measured['runs'])],
                  package=packaged,build=build,embedded_assets=payload,local=local,wall=wall,
                  evidence_sha256={p:sha(DOCS/p) for p in evidence},
                  source_sha256={p:sha(SOLUTION_ROOT/p) for p in sources},
                  official_score=False,full_1b_completed=False,
                  limitations=['Historical rng7 slices, repeatedly inspected; no blind or official accuracy claim',
                               'CPU extrapolated from repeated 10M requests, not a complete 1B run',
                               'Position exclusion covers only new wall parameters and local geometry weights',
                               '48 exact reserved queries still expose structural approximation errors'])
    # All checks finish before canonical replacement. Named previous releases survive.
    if not args.retain:
        shutil.copy2(binary,SOLUTION_ROOT/'bin/estimate');shutil.copy2(package,SOLUTION_ROOT/'submission.zip')
        assert sha(SOLUTION_ROOT/'bin/estimate')==binary_sha and sha(SOLUTION_ROOT/'submission.zip')==zip_sha
    target=DOCS/('stage10_accuracy_manifest.json' if args.retain else 'stage10_manifest.json')
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(BUILD/(name+'_build.json'),DOCS/(name+'_build.json'))
    print(json.dumps({k:manifest[k] for k in ('candidate','retained_alternative','validation_accuracy','test_accuracy',
                                          'mean_10m_cpu_seconds','projected_total','joint_gate_passed','projected_total_delta_vs_stage8')},indent=2))


if __name__=='__main__':main()
