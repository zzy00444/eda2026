"""Verify actual stage-11 artifacts before local promotion or extra retention."""
import argparse
import hashlib
import json
import lzma
import shutil
import struct
import zipfile
import numpy as np
from displacement_assets import sha
from project_paths import REPO_ROOT,SOLUTION_ROOT,DATA
from release_stage8 import check_payload

DOCS=REPO_ROOT/'docs';BUILD=SOLUTION_ROOT/'build/displacement'
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('name');ap.add_argument('--retain',action='store_true')
    ap.add_argument('--benchmark',default='stage11_final_benchmark.json');args=ap.parse_args();name=args.name
    assert name.replace('_','').isalnum()
    binary=SOLUTION_ROOT/'bin'/('estimate_'+name);package=SOLUTION_ROOT/(name+'_submission.zip')
    build=read(BUILD/(name+'_build.json'));analysis=read(DOCS/(name+'_analysis.json'))
    ref=read(DOCS/(name+'_reference.json'));wall=read(DOCS/(name+'_wall_export.json'))
    training=read(DOCS/(wall['model']+'.json'));packaged=read(DOCS/(name+'_package.json'))
    validation=read(DOCS/(name+'_validation.json'));bench=read(DOCS/args.benchmark)
    local=read(DOCS/(wall['base']+'_local_export.json'));localpath=BUILD/(wall['base']+'.local')
    bs=sha(binary);zs=sha(package);ls=sha(localpath)
    assert bs==build['binary_sha256']==analysis.get('binary_sha256',bs)==ref['binary_sha256']==packaged['binary_sha256']==validation['binary_sha256']
    assert zs==packaged['zip_sha256']==validation['zip_sha256']
    assert build['source_sha256']==sha(SOLUTION_ROOT/'src/displacement.cpp')
    assert build['local_runtime_sha256']==sha(SOLUTION_ROOT/'src/displacement_local_runtime.hpp')
    assert build['wall_runtime_sha256']==sha(SOLUTION_ROOT/'src/displacement_wall_runtime.hpp')
    assert build['wall_header_sha256']==wall['header_sha256']==sha(BUILD/(name+'.wall.hpp'))
    assert wall['model_sha256']==training['model_sha256']==ref['wall_model_sha256']==sha(BUILD/(wall['model']+'.txt'))
    assert training['fit_rows']==4233 and not training['test_labels_used'] and not training['architecture_bound_used']
    assert wall['trees']==ref['wall_trees']
    assert ls==local['asset_sha256']==build['local_asset_sha256']==ref['local_asset_sha256']
    assert localpath.stat().st_size==local['asset_bytes']
    evidence=[args.benchmark,name+'_analysis.json',name+'_reference.json',name+'_wall_export.json',name+'_package.json',name+'_validation.json',wall['model']+'.json',wall['base']+'_local_export.json']
    evidence += [wall['base']+'_profile.json','stage11_profiles.json','stage11_tile_screen.json','stage11_match.json']
    for part in local['parts']:
        record=read(DOCS/(part['factors']+'_training.json'))
        assert record['factors_sha256']==part['factors_sha256']==sha(BUILD/(part['factors']+'_factors.npz'))
        assert not record['public_delay_labels_used'] and record['geometry_training_rows_only'] and record['scratch_removed']
        assert record['architecture_sha256']==sha(BUILD/'oracle/oracle_arch.txt')
        assert part['direct']==record.get('direct',False)
        evidence.append(part['factors']+'_training.json')
    assert {'-DEST_LOCAL','-DEST_WALL','-DEST_FAST_ROUND'}<=set(build['compiler_flags']) and '-DEST_TRACE' not in build['compiler_flags']
    assert binary.stat().st_size==build['binary_bytes']==packaged['binary_bytes']
    assert package.stat().st_size==packaged['zip_bytes']<100_000_000
    with zipfile.ZipFile(package) as z:
        assert z.namelist()==['bin/estimate'] and z.getinfo('bin/estimate').external_attr>>16==0o100755
        assert hashlib.sha256(z.read('bin/estimate')).hexdigest()==bs
    payload=check_payload(binary,build)
    with binary.open('rb') as s:
        s.seek(-112,2);first=struct.unpack('<Q',s.read(16)[8:])[0];s.seek(first-32);desc=s.read(32)
        assert desc[:8] in (b'PGLCF001',b'PGLCZ001');offset,size,reserved=struct.unpack('<3Q',desc[8:])
        assert offset+size==first-32;s.seek(offset);embedded=s.read(size)
        if desc[:8]==b'PGLCZ001':
            compression=read(DOCS/(wall['base']+'_local_compression.json'))
            assert compression['exact_byte_roundtrip'] and compression['raw_sha256']==ls
            assert hashlib.sha256(embedded).hexdigest()==build['local_compressed_sha256']==compression['compressed_sha256']
            assert reserved==local['asset_bytes'] and size==compression['compressed_bytes']
            embedded=lzma.decompress(embedded);evidence.append(wall['base']+'_local_compression.json')
        else:assert not reserved and size==local['asset_bytes']
        assert hashlib.sha256(embedded).hexdigest()==ls
    assert ref['rows']==1_000_000 and ref['python_cpp_mismatch_count']==ref['stage8_reference_mismatch_count']==0
    assert ref['both_branches_reconstructed'] and ref['float32_fma_emulated']
    assert ref['reference_sha256']==sha(BUILD/(name+'_independent_reference.npy'))
    assert np.array_equal(np.load(BUILD/(name+'_independent_reference.npy')),np.load(BUILD/(name+'_prediction.npy')))
    assert ref['output_sha256']==analysis['output_sha256']==sha(BUILD/(name+'_result.csv'))
    assert analysis['input_sha256']==sha(DATA/'data/delay_estimate_ans.csv')
    for key in ('valid','test'):
        assert analysis[key]==wall[key]==ref['splits'][key]['updated']
        assert analysis[key]['rows']==50000 and analysis[key]['accuracy']>=98
    assert wall['position_assess']==ref['splits']['position_assess']['updated'] and wall['position_assess']['accuracy']>=98
    oracle=ref['oracle'];assert oracle['python_cpp_mismatch_count']==0
    for key,path in [('truth','oracle_answers.csv'),('request','oracle_request.csv'),('spec','oracle_spec.json')]:
        assert oracle[key+'_sha256']==sha(BUILD/path)
    assert oracle['output_sha256']==sha(BUILD/(name+'_oracle.csv'))
    assert validation['all_passed'];checks={}
    for suffix,digest in validation['reports'].items():
        path=DOCS/(name+'_'+suffix+'.json');assert sha(path)==digest;checks[suffix]=read(path);evidence.append(path.name)
    assert len(checks['integration']['interface_checks'])==11
    assert checks['contract']['all_passed'] and len(checks['contract']['checks'])==11
    assert checks['standalone']['standalone_zip_test']=='passed' and checks['standalone']['binary_sha256']==bs
    assert bench['runs']==5 and bench['rows']>=10_000_000 and bench['cpu_limit']==1 and bench['memory_limit_bytes']==2_000_000_000
    comparison={};rows=bench['rows']
    for candidate,measured in bench['candidates'].items():
        ca=read(DOCS/(candidate+'_analysis.json'))
        assert measured['binary_sha256']==sha(SOLUTION_ROOT/'bin'/('estimate_'+candidate))
        assert measured['reference_1m_sha256']==ca['output_sha256'] and len(measured['runs'])==5
        hashes=set()
        for run in measured['runs']:
            assert run['rows']==rows and run['repeated_output_verified'] and run['max_observed_threads']==1
            assert run['container_peak_bytes']<2_000_000_000 and run['cpu_seconds']*1_000_000_000/rows<1800
            hashes.add(run['output_sha256'])
        assert len(hashes)==1
        cpu=sum(r['cpu_seconds'] for r in measured['runs'])/5*1_000_000_000/rows
        assert abs(cpu-measured['projected_1b_cpu_seconds'])<1e-7
        acc=ca['test']['accuracy'];comparison[candidate]=dict(test_accuracy=acc,projected_1b_cpu_seconds=cpu,projected_total=.8*acc+20-cpu/120)
    current=comparison[name];stage8=comparison['displacement_cal16'];incumbent=comparison['stage10_local96w64']
    joint=current['projected_total']>=stage8['projected_total'] and current['projected_total']>=incumbent['projected_total']
    if args.retain:
        previous=read(DOCS/'stage10_accuracy_manifest.json')
        if (DOCS/'stage11_accuracy_manifest.json').exists():previous=read(DOCS/'stage11_accuracy_manifest.json')
        assert current['test_accuracy']>=previous['test_accuracy'] and analysis['valid']['accuracy']>=previous['validation_accuracy']
    else:assert joint,('joint score gate failed',current,stage8,incumbent)
    backups={}
    for filename in ('stage8_manifest.json','stage9_manifest.json','stage9_accuracy_manifest.json','stage10_manifest.json','stage10_accuracy_manifest.json'):
        old=read(DOCS/filename);oldname=old.get('candidate','displacement_cal16')
        oldbinary=SOLUTION_ROOT/'bin'/('estimate_'+oldname);oldzip=SOLUTION_ROOT/(oldname+'_submission.zip')
        assert sha(oldbinary)==old['package']['binary_sha256'] and sha(oldzip)==old['package']['zip_sha256']
        backups[filename]=sha(DOCS/filename)
    measured=bench['candidates'][name];delta=current['projected_total']-stage8['projected_total']
    manifest=dict(stage=11,date='2026-10-21',candidate=name,retained_alternative=args.retain,
                  binary='solution/bin/estimate_'+name if args.retain else 'solution/bin/estimate',
                  submission='solution/'+name+'_submission.zip' if args.retain else 'solution/submission.zip',
                  validation_accuracy=analysis['valid']['accuracy'],**current,full_public_accuracy=analysis['full_public']['accuracy'],
                  position_assess=wall['position_assess'],reserved_48=oracle['reserved_48'],python_cpp_mismatch_count=0,
                  comparison_same_batch=comparison,joint_gate_passed=joint,projected_total_delta_vs_stage8=delta,
                  target_98_25_and_delta_point2_achieved=analysis['valid']['accuracy']>=98.25 and current['test_accuracy']>=98.25 and delta>=.2,
                  benchmark_rows=rows,max_container_bytes=measured['max_container_bytes'],max_threads=1,
                  cpu_budget_seconds=1800,package=packaged,build=build,embedded_assets=payload,local=local,wall=wall,
                  evidence_sha256={p:sha(DOCS/p) for p in sorted(set(evidence))},preserved_manifest_sha256=backups,
                  source_sha256={p:sha(SOLUTION_ROOT/p) for p in ['src/displacement.cpp','src/displacement_local_runtime.hpp','src/displacement_wall_runtime.hpp',
                    *['tools/'+p for p in ('displacement_run.py','displacement_local_train.py','displacement_tile_train.py','displacement_relative_refine.py','displacement_partition_export.py','displacement_partition_screen.py','displacement_tile_select.py','displacement_local_compress.py','displacement_local_verify.py','release_stage11.py')]]},
                  official_score=False,full_1b_completed=False,
                  limitations=['Historical public regression slices; no blind or official accuracy claim',
                               'CPU extrapolated from repeated input; not full 1B',
                               'Train source-coordinate exclusion applies only to new parameters, not inherited baseline'])
    if not args.retain:
        shutil.copy2(binary,SOLUTION_ROOT/'bin/estimate');shutil.copy2(package,SOLUTION_ROOT/'submission.zip')
        assert sha(SOLUTION_ROOT/'bin/estimate')==bs and sha(SOLUTION_ROOT/'submission.zip')==zs
    dest=DOCS/('stage11_accuracy_manifest.json' if args.retain else 'stage11_manifest.json')
    dest.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ('candidate','validation_accuracy','test_accuracy','projected_total','projected_total_delta_vs_stage8','joint_gate_passed','target_98_25_and_delta_point2_achieved')},indent=2))


if __name__=='__main__':main()
