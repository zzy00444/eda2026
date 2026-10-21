"""Verify frozen residual references against deployed outputs, without fitting."""
import argparse
import json

import numpy as np
import pandas as pd

from displacement_analyze import OUT, describe, points
from displacement_assets import sha
from displacement_generalization import mean_delta_interval
from displacement_residual import indices
from project_paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    geometry = np.load(OUT / 'geometry.npz')
    base = np.load(OUT / 'displacement_cal16_prediction.npy')
    y = geometry['y']
    train, valid, test, assess = indices(geometry)
    position = geometry['sx'].astype(np.int64)*550+geometry['sy']
    oracle_path = OUT / 'oracle_answers.csv'
    oracle = pd.read_csv(oracle_path)
    report = {
        'fit_rows': len(train), 'position_assess_rows': len(assess),
        'baseline': 'displacement_cal16',
        'oracle_truth_sha256': sha(oracle_path),
        'oracle_request_sha256': sha(OUT / 'oracle_request.csv'),
        'oracle_spec_sha256': sha(OUT / 'oracle_spec.json'),
        'caveat': 'Historical rng7 regression slices, not blind; only new residual parameters exclude held source coordinates',
        'candidates': {},
    }
    for name in args.names:
        pred_path = OUT / (name+'_prediction.npy')
        ref_path = OUT / (name+'_reference.npy')
        pred, ref = np.load(pred_path), np.load(ref_path)
        assert pred.shape == ref.shape == y.shape
        mismatch = np.flatnonzero(pred != ref)
        assert len(mismatch) == 0, (name, mismatch[:10].tolist())
        deployed = pd.read_csv(OUT / (name+'_oracle.csv'))
        assert deployed.From.equals(oracle.From) and deployed.To.equals(oracle.To)
        item = {
            'binary_sha256': sha(REPO_ROOT / 'solution/bin' / ('estimate_'+name)),
            'output_sha256': sha(OUT / (name+'_result.csv')),
            'reference_sha256': sha(ref_path), 'prediction_sha256': sha(pred_path),
            'export_report_sha256': sha(REPO_ROOT / 'docs' / (name+'_export.json')),
            'rows': len(pred), 'python_cpp_mismatch_count': len(mismatch),
            'max_absolute_difference': int(np.max(np.abs(pred.astype(np.int64)-ref))),
            'position_assess': describe(y[assess], pred[assess]),
            'oracle_output_sha256': sha(OUT / (name+'_oracle.csv')),
            'official_eight': describe(oracle.delay.to_numpy()[:8], deployed.delay.to_numpy()[:8]),
            'reserved_48': describe(oracle.delay.to_numpy()[8:], deployed.delay.to_numpy()[8:]),
        }
        for split_name, ix in [('valid', valid), ('test', test), ('position_assess', assess)]:
            item['paired_'+split_name+'_delta_vs_stage8'] = mean_delta_interval(
                points(y[ix], pred[ix])-points(y[ix], base[ix]), position[ix])
        report['candidates'][name] = item
    path = REPO_ROOT / 'docs/stage9_generalization.json'
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
