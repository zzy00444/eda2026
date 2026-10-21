"""Validate real C++ branches on valid only before freezing final candidates."""
import argparse
import json
import numpy as np
import pandas as pd
from displacement_analyze import OUT,points
from displacement_assets import sha
from displacement_local_verify import load_base,load_local,predict
from displacement_residual import indices,make_features
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('names',nargs='+');args=ap.parse_args()
    g=np.load(OUT/'geometry.npz');_,va,_,_=indices(g);vg={k:g[k][va] for k in g.files};base=load_base()
    wall=make_features(vg,np.zeros(len(va))).landing_wall.to_numpy()>0
    fixed=np.load(OUT/'stage10_local96w64_prediction.npy')[va]
    for name in args.names:
        path=OUT/(name+'_result.csv');actual=pd.read_csv(path,usecols=['delay']).delay.to_numpy(np.int32)[va]
        ref,_=predict(vg,base,load_local(OUT/(name+'.local')))
        assert np.array_equal(ref,actual),(name,np.count_nonzero(ref!=actual))
        raw_accuracy=float(points(vg['y'],actual).mean());actual[wall]=fixed[wall]
        package=REPO_ROOT/'solution'/(name+'_submission.zip')
        report=dict(candidate=name,valid_no_wall_accuracy=raw_accuracy,valid_fixed_wall_accuracy=float(points(vg['y'],actual).mean()),
                    valid_python_cpp_mismatches=0,real_both_branches=True,wall_outputs_fixed=True,test_labels_used=False,
                    binary_sha256=sha(REPO_ROOT/'solution/bin'/('estimate_'+name)),output_sha256=sha(path),
                    zip_bytes=package.stat().st_size if package.exists() else None)
        (REPO_ROOT/'docs'/(name+'_screen_reference.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
