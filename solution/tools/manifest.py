"""Record data/source fingerprints and exact training split for reproducibility."""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import lightgbm as lgb
from experiment import ROOT, DATA, BUILD

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()

order=np.random.default_rng(7).permutation(1_000_000)
train=np.concatenate((order[:300_000],order[400_000:]))[:600_000]
valid=order[300_000:350_000]
test=order[350_000:400_000]
assert not np.intersect1d(train,valid).size
assert not np.intersect1d(train,test).size
assert not np.intersect1d(valid,test).size
np.savez_compressed(BUILD / "fixed_split.npz",train=train,valid=valid,test=test)
files=list((DATA / "arch").glob("*.json"))+[DATA / "data/delay_estimate_ans.csv"]
files+=list((ROOT / "solution/src").glob("*.cpp"))+list((ROOT / "solution/tools").glob("*.py"))
files+=list((ROOT / "solution/tools").glob("*.sh"))
files += [BUILD / "arch_fast25.txt",BUILD / "generated/model.hpp",BUILD / "generated/architecture.bin"]
manifest={"python":platform.python_version(),"packages":{"numpy":np.__version__,"pandas":pd.__version__,
    "scipy":scipy.__version__,"lightgbm":lgb.__version__},
    "seed":7,"train_rows":len(train),"valid_rows":len(valid),"test_rows":len(test),
    "training_command":"python -u tools/experiment.py --train-size 600000 --rounds 25 --leaves 63 --baseline architecture --threads 4 --learning-rate 0.26 --name arch_fast25",
    "export_command":"python -u tools/export_cpp.py --model arch_fast25 --trees 25 --float",
    "model":"arch_fast25","trees":25,"float32":True,
    "sha256":{str(p.relative_to(ROOT)):sha(p) for p in files}}
(BUILD / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print("Saved manifest and disjoint 600k/50k/50k split")
