"""Refit the five net-correction coefficients against a folded base predictor.

The stage-5 coefficients were fitted to the stage-5 one-tree residual.  A
different base model changes that residual, so the coefficients have to be
refitted on the training split before they are compiled in.  Validation picks
the candidate; the test split is evaluated once.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

# scripts in this directory import each other and project_paths directly
from project_paths import BUILD  # noqa: E402
from experiment import competition_score  # noqa: E402
from topology_experiment import load_features, topology_features  # noqa: E402
from third_stage_experiment import split  # noqa: E402
from net_correction_experiment import net_correction_features, weighted_l1  # noqa: E402
from fold_features import foldable_matrix  # noqa: E402

parser_model = sys.argv[1] if len(sys.argv) > 1 else "foldable_lr08_300r"
trees = int(sys.argv[2]) if len(sys.argv) > 2 else 300
output = sys.argv[3] if len(sys.argv) > 3 else "net_correction_folded.json"

x, y = load_features()
x, _ = topology_features(x)
base = x.architecture_delay.to_numpy().copy()
matrix, feature_names, _ = foldable_matrix(x, base)
model = lgb.Booster(model_file=str(BUILD / f"{parser_model}.txt"))
assert model.feature_name() == feature_names, "foldable feature order mismatch"
correction = model.predict(matrix, num_iteration=trees, num_threads=8)
new_base = base + correction

features_net, details = net_correction_features(x)
f = features_net.to_numpy(np.float64)
va, te, tr = split(len(y))

baseline_valid = competition_score(y[va], new_base[va])
print("folded base: valid %.6f" % baseline_valid, flush=True)
coefficients, info = weighted_l1(f[tr], y[tr] - new_base[tr], y[tr])
valid = competition_score(y[va], new_base[va] + f[va] @ coefficients)
test_base = competition_score(y[te], new_base[te])
test = competition_score(y[te], new_base[te] + f[te] @ coefficients)
print("L1 coefficients", np.round(coefficients, 6).tolist(), flush=True)
print("valid %.6f (base %.6f)  test %.6f (base %.6f)" % (valid, baseline_valid, test, test_base), flush=True)
report = {
    "base_model": {"primary": parser_model, "trees": trees},
    "coefficients": coefficients.tolist(),
    "fit": info,
    "base_validation": baseline_valid,
    "corrected_validation": valid,
    "base_test": test_base,
    "corrected_test": test,
    "structure": details,
    "selected": "weighted_l1_five_coefficients",
}
(BUILD / output).write_text(json.dumps(report, indent=2), encoding="utf-8")
