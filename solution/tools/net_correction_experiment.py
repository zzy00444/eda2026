"""Fit small source-net corrections without changing deployed artifacts.

The default predictor is reconstructed from the published LightGBM models in
original public CSV order.  Coefficients use only the fixed rng7 training rows;
validation chooses one candidate before the test score is computed once.

Only horizontally crossable Block records enter landing geometry.  In
particular, the un-crossable Block 0 must never be treated as a wall tunnel.
All output is isolated under build/net_correction_experiment/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack

from experiment import BUILD, DATA, competition_score
from topology_experiment import load_features, split, topology_features


def net_correction_features(x: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Derive gap and landing terms entirely from public architecture JSON.

    For a source net with displacement (nx,ny), l is its nominal landing.  A
    horizontal net whose landing falls inside a horizontally crossable Block
    skips that Block's missing SRBs, extending l.x by its inclusive width.

    gap_excess = gap(s,l) + gap(l,t) - gap(s,t).
    Four one-hot wall features contain the configured crossing delay, rather
    than the other implementation's fitted 117/292/371/481 ps constants.
    """
    arch = DATA / "arch"
    ports = json.loads((arch / "SRB_Port.json").read_text(encoding="utf-8"))["Port"]
    names = [p["Name"] for p in ports]
    ids = {n: i for i, n in enumerate(names)}
    nets = json.loads((arch / "SRB_Net.json").read_text(encoding="utf-8"))["Nets"]
    gap = json.loads((arch / "SRB_Gap.json").read_text(encoding="utf-8"))["Gap"]
    blocks = [b for b in gap["Block"] if b["horizontal crossable"]]
    ndx = np.zeros(len(names), dtype=np.int32)
    ndy = ndx.copy()
    for net in nets:
        ndx[ids[net["from"]]] = net["delta x"]
        ndy[ids[net["from"]]] = net["delta y"]
    sx, sy, tx, ty = [x[c].to_numpy(np.int32) for c in ("sx", "sy", "tx", "ty")]
    sp = x.source_port.astype(np.int32).to_numpy()
    tp = x.target_port.astype(np.int32).to_numpy()
    nx, ny = ndx[sp], ndy[sp]
    nominal_x, nominal_y = sx + nx, sy + ny
    landing_x = nominal_x.copy()
    wall = np.zeros((len(x), 4), dtype=np.float64)
    skip = np.zeros(len(x), dtype=bool)
    # Group bands sharing a corridor: a door is a y in this interval not
    # covered by ANY horizontal-crossable block in that corridor.
    corridors = sorted({(b["left"], b["right"]) for b in blocks})
    for left, right in corridors:
        bands = [b for b in blocks if b["left"] == left and b["right"] == right]
        max_y = max(int(sy.max()), int(ty.max()), max(b["upper"] for b in bands))
        door = np.ones(max_y + 1, dtype=np.int8)
        for b in bands:
            door[b["lower"]:b["upper"] + 1] = 0
        door_prefix = np.r_[0, np.cumsum(door, dtype=np.int32)]
        ylo, yhi = np.minimum(sy, ty), np.maximum(sy, ty)
        has_door = door_prefix[yhi + 1] > door_prefix[ylo]
        for b in bands:
            hit = ((ny == 0) & (nx != 0) &
                   (sy >= b["lower"]) & (sy <= b["upper"]) &
                   (nominal_x >= left) & (nominal_x <= right) &
                   (((sx < left) & (nx > 0)) | ((sx > right) & (nx < 0))))
            if not hit.any():
                continue
            width = right - left + 1
            landing_x[hit] += np.sign(nx[hit]) * width
            skip |= hit
            # Whether landing -> target crosses the corridor is the relevant
            # question after the source net itself has crossed it.
            cross_after = (((landing_x < left) & (tx > right)) |
                           ((tx < left) & (landing_x > right)))
            corridor_target = (tx >= left) & (tx <= right)
            delay = float(b["horizontal cross delay"])
            wall[hit & cross_after & has_door, 0] += delay
            wall[hit & cross_after & ~has_door, 1] += delay
            wall[hit & ~cross_after & ~corridor_target, 2] += delay
            wall[hit & ~cross_after & corridor_target, 3] += delay

    def segment(xa, ya, xb, yb):
        total = np.zeros(len(x), dtype=np.float64)
        for line in gap["Line"]:
            aa, bb = (xa, xb) if line["direction"] == "vertical" else (ya, yb)
            total += line["delay"] * ((np.minimum(aa, bb) <= line["site"]) &
                                       (line["site"] < np.maximum(aa, bb)))
        return total

    direct = segment(sx, sy, tx, ty)
    nominal_gap = (segment(sx, sy, nominal_x, nominal_y) +
                   segment(nominal_x, nominal_y, tx, ty) - direct)
    landing_gap = (segment(sx, sy, landing_x, nominal_y) +
                   segment(landing_x, nominal_y, tx, ty) - direct)
    f = pd.DataFrame({
        "gap_landing_excess": landing_gap,
        "wall_return_with_door": wall[:, 0],
        "wall_return_without_door": wall[:, 1],
        "wall_target_across": wall[:, 2],
        "wall_target_in_corridor": wall[:, 3],
    })
    same = (sx == tx) & (sy == ty) & (sp == tp)
    f.loc[same] = 0
    details = {
        "horizontal_crossable_block_ids": [b["id"] for b in blocks],
        "excluded_block_ids": [b["id"] for b in gap["Block"] if not b["horizontal crossable"]],
        "nominal_gap_nonzero_rows": int(np.count_nonzero(nominal_gap)),
        "landing_gap_nonzero_rows": int(np.count_nonzero(landing_gap)),
        "block_landing_skip_rows": int(skip.sum()),
        "gap_quantiles": np.quantile(landing_gap, [0, .5, .9, .99, 1]).tolist(),
        "wall_class_rows": dict(zip(f.columns[1:], map(int, (wall != 0).sum(axis=0)))),
        "formula": "gap(s,landing)+gap(landing,t)-gap(s,t); landing.x += sign(net.dx)*(right-left+1) when nominal landing hits a horizontal-crossable block",
    }
    return f, details


def predictor(threads: int, short_trees: int = 25, fast_trees: int = 18,
              refine_distance: int = 100, cascade_trees: int = 10):
    """Recreate the actual candidate; cached features retain categorical codes."""
    x, y = load_features()
    x, _ = topology_features(x)
    primary = lgb.Booster(model_file=str(BUILD / "s3_topology_lr32_511l_25r.txt"))
    base = x.architecture_delay.to_numpy(np.float64)
    distance = x.manhattan.to_numpy(np.int32)
    if not (0 <= fast_trees <= short_trees <= primary.num_trees()):
        raise ValueError("Require 0 <= fast-trees <= short-trees <= primary tree count")
    if not (0 <= cascade_trees <= 10):
        raise ValueError("cascade-trees must be between zero and ten")
    # LightGBM num_iteration=0 means ALL trees, so an architecture-only
    # configuration must explicitly bypass its predictor.
    full = (primary.predict(x, num_iteration=short_trees, num_threads=threads)
            if short_trees else np.zeros(len(x), dtype=np.float64))
    fast = (primary.predict(x, num_iteration=fast_trees, num_threads=threads)
            if fast_trees else np.zeros(len(x), dtype=np.float64))
    correction = np.where(distance < refine_distance, full, fast)
    if cascade_trees:
        cascade = lgb.Booster(model_file=str(BUILD / "medium_cascade_127l_10r.txt"))
        c = cascade.predict(x, num_iteration=cascade_trees, num_threads=threads)
        correction += np.where((distance >= 100) & (distance < 300), c, 0)
    raw = base + correction
    same = ((x.sx.to_numpy() == x.tx.to_numpy()) & (x.sy.to_numpy() == x.ty.to_numpy()) &
            (x.source_port.astype(int).to_numpy() == x.target_port.astype(int).to_numpy()))
    raw[same] = 0
    return x, np.asarray(y, dtype=np.float64), raw


def weighted_l1(features, residual, truth):
    """Exact weighted L1 linear fit over affected training rows (HiGHS LP)."""
    active = np.any(features != 0, axis=1)
    a, r, y = features[active], residual[active], truth[active]
    n, k = a.shape
    aa = coo_matrix(a).tocsr()
    ident = eye(n, format="csr")
    constraint = vstack((hstack((aa, -ident)), hstack((-aa, -ident))), format="csr")
    result = linprog(np.r_[np.zeros(k), 1.0 / np.maximum(y, 200.0)],
                     A_ub=constraint, b_ub=np.r_[r, -r],
                     bounds=[(-2.0, 3.0)] * k + [(0, None)] * n,
                     method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:k], {"active_training_rows": int(n), "optimizer": "scipy.optimize.linprog/highs", "weighted_l1": float(result.fun)}


def training_score_fit(f, y, raw, initial, rounds=3):
    """Small bounded coordinate search using training scores only."""
    active = np.any(f != 0, axis=1)
    a, yy, pp = f[active], y[active], raw[active]
    coef = np.array(initial, dtype=np.float64)
    for step, radius in ((.1, 2.0), (.02, .2)):
        for _ in range(rounds):
            before = coef.copy()
            for col in range(a.shape[1]):
                others = pp + a @ coef - a[:, col] * coef[col]
                values = np.arange(max(-2, coef[col] - radius), min(3, coef[col] + radius) + step/2, step)
                best = max(values, key=lambda v: competition_score(yy, others + a[:, col] * v))
                coef[col] = float(best)
            if np.allclose(before, coef):
                break
    return coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="net_correction_experiment")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--short-trees", type=int, default=25)
    ap.add_argument("--fast-trees", type=int, default=18)
    ap.add_argument("--refine-distance", type=int, default=100)
    ap.add_argument("--cascade-trees", type=int, default=10,
                    help="Zero disables the medium-distance cascade")
    ap.add_argument("--l1-only", action="store_true",
                    help="Fit only weighted L1 and compare it with the baseline")
    ap.add_argument("--cpu-1b", type=float, default=(15.831966 + 15.992175) * 50,
                    help="Current measured-10M CPU extrapolation, seconds for 1B")
    args = ap.parse_args()
    out = BUILD / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x, y, raw = predictor(args.threads, args.short_trees, args.fast_trees,
                           args.refine_distance, args.cascade_trees)
    f, structure = net_correction_features(x)
    va, te, tr = split(len(y))
    assert not np.intersect1d(tr, va).size and not np.intersect1d(tr, te).size
    assert np.array_equal(x.index.to_numpy(), np.arange(len(x)))
    full = f.to_numpy(np.float64)
    ft, yt, pt = full[tr], y[tr], raw[tr]
    grouped = np.c_[ft[:, 0], ft[:, 1:].sum(axis=1)]
    print("rows", len(y), "active", int(np.any(full != 0, axis=1).sum()), flush=True)
    candidates = [{"name": "baseline", "coefficients": np.zeros(full.shape[1]).tolist()}]
    if not args.l1_only:
        # Fit one gap coefficient and one shared wall coefficient, then expand it.
        c2 = training_score_fit(grouped, yt, pt, [0.0, 0.0])
        candidates.append({"name": "score_two_coefficients", "coefficients": np.r_[c2[0], np.repeat(c2[1], 4)].tolist()})
        print("two coefficients", c2, flush=True)
    cl1, fitinfo = weighted_l1(ft, yt - pt, yt)
    candidates.append({"name": "weighted_l1_five_coefficients", "coefficients": cl1.tolist(), "fit": fitinfo})
    print("L1", cl1, flush=True)
    if not args.l1_only:
        cs = training_score_fit(ft, yt, pt, cl1)
        candidates.append({"name": "score_five_coefficients", "coefficients": cs.tolist()})
        # Shrinkage is a validation decision, like selecting regularisation strength.
        for scale in (.5, .75):
            candidates.append({"name": f"score_five_shrink_{scale:g}", "coefficients": (cs * scale).tolist()})
    base_valid = competition_score(y[va], raw[va])
    for c in candidates:
        coef = np.asarray(c["coefficients"])
        c["training_score"] = competition_score(yt, pt + ft @ coef)
        c["validation_score"] = competition_score(y[va], raw[va] + full[va] @ coef)
        delta = c["validation_score"] - base_valid
        c["max_extra_1b_cpu_seconds_for_break_even"] = 96.0 * delta
        c["max_extra_nanoseconds_per_row_for_break_even"] = 96.0 * delta
        c["validation_total_if_zero_extra_cpu"] = .8*c["validation_score"] - args.cpu_1b/120 + 20
        print(c["name"], "valid", c["validation_score"], "extra_cpu_budget", 96*delta, flush=True)
    selected = max(candidates, key=lambda c: c["validation_score"])
    # This is the first and only test evaluation of a fitted correction.
    coefficients = np.asarray(selected["coefficients"])
    corrected_test = raw[te] + full[te] @ coefficients
    base_test = competition_score(y[te], raw[te])
    test_score = competition_score(y[te], corrected_test)
    report = {
        "split": "original_public_rows_rng7_train=order[:300000]+order[400000:]_validation=order[300000:350000]_test=order[350000:400000]",
        "rows": {"train": len(tr), "validation": len(va), "test": len(te)},
        "base_model": {"primary": "s3_topology_lr32_511l_25r", "short_trees": args.short_trees, "fast_trees": args.fast_trees,
                       "short_distance": args.refine_distance, "cascade": "medium_cascade_127l_10r" if args.cascade_trees else None,
                       "cascade_trees": args.cascade_trees, "cascade_band": [100, 300] if args.cascade_trees else None},
        "feature_names": f.columns.tolist(), "structure": structure,
        "candidates_training_and_validation_only": candidates,
        "selected": selected["name"], "coefficients": coefficients.tolist(),
        "test_once": {"baseline_accuracy": base_test, "selected_accuracy": test_score,
                      "accuracy_gain": test_score-base_test,
                      "max_extra_1b_cpu_seconds_for_break_even": 96*(test_score-base_test)},
        "cpu_reference_1b_seconds": args.cpu_1b,
        "total_score_formula": ".8*accuracy-CPU_seconds/120+20 (consistency=100, CPU<1800)",
        "runtime_status": "No C++ runtime timing: accuracy gain is actionable only below the reported extra CPU budget.",
        "prediction_rounding": "numpy.rint(base architecture + primary/cascade residual + net correction), then clamp to zero; same pins forced zero",
        "seconds": time.perf_counter()-started,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected["name"], "test_once": report["test_once"], "report": str(out / "metrics.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
