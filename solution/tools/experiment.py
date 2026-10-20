"""Train and evaluate an offline delay estimator on public examples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


from project_paths import BUILD, DATA, REPO_ROOT as ROOT


def competition_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    predicted = np.maximum(0, np.rint(predicted))
    points = np.zeros(len(actual), dtype=np.float64)
    nonzero = actual != 0
    points[nonzero] = 100 * (
        1 - np.tanh(4 * np.abs(predicted[nonzero] - actual[nonzero]) / actual[nonzero])
    )
    points[~nonzero] = 100 * (predicted[~nonzero] == 0)
    return float(points.mean())


def pin_parts(series: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    parts = series.str.extract(r"^SRB_(\d+)_(\d+)/(.+)$")
    if parts.isna().any().any():
        raise ValueError("A pin name does not match SRB_x_y/port")
    return (
        parts[0].to_numpy(dtype=np.int16),
        parts[1].to_numpy(dtype=np.int16),
        parts[2],
    )


def port_features(ports: pd.Series, names: list[str], prefix: str) -> dict[str, np.ndarray]:
    ids = pd.Categorical(ports, categories=names).codes.astype(np.int16)
    if np.any(ids < 0):
        raise ValueError("A sample references a port absent from SRB_Port.json")
    family = ports.str.replace(r"\[\d+\]$", "", regex=True)
    families = sorted({name.split("[")[0] for name in names})
    family_ids = pd.Categorical(family, categories=families).codes.astype(np.int16)
    bit = ports.str.extract(r"\[(\d+)\]$")[0].fillna("-1").to_numpy(dtype=np.int8)
    return {
        f"{prefix}_port": ids,
        f"{prefix}_family": family_ids,
        f"{prefix}_bit": bit,
    }


def features(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    sx, sy, sp = pin_parts(frame["From"])
    tx, ty, tp = pin_parts(frame["To"])
    dx = tx - sx
    dy = ty - sy
    ax = np.abs(dx)
    ay = np.abs(dy)

    arch = DATA / "arch"
    names = [p["Name"] for p in json.loads((arch / "SRB_Port.json").read_text(encoding="utf-8"))["Port"]]
    gaps = json.loads((arch / "SRB_Gap.json").read_text(encoding="utf-8"))["Gap"]
    gap_x = np.zeros(len(frame), dtype=np.int16)
    gap_y = np.zeros(len(frame), dtype=np.int16)
    xlo, xhi = np.minimum(sx, tx), np.maximum(sx, tx)
    ylo, yhi = np.minimum(sy, ty), np.maximum(sy, ty)
    for line in gaps["Line"]:
        site = line["site"]
        if line["direction"] == "vertical":
            gap_x += line["delay"] * ((xlo <= site) & (site < xhi))
        else:
            gap_y += line["delay"] * ((ylo <= site) & (site < yhi))

    result = pd.DataFrame(
        {
            "sx": sx,
            "sy": sy,
            "tx": tx,
            "ty": ty,
            "dx": dx,
            "dy": dy,
            "ax": ax,
            "ay": ay,
            "manhattan": ax + ay,
            "max_axis": np.maximum(ax, ay),
            "gap_x": gap_x,
            "gap_y": gap_y,
            "same_row": (sy == ty).astype(np.int8),
            "same_column": (sx == tx).astype(np.int8),
            **port_features(sp, names, "source"),
            **port_features(tp, names, "target"),
        }
    )
    # Every east-west crossing of these hard blocks may alter the shortest route.
    for block in gaps["Block"]:
        if block["horizontal crossable"]:
            row = ((block["lower"] <= sy) & (sy <= block["upper"])) | (
                (block["lower"] <= ty) & (ty <= block["upper"])
            )
            crossed = (xlo < block["left"]) & (xhi > block["right"])
            result[f"cross_block_{block['id']}"] = (row & crossed).astype(np.int8)

    nets = json.loads((arch / "SRB_Net.json").read_text(encoding="utf-8"))["Nets"]
    net_dx = {net["from"]: net["delta x"] for net in nets}
    net_dy = {net["from"]: net["delta y"] for net in nets}
    first_dx = sp.map(net_dx).fillna(0).to_numpy(dtype=np.int16)
    first_dy = sp.map(net_dy).fillna(0).to_numpy(dtype=np.int16)
    result["first_dx"] = first_dx
    result["first_dy"] = first_dy
    remaining_x, remaining_y = dx - first_dx, dy - first_dy
    result["remaining_x"] = remaining_x
    result["remaining_y"] = remaining_y
    result["remaining_ax"] = np.abs(remaining_x)
    result["remaining_ay"] = np.abs(remaining_y)
    for name, values, period in (
        ("dx", dx, 10), ("dy", dy, 12),
        ("ax", ax, 10), ("ay", ay, 12),
        ("remaining_x", remaining_x, 10), ("remaining_y", remaining_y, 12),
        ("remaining_ax", np.abs(remaining_x), 10),
        ("remaining_ay", np.abs(remaining_y), 12),
    ):
        result[f"{name}_mod"] = values % period
    result["sy_mod100"] = sy % 100
    result["ty_mod100"] = ty % 100

    for column in ("source_port", "source_family", "target_port", "target_family"):
        result[column] = result[column].astype("category")
    baseline = 309 + 12.7 * ax + 10.6 * ay + 0.93 * (gap_x + gap_y)
    return result, baseline


def stage3_features(x: pd.DataFrame, baseline: np.ndarray) -> pd.DataFrame:
    """Add cheap endpoint interactions that are also reproducible in C++."""
    result = x.copy()
    source_port = result["source_port"].cat.codes.to_numpy(dtype=np.int32)
    target_port = result["target_port"].cat.codes.to_numpy(dtype=np.int32)
    source_family = result["source_family"].cat.codes.to_numpy(dtype=np.int32)
    target_family = result["target_family"].cat.codes.to_numpy(dtype=np.int32)
    source_bit = result["source_bit"].to_numpy(dtype=np.int32)
    target_bit = result["target_bit"].to_numpy(dtype=np.int32)
    result["port_pair"] = source_port * 512 + target_port
    result["family_pair"] = source_family * 64 + target_family
    result["bit_pair"] = source_bit * 16 + target_bit
    result["direction"] = 2 * (result["dx"].to_numpy() < 0) + (result["dy"].to_numpy() < 0)
    distance = result["manhattan"].to_numpy(dtype=np.int32)
    result["distance_bucket"] = (distance >= 10) + (distance >= 30) + (distance >= 100) + (distance >= 300)
    linear = (309 + 12.7 * result["ax"].to_numpy() + 10.6 * result["ay"].to_numpy()
              + 0.93 * (result["gap_x"].to_numpy() + result["gap_y"].to_numpy()))
    result["architecture_minus_linear"] = baseline - linear
    block_columns = [c for c in result.columns if c.startswith("cross_block_")]
    result["cross_block_count"] = result[block_columns].sum(axis=1).astype(np.int8)
    result["source_region"] = result["sx"] // 10
    result["target_region"] = result["tx"] // 10
    result["same_region"] = (result["source_region"] == result["target_region"]).astype(np.int8)
    result["axis_dominant"] = (result["ax"].to_numpy() >= result["ay"].to_numpy()).astype(np.int8)
    for column in ("port_pair", "family_pair", "bit_pair", "direction", "distance_bucket",
                   "source_region", "target_region"):
        result[column] = result[column].astype("category")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=900_000)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--leaves", type=int, default=63)
    parser.add_argument("--name", default="periodic_v2")
    parser.add_argument("--baseline", choices=("linear", "architecture"), default="linear")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--stage3-features", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(DATA / "data" / "delay_estimate_ans.csv")
    if args.train_size > len(frame) - 100_000:
        parser.error("Training must exclude the fixed 50k validation and 50k test sets")
    x, baseline = features(frame)
    if args.baseline == "architecture":
        from architecture import predict
        baseline = predict(frame)
        x["architecture_delay"] = baseline
    if args.stage3_features:
        x = stage3_features(x, baseline)
    y = frame["delay"].to_numpy(dtype=np.float32)
    order = np.random.default_rng(7).permutation(len(frame))
    # Preserve the validation/test sets from the first completed 300k experiment.
    valid = order[300_000:350_000]
    test = order[350_000:400_000]
    train = np.concatenate((order[:300_000], order[400_000:]))[:args.train_size]
    print("examples", len(frame), "features", len(x.columns), flush=True)
    print("baseline_valid", competition_score(y[valid], baseline[valid]), flush=True)

    residual = y - baseline
    categorical = ["source_port", "source_family", "target_port", "target_family"]
    train_data = lgb.Dataset(
        x.iloc[train],
        label=residual[train],
        weight=1 / np.maximum(y[train], 200),
        categorical_feature=categorical,
        free_raw_data=False,
    )
    valid_data = lgb.Dataset(
        x.iloc[valid],
        label=residual[valid],
        weight=1 / np.maximum(y[valid], 200),
        categorical_feature=categorical,
        reference=train_data,
        free_raw_data=False,
    )
    started = time.perf_counter()
    model = lgb.train(
        {
            "objective": "regression_l1",
            "metric": "l1",
            "learning_rate": args.learning_rate,
            "num_leaves": args.leaves,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.9,
            "verbosity": -1,
            "num_threads": args.threads,
            "force_col_wise": True,
            "seed": 7,
        },
        train_data,
        num_boost_round=args.rounds,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )
    metrics = {"train_size": len(train), "valid_size": len(valid), "test_size": len(test),
               "split": "rng7_valid300000:350000_test350000:400000",
               "training_seconds": time.perf_counter() - started}
    BUILD.mkdir(parents=True, exist_ok=True)
    for name, indices in (("valid", valid), ("test", test)):
        prediction = baseline[indices] + model.predict(x.iloc[indices], num_threads=args.threads)
        print(
            name,
            "score",
            competition_score(y[indices], prediction),
            "mae",
            float(np.mean(np.abs(np.rint(prediction) - y[indices]))),
            flush=True,
        )
        metrics[name] = {"score": competition_score(y[indices], prediction),
                         "mae": float(np.mean(np.abs(np.rint(prediction) - y[indices])))}
        pd.DataFrame({"row": indices, "actual": y[indices], "prediction": prediction}).to_csv(
            BUILD / f"{args.name}_{name}.csv", index=False)
    print("best_iteration", model.best_iteration, flush=True)
    metrics["best_iteration"] = model.best_iteration
    metrics["baseline"] = args.baseline
    metrics["stage3_features"] = args.stage3_features
    model.save_model(BUILD / f"{args.name}.txt")
    (BUILD / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (BUILD / f"{args.name}_features.json").write_text(json.dumps(list(x.columns)), encoding="utf-8")


if __name__ == "__main__":
    main()
