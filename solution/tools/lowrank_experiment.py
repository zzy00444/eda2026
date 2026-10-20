"""Fast residual-model experiments for stage 3.

The script evaluates compact residual tables against the fixed RNG-7 split.
It intentionally keeps all fitted quantities in memory and writes only small
diagnostic npz files when requested.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


from project_paths import BUILD, DATA as DATA_ROOT

DATA = DATA_ROOT / "data"


def score(y: np.ndarray, p: np.ndarray) -> float:
    p = np.maximum(0, np.rint(p))
    nz = y != 0
    q = np.empty(y.size, dtype=np.float64)
    q[nz] = 100 * (1 - np.tanh(4 * np.abs(p[nz] - y[nz]) / y[nz]))
    q[~nz] = 100 * (p[~nz] == 0)
    return float(q.mean())


def load() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = pd.read_csv(DATA / "delay_estimate_ans.csv")
    a = np.load(BUILD / "architecture_prediction.npy").astype(np.float32)
    def parts(s: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = s.str.extract(r"^SRB_(\d+)_(\d+)/(.+)$")
        return (z[0].to_numpy(np.int16), z[1].to_numpy(np.int16),
                pd.factorize(z[2], sort=True)[0].astype(np.int32))
    sx, sy, sp0 = parts(df.From)
    tx, ty, tp0 = parts(df.To)
    # Factorize jointly so port ids are stable between source and target.
    names = pd.concat([df.From.str.extract(r"^SRB_\d+_\d+/(.+)$")[0],
                       df.To.str.extract(r"^SRB_\d+_\d+/(.+)$")[0]],
                      ignore_index=True)
    cats = pd.Categorical(names)
    sp = cats.codes[:len(df)].astype(np.int16)
    tp = cats.codes[len(df):].astype(np.int16)
    dx, dy = tx.astype(np.int16) - sx, ty.astype(np.int16) - sy
    y = df.delay.to_numpy(np.float32)
    z = {"sx": sx, "sy": sy, "tx": tx, "ty": ty, "sp": sp, "tp": tp,
         "dx": dx, "dy": dy, "y": y, "base": a}
    return df, z


def fixed_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(7).permutation(n)
    return order[300000:350000], order[350000:400000], np.r_[order[:300000], order[400000:]]


def group_mean(key: np.ndarray, values: np.ndarray, train: np.ndarray,
               fallback: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    key = np.asarray(key, dtype=np.int64)
    m = int(key.max()) + 1
    sums = np.bincount(key[train], weights=values[train], minlength=m)
    cnt = np.bincount(key[train], minlength=m)
    out = np.full(m, fallback, dtype=np.float32)
    np.divide(sums, cnt, out=out, where=cnt > 0)
    return out, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    _, z = load()
    y, b = z["y"], z["base"]
    r = y - b
    valid, test, train = fixed_split(len(y))
    print("baseline", score(y[valid], b[valid]), score(y[test], b[test]),
          "resid", float(r[train].mean()), float(r[train].std()), flush=True)

    def report(name: str, p: np.ndarray) -> None:
        print(name, "valid", score(y[valid], p[valid]),
              "test", score(y[test], p[test]),
              "mae", float(np.mean(np.abs(np.rint(p[test]) - y[test]))), flush=True)

    # Pair means are the cheapest useful correction and are also a lower bound
    # for any factor model with a free pair intercept.
    pair = z["sp"].astype(np.int64) * 512 + z["tp"]
    pm, pc = group_mean(pair, r, train)
    report("pair", b + pm[pair])

    # Add periodic displacement and coarse distance effects.  The product is
    # intentionally kept small enough for direct C++ table lookup.
    dxm, dym = z["dx"] % 10, z["dy"] % 12
    for form, key in [
        ("pair_dxdy_mod", pair * 120 + dxm * 12 + dym),
        ("pair_dxmod", pair * 10 + dxm),
        ("pair_dymod", pair * 12 + dym),
        ("pair_dir", pair * 4 + (2 * (z["dx"] < 0) + (z["dy"] < 0))),
    ]:
        gm, cnt = group_mean(key, r, train)
        # Shrink poorly supported groups toward the pair intercept.
        parent = pm[pair]
        n = cnt[key]
        w = n / (n + 8.0)
        corr = parent + w * (gm[key] - parent)
        report(form, b + corr)

    # Two-way additive model over pair and displacement; one sweep is a cheap
    # alternating least-squares approximation to a rank-1 interaction.
    keyd = dxm * 12 + dym
    dm, dc = group_mean(keyd, r, train)
    corr = pm[pair] + dm[keyd] - r[train].mean()
    report("pair_plus_mod", b + corr)

    # Back-fitted additive effects.  This is a compact alternative to trees:
    # each feature is one integer table lookup at inference time.  Effects are
    # fitted on residuals with a small count prior, so rare categories do not
    # memorize the public split.
    sx, sy, tx, ty = (z[k].astype(np.int64) for k in ("sx", "sy", "tx", "ty"))
    direction = 2 * (z["dx"] < 0) + (z["dy"] < 0)
    keys = [
        ("pair", pair, 12.0), ("sp", z["sp"], 30.0), ("tp", z["tp"], 30.0),
        ("bitpair", (z["sp"] % 16) * 16 + (z["tp"] % 16), 20.0),
        ("sx", sx, 40.0), ("tx", tx, 40.0),
        ("sy100", sy % 100, 40.0), ("ty100", ty % 100, 40.0),
        ("dx", z["dx"] + 120, 40.0), ("dy", z["dy"] + 552, 40.0),
        ("dxmod", dxm, 50.0), ("dymod", dym, 50.0),
        ("dir", direction, 50.0),
        ("distbucket", (np.abs(z["dx"]) + np.abs(z["dy"]) >= 10).astype(np.int64), 50.0),
    ]
    effects = [np.zeros(int(k.max()) + 1, np.float32) for _, k, _ in keys]
    fit = np.zeros(len(y), np.float32)
    prior = float(r[train].mean())
    for it in range(5):
        for j, (name, k, shrink) in enumerate(keys):
            # Remove this effect before estimating its new partial residual.
            fit -= effects[j][k]
            partial = r - fit
            sums = np.bincount(k[train], weights=partial[train], minlength=len(effects[j]))
            cnt = np.bincount(k[train], minlength=len(effects[j]))
            e = np.zeros_like(effects[j])
            np.divide(sums, cnt + shrink, out=e, where=cnt > 0)
            effects[j] = e
            fit += e[k]
        report(f"additive_{it+1}", b + fit)
    if args.save:
        np.savez(BUILD / "lowrank_additive.npz",
                 **{f"effect_{i}": e for i, e in enumerate(effects)},
                 **{f"key_{i}": k for i, (_, k, _) in enumerate(keys)},
                 names=np.array([n for n, _, _ in keys]))

    if args.save:
        np.savez(BUILD / "lowrank_parsed.npz", **z,
                 valid=valid, test=test, train=train)


if __name__ == "__main__":
    main()
