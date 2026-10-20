"""Fold a foldable correction into the architecture table and emit a C++ header.

Output (solution/build/<out>/): architecture.bin, architecture.S, model.hpp,
export.json.  Each table entry becomes

    table_scaled + round(12 * correction(key))

so the runtime prediction is

    (slopes . (dx,dy) + entry) / 12 + gap_x + gap_y + net_correction

which equals the model prediction plus the analytic terms, in one lookup.  The
table keeps its stage-5 size and the 32767 "unreachable pair" sentinel.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# scripts in this directory import each other and project_paths directly
from project_paths import BUILD, DATA  # noqa: E402
from experiment import features, stage3_features  # noqa: E402

SENTINEL = 32767
FOLDABLE = {"source_port", "source_family", "source_bit", "target_port", "target_family",
            "target_bit", "first_dx", "first_dy", "dx_mod", "dy_mod", "ax_mod", "ay_mod",
            "remaining_x_mod", "remaining_y_mod", "port_pair", "family_pair", "bit_pair",
            "direction", "table_scaled", "envelope_proxy"}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="foldable_lr08_300r")
parser.add_argument("--trees", type=int, default=300)
parser.add_argument("--out", default="generated_s7_fold")
parser.add_argument("--net-coefficients", default="")
args = parser.parse_args()

meta = json.loads((BUILD / "architecture_meta.json").read_text())
names = meta["ports"]
sources, targets = meta["sources"], meta["targets"]
slopes = np.array(meta["slopes_scaled12"])
table = np.load(BUILD / "architecture_table.npy")
assert table.dtype == np.int16 and table.shape == (4, len(sources), 120, len(targets))
n_src, n_tgt = len(sources), len(targets)

gaps = json.loads((DATA / "arch" / "SRB_Gap.json").read_text(encoding="utf-8"))["Gap"]
nets = json.loads((DATA / "arch" / "SRB_Net.json").read_text(encoding="utf-8"))["Nets"]
net_dx = {n["from"]: n["delta x"] for n in nets}
net_dy = {n["from"]: n["delta y"] for n in nets}
families = sorted({n.split("[")[0] for n in names})
family_of = {n: families.index(n.split("[")[0]) for n in names}
bit_of = {n: (int(n.split("[")[1][:-1]) if "[" in n else -1) for n in names}

# The model was trained on the codes pandas assigned to the categorical columns,
# so the fold has to reproduce those exact maps.
frame = pd.read_csv(DATA / "data" / "delay_estimate_ans.csv")
x, _ = features(frame)
x = stage3_features(x, np.zeros(len(frame)))
categorical = {c: list(x[c].cat.categories) for c in x.columns
               if isinstance(x[c].dtype, pd.CategoricalDtype)}
del x, frame


def code_map(feature):
    return {value: index for index, value in enumerate(categorical[feature])}


def encode(feature, values):
    lookup = code_map(feature)
    return np.array([lookup.get(v, -1) for v in values], dtype=np.float32)


model = lgb.Booster(model_file=str(BUILD / f"{args.model}.txt"))
feature_names = model.feature_name()
missing = [f for f in feature_names if f not in FOLDABLE]
assert not missing, f"model uses non-foldable features: {missing}"

family_code = code_map("source_family")
tfamily_code = code_map("target_family")
src_port_code = code_map("source_port")
tgt_port_code = code_map("target_port")

tgt_ids = np.array(targets, dtype=np.int64)
tgt_codes = np.array([tgt_port_code.get(int(t), -1) for t in tgt_ids], dtype=np.float32)
tgt_family_cat = np.array([tfamily_code.get(family_of[names[int(t)]], -1) for t in tgt_ids], dtype=np.float32)
tgt_bit = np.array([bit_of[names[int(t)]] for t in tgt_ids], dtype=np.float32)

rr = np.repeat(np.arange(120), n_tgt)
ti = np.tile(np.arange(n_tgt), 120)
dm = rr // 12
dym = rr % 12
blocks_total = 120 * n_tgt

folded = table.copy()
predicted_rows = 0
for direction in range(4):
    dx_mod = dm.astype(np.float32)
    dy_mod = dym.astype(np.float32)
    ax_mod = ((10 - dm) % 10 if direction in (2, 3) else dm).astype(np.float32)
    ay_mod = ((12 - dym) % 12 if direction in (1, 3) else dym).astype(np.float32)
    for si in range(n_src):
        name = names[sources[si]]
        fdx, fdy = net_dx.get(name, 0), net_dy.get(name, 0)
        block = table[direction, si].reshape(blocks_total)
        stack = np.stack([table[o, si].reshape(blocks_total) for o in range(4)])
        stack = np.where(stack == SENTINEL, -32768, stack)
        source_cat = src_port_code.get(sources[si], -1)
        source_family_cat = family_code.get(family_of[name], -1)
        source_bit = bit_of[name]
        columns = {
            "source_port": np.float32(source_cat),
            "source_family": np.float32(source_family_cat),
            "source_bit": np.float32(source_bit),
            "target_port": tgt_codes[ti],
            "target_family": tgt_family_cat[ti],
            "target_bit": tgt_bit[ti],
            "first_dx": np.float32(fdx),
            "first_dy": np.float32(fdy),
            "dx_mod": dx_mod,
            "dy_mod": dy_mod,
            "ax_mod": ax_mod,
            "ay_mod": ay_mod,
            "remaining_x_mod": ((dm - fdx) % 10).astype(np.float32),
            "remaining_y_mod": ((dym - fdy) % 12).astype(np.float32),
            "direction": np.float32(direction),
            "table_scaled": block.astype(np.float32),
            "envelope_proxy": stack.max(axis=0).astype(np.float32),
        }
        columns["port_pair"] = encode(
            "port_pair", np.full(blocks_total, source_cat, dtype=np.int64) * 512 + tgt_codes[ti].astype(np.int64))
        columns["family_pair"] = encode(
            "family_pair", np.full(blocks_total, source_family_cat, dtype=np.int64) * 64 +
            tgt_family_cat[ti].astype(np.int64))
        columns["bit_pair"] = encode(
            "bit_pair", np.full(blocks_total, source_bit, dtype=np.int64) * 16 + tgt_bit[ti].astype(np.int64))
        matrix = np.empty((blocks_total, len(feature_names)), dtype=np.float32)
        for index, feature in enumerate(feature_names):
            matrix[:, index] = columns[feature]
        correction = model.predict(matrix, num_iteration=args.trees, num_threads=8)
        delta = np.rint(correction * 12).astype(np.int32).reshape(120, n_tgt)
        original = block.reshape(120, n_tgt)
        merged = np.where(original == SENTINEL, SENTINEL,
                          np.clip(original.astype(np.int32) + delta, -32000, 32000))
        folded[direction, si] = merged.astype(np.int16)
        predicted_rows += blocks_total
        if si % 64 == 0 or si == n_src - 1:
            print(f"direction={direction} source={si + 1}/{n_src} rows={predicted_rows}", flush=True)

out = BUILD / args.out
out.mkdir(parents=True, exist_ok=True)
folded.tofile(out / "architecture.bin")
(out / "architecture.S").write_text('''// Link the folded table into the executable; no runtime file dependency.
.section .rodata
.balign 64
.global architecture_table
architecture_table:
.incbin "architecture.bin"
.section .note.GNU-stack,"",@progbits
''', encoding="utf-8", newline="\n")

coefficients = [0.9405376344086025, 1.6152054794520552, 1.9533377522462807,
                0.3491780821917809, 1.3123287671232877]
if args.net_coefficients:
    coefficients = json.loads(Path(args.net_coefficients).read_text(encoding="utf-8"))["coefficients"]
assert len(coefficients) == 5

lines = ["// Generated by fold_model.py from foldable model; do not edit.", "#pragma once",
         "#define FOLDED_MODEL 1",
         "#define TOPOLOGY_FEATURES 0",
         "#define FAST_GAP_TABLES 1",
         "#define NET_CORRECTION 1",
         "using model_real = double;",
         "#include <cstdint>", "#include <cstring>", "#include <string_view>",
         "struct PortInfo { const char* name; int family, bit, dx, dy, source, target; };",
         "static const PortInfo ports[] = {"]
source_ids = {p: i for i, p in enumerate(sources)}
target_ids = {p: i for i, p in enumerate(targets)}
for i, name in enumerate(names):
    lines.append("{" + json.dumps(name) + f",{family_of[name]},{bit_of[name]},"
                 f"{net_dx.get(name, 0)},{net_dy.get(name, 0)},"
                 f"{source_ids.get(i, -1)},{target_ids.get(i, -1)}" + "},")
lines += ["};", f"constexpr int n_sources={n_src}, n_targets={n_tgt};"]

packed = [int.from_bytes(n.encode("ascii"), "little") for n in names]
assert max(map(len, names)) <= 8
empty = (1 << 64) - 1
slot_bits = 11
slot_count = 1 << slot_bits
slot_mask = slot_count - 1


def build_slots(seed):
    occupied = {}
    for value in packed:
        slot = ((value * seed) & empty) >> (64 - slot_bits)
        probe = 0
        while slot in occupied:
            slot = (slot + 1) & slot_mask
            probe += 1
            if probe > 8:
                return None
        occupied[slot] = value
    keys = [empty] * slot_count
    ids = [-1] * slot_count
    for index, value in enumerate(packed):
        slot = ((value * seed) & empty) >> (64 - slot_bits)
        while keys[slot] != empty and keys[slot] != value:
            slot = (slot + 1) & slot_mask
        keys[slot] = value
        ids[slot] = index
    return keys, ids


seed = 0x9E3779B97F4A7C15
slot_keys = slot_ids = None
for _ in range(4096):
    built = build_slots(seed)
    if built is not None:
        slot_keys, slot_ids = built
        break
    seed = (seed * 0x2545F4914F6CDD1D + 0x9E3779B97F4A7C15) & empty
assert slot_keys is not None
lines.append("static const std::uint64_t port_slot_keys[]={" +
             ",".join(("~0ULL" if v == empty else f"{v}ULL") for v in slot_keys) + "};")
lines.append("static const short port_slot_ids[]={" + ",".join(map(str, slot_ids)) + "};")
lines.append("static const std::uint64_t port_packed[]={" + ",".join(f"{v}ULL" for v in packed) + "};")
lines.append("static const unsigned char port_lengths[]={" + ",".join(map(str, map(len, names))) + "};")
lines.append("static inline int lookup_port(std::string_view text) { std::uint64_t key=0; switch(text.size()) {")
for length in sorted(set(map(len, names))):
    lines.append(f"case {length}: std::memcpy(&key,text.data(),{length}); break;")
lines += ["default: return -1; }",
          f"unsigned slot=unsigned((key*{seed}ULL)>>{64 - slot_bits}u);",
          "for(;;) {",
          "    const std::uint64_t stored=port_slot_keys[slot];",
          "    if(stored==key) { const int id=port_slot_ids[slot];",
          "        return port_lengths[id]==text.size() && port_packed[id]==key ? id : -1; }",
          "    if(stored==~0ULL) return -1;",
          f"    slot=(slot+1u)&{slot_mask}u;",
          "} }"]
lines.append("static const int slopes[4][2] = " +
             json.dumps(slopes.tolist()).replace("[", "{").replace("]", "}") + ";")
for direction_name, axis in (("vertical", "x"), ("horizontal", "y")):
    axis_lines = [g for g in gaps["Line"] if g["direction"] == direction_name]
    limit = max(g["site"] for g in axis_lines) + 1
    prefix = [sum(g["delay"] for g in axis_lines if g["site"] < i) for i in range(limit + 1)]
    lines.append(f"constexpr int gap_{axis}_limit={limit};")
    lines.append(f"static const short gap_{axis}_prefix[]={{" + ",".join(map(str, prefix)) + "};")
wall_blocks = [b for b in gaps["Block"] if b["horizontal crossable"]]
corridors = {(b["left"], b["right"]) for b in wall_blocks}
assert len(corridors) == 1
left, right = next(iter(corridors))
height = max(b["upper"] for b in wall_blocks) + 1
delays = [0] * height
for b in wall_blocks:
    for y in range(b["lower"], b["upper"] + 1):
        delays[y] = b["horizontal cross delay"]
doors = np.r_[0, np.cumsum(np.array(delays) == 0)].tolist()
lines.append("static const double net_coefficients[]={" +
             ",".join(format(v, ".17g") for v in coefficients) + "};")
lines.append(f"constexpr int net_wall_left={left},net_wall_right={right},net_wall_height={height};")
lines.append("static const short net_wall_delays[]={" + ",".join(map(str, delays)) + "};")
lines.append("static const short net_door_prefix[]={" + ",".join(map(str, doors)) + "};")
(out / "model.hpp").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
(out / "export.json").write_text(json.dumps({
    "folded": True, "model": args.model, "trees": args.trees,
    "net_coefficients": coefficients, "features": feature_names,
    "entries": int(table.size), "predicted_rows": predicted_rows,
}, indent=2), encoding="utf-8", newline="\n")
print("folded table written to", out, flush=True)
