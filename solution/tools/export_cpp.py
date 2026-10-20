"""Export trained trees and topology tables into a standalone C++ estimator."""
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from experiment import DATA, BUILD, features, stage3_features

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="architecture_v3")
parser.add_argument("--trees", type=int, default=75)
parser.add_argument("--adaptive", action="store_true",
                    help="Use 50 trees for ordinary queries, up to 75 for short/horizontal-long sources")
parser.add_argument("--float", action="store_true", help="Use float32 tree expressions")
parser.add_argument("--generic", action="store_true",
                    help="Evaluate trees through the compact node table instead of expanded expressions")
parser.add_argument("--fast-categories", action="store_true",
                    help="Use direct category maps and shared bitsets for expanded categorical splits")
parser.add_argument("--packed-ports", action="store_true",
                    help="Use a verified 64-bit perfect hash for names up to eight bytes")
parser.add_argument("--output-dir", default="generated",
                    help="Directory below build for isolated candidate artifacts")
parser.add_argument("--fast-trees", type=int, default=0,
                    help="Use this prefix outside the short-distance refinement gate")
parser.add_argument("--refine-distance", type=int, default=100)
parser.add_argument("--cascade-model", default="",
                    help="Optional second residual model appended after the primary prefix")
parser.add_argument("--cascade-trees", type=int, default=0)
parser.add_argument("--cascade-min-distance", type=int, default=100)
parser.add_argument("--cascade-max-distance", type=int, default=300)
parser.add_argument("--net-correction", default="",
                    help="Training-only fitted net-correction metrics JSON below build")
args = parser.parse_args()
if args.trees < 0:
    parser.error("--trees must be nonnegative; zero exports the architecture baseline only")
if args.adaptive and args.trees != 75:
    parser.error("Adaptive mode requires --trees 75")
if args.fast_trees and (args.fast_trees < 1 or args.fast_trees >= args.trees or args.adaptive):
    parser.error("--fast-trees must be a smaller positive prefix and cannot be combined with --adaptive")
if args.cascade_model and args.adaptive:
    parser.error("--cascade-model cannot be combined with --adaptive")
if args.cascade_trees and not args.cascade_model:
    parser.error("--cascade-trees requires --cascade-model")
if args.net_correction and (args.adaptive or args.float):
    parser.error("net-correction coefficients are fitted to the double, non-adaptive predictor")
out = BUILD / args.output_dir
out.mkdir(parents=True, exist_ok=True)
model = lgb.Booster(model_file=str(BUILD / f"{args.model}.txt"))
if args.trees > model.num_trees():
    parser.error("--trees exceeds the available primary trees")
dump = model.dump_model(num_iteration=args.trees)
if args.trees == 0:
    dump["tree_info"] = []
net_config = None
if args.net_correction:
    net_config = json.loads((BUILD / args.net_correction).read_text(encoding="utf-8"))
    base_config = net_config["base_model"]
    expected_fast = args.fast_trees or args.trees
    if (base_config["primary"] != args.model or base_config["short_trees"] != args.trees or
            base_config["fast_trees"] != expected_fast or
            base_config["cascade_trees"] != args.cascade_trees or
            base_config["short_distance"] != args.refine_distance):
        parser.error("net-correction fit does not match the exported predictor")
    if args.cascade_model and (base_config["cascade"] != args.cascade_model or
            base_config["cascade_band"] != [args.cascade_min_distance, args.cascade_max_distance]):
        parser.error("net-correction fit does not match the cascade model or distance band")
    if len(net_config["coefficients"]) != 5:
        parser.error("net-correction requires five coefficients")
cascade_model = None
cascade_dump = None
if args.cascade_model:
    if args.cascade_trees <= 0:
        parser.error("--cascade-trees must be positive with --cascade-model")
    cascade_model = lgb.Booster(model_file=str(BUILD / f"{args.cascade_model}.txt"))
    if args.cascade_trees > cascade_model.num_trees():
        parser.error("--cascade-trees exceeds the available cascade trees")
    if cascade_model.feature_name() != model.feature_name():
        parser.error("cascade model feature names do not match primary model")
    cascade_dump = cascade_model.dump_model(num_iteration=args.cascade_trees)
frame = pd.read_csv(DATA / "data/delay_estimate_ans.csv")
x, linear_baseline = features(frame)
if "architecture_delay" in model.feature_name():
    from architecture import predict
    baseline = predict(frame)
    x["architecture_delay"] = baseline
else:
    baseline = linear_baseline
if "port_pair" in model.feature_name():
    x = stage3_features(x, baseline)
if "architecture_envelope" in model.feature_name():
    from topology_experiment import topology_features
    x, _ = topology_features(x)
meta = json.loads((BUILD / "architecture_meta.json").read_text())
names = meta["ports"]
families = sorted({n.split("[")[0] for n in names})
nets = json.loads((DATA / "arch/SRB_Net.json").read_text())["Nets"]
net_dx = {n["from"]: n["delta x"] for n in nets}
net_dy = {n["from"]: n["delta y"] for n in nets}
source_ids = {p: i for i, p in enumerate(meta["sources"])}
target_ids = {p: i for i, p in enumerate(meta["targets"])}
categorical = [name for name in x.columns if isinstance(x[name].dtype, pd.CategoricalDtype)]
category_maps = {}
for feature, categories in zip(categorical, dump["pandas_categorical"]):
    category_maps[feature] = {c: i for i, c in enumerate(categories)}
if len(category_maps) != len(dump["pandas_categorical"]):
    raise ValueError("Model categorical metadata does not match exported features")

def port_key(value):
    key = len(value)
    for ch in value.encode("ascii"):
        key = (key * 257 + ch) & ((1 << 64) - 1)
    return key

port_keys = [port_key(name) for name in names]
if len(set(port_keys)) != len(port_keys):
    raise ValueError("port key collision")
real = "float" if args.float else "double"
lines = ["// Generated by tools/export_cpp.py; do not edit.", "#pragma once",
         f"#define TOPOLOGY_FEATURES {int('architecture_envelope' in model.feature_name())}",
         f"#define FAST_GAP_TABLES {int(args.fast_categories)}",
         f"#define NET_CORRECTION {int(net_config is not None)}",
         f"using model_real = {real};",
         "#include <cstdint>", "#include <string_view>",
         "struct PortInfo { const char* name; int family, bit, dx, dy, source, target; int sp, sf, tp, tf; };",
         "static const PortInfo ports[] = {"]
for i, name in enumerate(names):
    family = families.index(name.split("[")[0])
    bit = int(name.split("[")[1][:-1]) if "[" in name else -1
    cats = [category_maps[c].get(v, -1) for c, v in zip(categorical, (i, family, i, family))]
    vals = [family, bit, net_dx.get(name, 0), net_dy.get(name, 0),
            source_ids.get(i, -1), target_ids.get(i, -1), *cats]
    lines.append("{" + json.dumps(name) + "," + ",".join(map(str, vals)) + "},")
lines += ["};", f"constexpr int n_sources={len(source_ids)}, n_targets={len(target_ids)};",
          ]
if args.packed_ports:
    if max(map(len, names)) > 8:
        parser.error("--packed-ports requires names no longer than eight bytes")
    packed = [int.from_bytes(name.encode("ascii"), "little") for name in names]
    # Open addressing into a table that stays in L1.  The previous layout used a
    # 65536-slot displacement array (128 KB), which the 57 MB architecture table
    # evicts from cache, so every pin paid a cold miss on that array as well.
    # Exact 64-bit key comparison is kept, so behaviour is unchanged.
    slot_count = 1024
    while slot_count < 4 * len(names):
        slot_count *= 2
    slot_bits = slot_count.bit_length() - 1
    slot_mask = slot_count - 1
    empty = (1 << 64) - 1

    def build(seed):
        keys_map = {}
        total_probe = 0
        for value in packed:
            slot = ((value * seed) & empty) >> (64 - slot_bits)
            probe = 0
            while slot in keys_map:
                slot = (slot + 1) & slot_mask
                probe += 1
                if probe > 8:
                    return None, None
            keys_map[slot] = value
            total_probe += probe
        keys = [empty] * slot_count
        ids = [-1] * slot_count
        for index, value in enumerate(packed):
            slot = ((value * seed) & empty) >> (64 - slot_bits)
            while keys[slot] != empty and keys[slot] != value:
                slot = (slot + 1) & slot_mask
            keys[slot] = value
            ids[slot] = index
        return (keys, ids), total_probe

    seed = 0x9E3779B97F4A7C15
    best = None
    for attempt in range(4096):
        built, probe = build(seed)
        if built is not None and (best is None or probe < best[0]):
            best = (probe, seed, built)
            if probe == 0:
                break
        seed = (seed * 0x2545F4914F6CDD1D + 0x9E3779B97F4A7C15) & empty
    if best is None:
        raise ValueError("Cannot find a port hash with bounded probes")
    probe, seed, (slot_keys, slot_ids) = best
    print("port hash slots", slot_count, "seed", hex(seed), "extra probes", probe, flush=True)
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
else:
    lines += ["static inline std::uint64_t port_key(std::string_view text) { std::uint64_t k=text.size(); for(unsigned char c:text) k=k*257+c; return k; }",
              "static inline int lookup_port(std::string_view text) { switch(port_key(text)) {"]
    lines += [f"case {port_keys[i]}ULL: return {i};" for i in range(len(names))]
    lines += ["default: return -1; } }"]
lines += ["static const int slopes[4][2] = " + json.dumps(meta["slopes_scaled12"]).replace("[", "{").replace("]", "}") + ";",
          "struct TreeNode { int left,right,feature; double value; unsigned cb, ce; };",
          "static const TreeNode nodes[] = {"]
nodes = []
cat_values = []
roots = []
cascade_roots = []
node_categories = {}
def visit(node):
    i = len(nodes)
    nodes.append(None)
    if "leaf_value" in node:
        nodes[i] = (-1,-1,-1,node["leaf_value"],0,0)
    else:
        cb = ce = 0
        if node["decision_type"] == "==":
            cb = len(cat_values)
            members = [int(c) for c in node["threshold"].split("||")]
            words = [0] * (max(members)//64+1)
            for c in members:
                words[c//64] |= 1 << (c%64)
            cat_values.extend(words)
            ce = len(words)
            node_categories[id(node)] = (cb, ce)
            threshold = 0.
        elif node["decision_type"] == "<=":
            threshold = float(node["threshold"])
        else:
            raise ValueError(node["decision_type"])
        left = visit(node["left_child"])
        right = visit(node["right_child"])
        nodes[i] = (left,right,node["split_feature"],threshold,cb,ce)
    return i
for tree in dump["tree_info"]:
    roots.append(visit(tree["tree_structure"]))
if cascade_dump is not None:
    for tree in cascade_dump["tree_info"]:
        cascade_roots.append(visit(tree["tree_structure"]))
for left,right,feature,value,cb,ce in nodes:
    lines.append(f"{{{left},{right},{feature},{value:.17g},{cb},{ce}}},")
lines += ["};", "static const int roots[] = {"+",".join(map(str, roots))+"};",
          "static const int cascade_roots[] = {"+",".join(map(str, cascade_roots))+"};",
          "static const unsigned long long categories[] = {"+(",".join(str(v)+"ULL" for v in cat_values) or "0")+"};",
          f"constexpr int n_features={len(model.feature_name())};"]
lines += [f"constexpr int F_{name}={i};" for i, name in enumerate(model.feature_name())]
# LightGBM stores pandas categorical values as dense codes.  Keep the exact
# value-to-code maps in the generated header so C++ follows pandas semantics.
for index, feature in enumerate(categorical):
    values = list(category_maps[feature].keys())
    lines.append("static const int category_values_%d[] = {%s};" %
                 (index, ",".join(str(int(v)) for v in values)))
    lines.append("static constexpr int category_size_%d=%d;" % (index, len(values)))
lines.append("static inline int category_lookup(const int* values,int count,int value) { int lo=0,hi=count; while(lo<hi) { int mid=lo+(hi-lo)/2; if(values[mid]<value) lo=mid+1; else hi=mid; } return (lo<count && values[lo]==value) ? lo : -1; }")
lines.append("static inline int category_code(int feature,int value) {")
for index, feature in enumerate(categorical):
    if args.fast_categories:
        mapping = {int(k): v for k, v in category_maps[feature].items()}
        low, high = min(mapping), max(mapping)
        dense = [mapping.get(k, -1) for k in range(low, high + 1)]
        if dense == list(range(len(dense))):
            lines.append(f"if(feature==F_{feature}) return unsigned(value-({low}))<{len(dense)}u ? value-({low}) : -1;")
        else:
            lines.append(f"if(feature==F_{feature}) {{ static const short map[]={{" + ",".join(map(str, dense)) + f"}}; return unsigned(value-({low}))<{len(dense)}u ? map[value-({low})] : -1; }}")
    else:
        lines.append(f"if(feature==F_{feature}) return category_lookup(category_values_{index},category_size_{index},value);")
lines.append("return value; }")
gaps = json.loads((DATA / "arch/SRB_Gap.json").read_text())["Gap"]
if net_config is not None:
    if not args.fast_categories:
        parser.error("--net-correction requires --fast-categories for prefix gap tables")
    wall_blocks = [b for b in gaps["Block"] if b["horizontal crossable"]]
    corridors = {(b["left"], b["right"]) for b in wall_blocks}
    if len(corridors) != 1:
        parser.error("net-correction C++ kernel requires one horizontal corridor")
    left, right = next(iter(corridors))
    height = max(b["upper"] for b in wall_blocks) + 1
    delays = [0] * height
    for b in wall_blocks:
        for y in range(b["lower"], b["upper"] + 1):
            delays[y] = b["horizontal cross delay"]
    doors = np.r_[0, np.cumsum(np.array(delays) == 0)].tolist()
    lines.append("static const double net_coefficients[]={" +
                 ",".join(format(v, ".17g") for v in net_config["coefficients"]) + "};")
    lines.append(f"constexpr int net_wall_left={left},net_wall_right={right},net_wall_height={height};")
    lines.append("static const short net_wall_delays[]={" + ",".join(map(str, delays)) + "};")
    lines.append("static const short net_door_prefix[]={" + ",".join(map(str, doors)) + "};")
if args.fast_categories:
    for direction, axis in (("vertical", "x"), ("horizontal", "y")):
        gap_lines_axis = [g for g in gaps["Line"] if g["direction"] == direction]
        limit = max(g["site"] for g in gap_lines_axis) + 1
        prefix = [sum(g["delay"] for g in gap_lines_axis if g["site"] < i)
                  for i in range(limit + 1)]
        lines.append(f"constexpr int gap_{axis}_limit={limit};")
        lines.append(f"static const short gap_{axis}_prefix[]={{" + ",".join(map(str, prefix)) + "};")
if "architecture_envelope" in model.feature_name():
    # An alternative potential cannot improve the selected quadrant once the
    # spatial penalty exceeds this exact, topology-only maximum table delta.
    table = np.load(BUILD / "architecture_table.npy", mmap_mode="r")
    bounds = []
    for direction in range(4):
        row = []
        for alternate in range(4):
            valid = (table[direction] != 32767) & (table[alternate] != 32767)
            delta = table[alternate].astype(np.int32) - table[direction]
            row.append(int(delta[valid].max()))
        bounds.append(row)
    lines.append("static const int envelope_bounds[4][4]=" +
                 json.dumps(bounds).replace("[", "{").replace("]", "}") + ";")
lines += ["struct GapLine {bool vertical; int site,delay;};", "static const GapLine gap_lines[] = {"]
lines += ["{" + ("true" if g["direction"] == "vertical" else "false") + f",{g['site']},{g['delay']}"+"}," for g in gaps["Line"]]
lines += ["};", "struct Block {int id,left,right,lower,upper;};", "static const Block blocks[] = {"]
lines += ["{"+",".join(str(b[k]) for k in ("id","left","right","lower","upper"))+"}," for b in gaps["Block"] if b["horizontal crossable"]]
lines += ["};"]
if args.fast_categories:
    lines.append("static inline bool category_match(model_real value,unsigned begin,unsigned count) { unsigned c=unsigned(int(value)); return c/64u<count && ((categories[begin+c/64u]>>(c&63u))&1ULL); }")
def expression(node):
    if "leaf_value" in node:
        return f"{node['leaf_value']:.17g}"
    feature = node["split_feature"]
    if node["decision_type"] == "==":
        if args.fast_categories:
            cb, count = node_categories[id(node)]
            condition = f"category_match(f[{feature}],{cb},{count})"
            return "("+condition+"?"+expression(node["left_child"])+":"+expression(node["right_child"])+")"
        members = [int(c) for c in node["threshold"].split("||")]
        words = [0] * (max(members)//64+1)
        for c in members:
            words[c//64] |= 1 << (c%64)
        conditions = []
        for i, bits in enumerate(words):
            if bits:
                conditions.append(f"(unsigned(int(f[{feature}]))/64=={i} && (({bits}ULL>>(unsigned(int(f[{feature}]))&63))&1))")
        condition = "("+"||".join(conditions)+")"
    else:
        condition = f"f[{feature}]<={float(node['threshold']):.17g}"
    return "("+condition+"?"+expression(node["left_child"])+":"+expression(node["right_child"])+")"
if not args.generic:
    for i, tree in enumerate(dump["tree_info"]):
        lines.append(f"static {real} tree_{i}(const {real}* f) {{ return {expression(tree['tree_structure'])}; }}")
    if cascade_dump is not None:
        for i, tree in enumerate(cascade_dump["tree_info"]):
            lines.append(f"static {real} cascade_tree_{i}(const {real}* f) {{ return {expression(tree['tree_structure'])}; }}")
    lines.append(f"static {real} correction(const {real}* f) {{ {real} s=0;")
    if cascade_dump is not None:
        for i in range(min(args.fast_trees, len(roots))):
            lines.append(f"s+=tree_{i}(f);")
        if args.fast_trees:
            lines.append(f"if(f[F_manhattan] < {args.refine_distance}) {{")
            for i in range(args.fast_trees, len(roots)):
                lines.append(f"s+=tree_{i}(f);")
            lines.append("}")
        else:
            for i in range(len(roots)):
                lines.append(f"s+=tree_{i}(f);")
        lines.append(f"if(f[F_manhattan]>={args.cascade_min_distance} && f[F_manhattan] < {args.cascade_max_distance}) {{")
        for i in range(len(cascade_roots)):
            lines.append(f"s+=cascade_tree_{i}(f);")
        lines.append("}")
    else:
        for i in range(len(roots)):
            if args.fast_trees and i == args.fast_trees:
                lines.append(f"if(f[F_manhattan]>={args.refine_distance}) return s;")
            if args.adaptive and i == 50:
                lines.append("if(f[F_manhattan]>=100 && std::abs(f[F_first_dx])!=10) return s;")
            lines.append(f"s+=tree_{i}(f);")
    lines += ["return s;}"]
else:
    lines += [
        "static inline bool node_category_match(const TreeNode& n, const model_real* f) {",
        "  int c=int(f[n.feature]);",
        "  return c>=0 && unsigned(c)/64u < n.ce-n.cb && ((categories[n.cb+unsigned(c)/64u]>>(unsigned(c)&63u))&1ULL);",
        "}",
        "static inline model_real eval_tree(int root, const model_real* f) {",
        "  int node=root;",
        "  while (nodes[node].left>=0) {",
        "    const TreeNode& n=nodes[node];",
        "    bool go_left = n.cb!=n.ce ? node_category_match(n,f) : f[n.feature]<=n.value;",
        "    node=go_left ? n.left : n.right;",
        "  }",
        "  return model_real(nodes[node].value);",
        "}",
        "static model_real correction(const model_real* f) { model_real s=0;",
    ]
    if cascade_dump is not None:
        for i in range(min(args.fast_trees, len(roots))):
            lines.append(f"s+=eval_tree(roots[{i}],f);")
        if args.fast_trees:
            lines.append(f"if(f[F_manhattan] < {args.refine_distance}) {{")
            for i in range(args.fast_trees, len(roots)):
                lines.append(f"s+=eval_tree(roots[{i}],f);")
            lines.append("}")
        else:
            for i in range(len(roots)):
                lines.append(f"s+=eval_tree(roots[{i}],f);")
        lines.append(f"if(f[F_manhattan]>={args.cascade_min_distance} && f[F_manhattan] < {args.cascade_max_distance}) {{")
        for i in range(len(cascade_roots)):
            lines.append(f"s+=eval_tree(cascade_roots[{i}],f);")
        lines.append("}")
    else:
        for i in range(len(roots)):
            if args.fast_trees and i == args.fast_trees:
                lines.append(f"if(f[F_manhattan]>={args.refine_distance}) return s;")
            if args.adaptive and i == 50:
                lines.append("if(f[F_manhattan]>=100 && std::abs(f[F_first_dx])!=10) return s;")
            lines.append(f"s+=eval_tree(roots[{i}],f);")
    lines += ["return s;}"]
(out / "model.hpp").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
np.load(BUILD / "architecture_table.npy").astype("<i2").tofile(out / "architecture.bin")
(out / "architecture.S").write_text('''// Link the table into the executable; no runtime file dependency.
.section .rodata
.balign 64
.global architecture_table
architecture_table:
.incbin "architecture.bin"
.section .note.GNU-stack,"",@progbits
''', encoding="utf-8", newline="\n")
(out / "export.json").write_text(json.dumps({"model":args.model, "trees":len(roots),
    "cascade_model":args.cascade_model, "cascade_trees":len(cascade_roots),
    "cascade_min_distance":args.cascade_min_distance, "cascade_max_distance":args.cascade_max_distance,
    "adaptive":args.adaptive,
    "float":args.float,
    "generic":args.generic,
    "fast_categories":args.fast_categories,
    "packed_ports":args.packed_ports,
    "topology_features":"architecture_envelope" in model.feature_name(),
    "fast_trees":args.fast_trees,
    "refine_distance":args.refine_distance,
    "net_correction":args.net_correction,
    "net_coefficients":net_config["coefficients"] if net_config is not None else [],
    "nodes":len(nodes), "feature_names":model.feature_name(),
    "categorical_maps":category_maps},indent=2), encoding="utf-8", newline="\n")
print("exported", args.model, len(roots), "trees", "cascade", len(cascade_roots), "trees", len(nodes), "nodes", flush=True)
