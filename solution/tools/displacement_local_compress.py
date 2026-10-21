"""Losslessly compress a frozen local asset; verify an exact byte roundtrip."""
import argparse
import json
import lzma
import time
from displacement_analyze import OUT
from displacement_assets import sha
from project_paths import REPO_ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('name')
    args=ap.parse_args();started=time.perf_counter();source=OUT/(args.name+'.local');target=OUT/(args.name+'.local.xz')
    raw=source.read_bytes();encoded=lzma.compress(raw,preset=6)
    assert lzma.decompress(encoded)==raw;target.write_bytes(encoded)
    report=dict(candidate=args.name,raw_sha256=sha(source),compressed_sha256=sha(target),raw_bytes=len(raw),
                compressed_bytes=len(encoded),exact_byte_roundtrip=True,codec='XZ/LZMA2 preset 6',
                runtime_library='liblzma.so.5 from the official fpga-ziguang image',elapsed_seconds=time.perf_counter()-started)
    (REPO_ROOT/'docs'/(args.name+'_local_compression.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
