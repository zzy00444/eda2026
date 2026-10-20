"""Package only the Linux executable and preserve POSIX execute permissions."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--binary", default="solution/bin/estimate")
parser.add_argument("--output", default="solution/submission.zip")
parser.add_argument("--report", default="solution/build/package.json")
args = parser.parse_args()
binary = root / args.binary
payload = binary.read_bytes()
assert payload[:4]==b"\x7fELF", "Expected a Linux ELF executable"
package = root / args.output
entry = zipfile.ZipInfo("bin/estimate",date_time=(2026,10,19,0,0,0))
entry.create_system = 3
entry.external_attr = 0o100755 << 16
entry.compress_type = zipfile.ZIP_DEFLATED
with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    archive.writestr(entry,payload,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
assert package.stat().st_size < 100_000_000
with zipfile.ZipFile(package) as archive:
    assert archive.namelist()==["bin/estimate"]
    assert (archive.getinfo("bin/estimate").external_attr>>16)&0o111==0o111
report={"binary_bytes":len(payload),"zip_bytes":package.stat().st_size,
        "binary_sha256":hashlib.sha256(payload).hexdigest(),
        "zip_sha256":hashlib.sha256(package.read_bytes()).hexdigest(),
        "contents":["bin/estimate"],"unix_mode":"100755"}
(root / args.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
