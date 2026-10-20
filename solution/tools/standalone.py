"""Extract and run the ZIP in a directory containing no model or architecture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

base = Path("/workspace")
parser = argparse.ArgumentParser()
parser.add_argument("--package", default="submission.zip")
parser.add_argument("--reference-executable", default="bin/estimate")
parser.add_argument("--report", default="standalone.json")
args = parser.parse_args()
with tempfile.TemporaryDirectory() as folder:
    work = Path(folder)
    with zipfile.ZipFile(base / args.package) as archive:
        assert archive.namelist()==["bin/estimate"]
        archive.extractall(work)
    executable = work / "bin/estimate"
    # Python zip extraction does not restore POSIX permissions by itself.
    executable.chmod(0o755)
    request=work / "request.csv"
    request.write_text("From,To\nSRB_18_95/A_FQ0,SRB_38_57/ZDN[7]\nSRB_67_99/ZLE[4],SRB_67_99/ZLE[4]\n")
    result=work / "result.csv"
    subprocess.run([str(executable),"-in",str(request),"-out",str(result)],cwd=work,check=True)
    reference=work / "reference.csv"
    subprocess.run([str(base / args.reference_executable),"-in",str(request),"-out",str(reference)],check=True)
    assert result.read_bytes()==reference.read_bytes()
    lines=result.read_text().splitlines()
    assert len(lines)==3 and lines[0]=="From,To,delay"
    assert lines[2]=="SRB_67_99/ZLE[4],SRB_67_99/ZLE[4],0"
    assert int(lines[1].rsplit(",",1)[1])>=0
    report={"standalone_zip_test":"passed","rows":2,"expected_output_verified":True,
            "binary_sha256":hashlib.sha256(executable.read_bytes()).hexdigest()}
    (base / args.report).write_text(json.dumps(report,indent=2))
    print(report)
