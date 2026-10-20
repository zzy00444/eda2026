"""Run observable CSV/interface checks using only the standalone executable."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import hashlib
import math

root = Path("/workspace")
parser = argparse.ArgumentParser()
parser.add_argument("--executable", default="bin/estimate")
parser.add_argument("--report", default="integration.json")
args = parser.parse_args()
executable = root / args.executable
reports = {}
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    request, result = work / "request.csv", work / "result.csv"
    def run(payload, ok=True):
        request.write_bytes(payload)
        child = subprocess.run([str(executable),"-in",str(request),"-out",str(result)],capture_output=True)
        assert (child.returncode==0)==ok,child.stderr.decode()
        return result.read_bytes()
    rows = [
        ("SRB_67_99/ZLE[4]","SRB_67_99/ZLE[4]"),
        ("SRB_18_95/A_FQ0","SRB_38_57/ZDN[7]"),
        ("SRB_89_372/ZLE[0]","SRB_114_366/ZSSB[1]"),
        ("SRB_67_99/S_DQ0","SRB_67_99/S_DQ0"),
    ]
    text = "From,To\n"+"\n".join(",".join(row) for row in rows)
    lf=run(text.encode())
    assert run(text.replace("\n","\r\n").encode())==lf
    assert run(b"\xef\xbb\xbf"+text.encode())==lf
    assert run((text+"\n\n").encode())==lf
    parsed=list(csv.DictReader(io.StringIO(lf.decode())))
    assert [(r["From"],r["To"]) for r in parsed]==rows
    assert parsed[0]["delay"]==parsed[3]["delay"]=="0"
    assert run(b"From,To\n")==b"From,To,delay\n"
    run(b"From,To\nSRB_1_1/UNKNOWN,SRB_1_2/ZLE[0]\n",ok=False)
    run(b"From,To\nSRB_1_1/ZLE[0]\n",ok=False)
    run(b"wrong,header\n",ok=False)
    reports["interface_checks"] = ["LF", "CRLF", "UTF8_BOM", "no_trailing_newline",
        "blank_lines", "order_and_names", "identical_pin_zero", "header_only", "invalid_port_rejected",
        "invalid_fields_rejected", "invalid_header_rejected"]
    check_file = root / "check.csv"
    if check_file.exists():
        with check_file.open(newline="") as stream:
            golden=list(csv.DictReader(stream))
        request_text="From,To\n"+"".join(r["From"]+","+r["To"]+"\n" for r in golden)
        result_bytes=run(request_text.encode())
        predictions=list(csv.DictReader(io.StringIO(result_bytes.decode())))
        points=[]
        details=[]
        for expected,predicted in zip(golden,predictions):
            y=int(expected["Min Delay"]);p=int(predicted["delay"])
            score=100*(1-math.tanh(4*abs(p-y)/y)) if y else 100*int(p==0)
            points.append(score)
            details.append({"actual":y,"prediction":p,"score":score})
        reports["eight_path_examples"]={"score":sum(points)/len(points),"details":details}
(root / args.report).write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
