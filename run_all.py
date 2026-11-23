"""Reproduce every number, chart, web file and the paper:  python run_all.py
Everything is written inside this folder (results/, out/). If the vivekmitra.com site folder sits two levels
up, the results and the paper are also copied there.
Author: Vivek Mitra
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT.parent.parent / "site"
os.environ["MPLCONFIGDIR"] = str(ROOT / "out" / "matplotlib")      # keep matplotlib's cache inside the repo
sys.path.insert(0, str(ROOT / "src"))

import evaluate  # noqa: E402
import figures   # noqa: E402

print("1. fit and score the models")
evaluate.run(ROOT)
print("2. check the GLM against an independent statsmodels formula fit")
subprocess.run([sys.executable, str(ROOT / "tests" / "test_glm.py")], check=True)
print("3. check the browser prices match Python")
subprocess.run(["node", str(ROOT / "tests" / "parity.mjs")], check=True)
print("4. figures and paper")
figures.make(ROOT / "results" / "pricing.json", ROOT / "out" / "figures")
subprocess.run("npm install --no-audit --no-fund --cache ../out/npm-cache", cwd=ROOT / "paper", shell=True, check=True)
subprocess.run(["node", str(ROOT / "paper" / "build_paper.mjs")], check=True)
subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "paper" / "to_pdf.ps1"),
                str(ROOT / "out" / "pricing.docx"), str(ROOT / "results" / "pricing.pdf")], check=True)
print("5. check every number in the paper against the results file")
subprocess.run([sys.executable, str(ROOT / "tests" / "check_paper.py")], check=True)

if SITE.exists():
    for name in ("pricing.json", "pricing.model.json", "pricing.parity.json"):
        shutil.copyfile(ROOT / "results" / name, SITE / "data" / name)
    shutil.copyfile(ROOT / "results" / "pricing.pdf", SITE / "papers" / "pricing.pdf")
    print(f"6. copied results and paper to {SITE}")
print("done")
