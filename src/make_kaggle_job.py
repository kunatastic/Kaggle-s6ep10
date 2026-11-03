import json
import os
import sys
from pathlib import Path

folds = sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4"
extra = " ".join(sys.argv[2:]).split()
slug = os.environ.get("KAGGLE_SLUG", "s6e10-generative-transformer")
src = Path("src/gen.py").read_text()
job = f'''import subprocess, sys
open("/kaggle/working/gen.py", "w").write({src!r})
cmd = [sys.executable, "-u", "/kaggle/working/gen.py", "--folds", "{folds}", "--out", "/kaggle/working/preds/gen", *{extra!r}]
print("running:", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, cwd="/kaggle/working").returncode)
'''
out = Path("kaggle_jobs") / slug
out.mkdir(parents=True, exist_ok=True)
(out / "run.py").write_text(job)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"kunal20/{slug}",
    "title": slug.replace("-", " "),
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "false",
    "competition_sources": ["playground-series-s6e10"],
    "dataset_sources": [], "kernel_sources": [], "model_sources": [],
}, indent=2))
print(f"wrote {out}/ for folds {folds} {' '.join(extra)}")
