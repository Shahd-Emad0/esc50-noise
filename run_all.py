"""ONE command for the whole project (resumable: finished runs are skipped).

  python run_all.py              full pipeline: data -> checks -> train all -> test once -> tables/figures -> report + slides
  python run_all.py --quick      ~10 min smoke test (1 seed, 1 epoch, 128 train clips) -> experiments/_quick
  python run_all.py --report     rebuild tables, report and slides only (e.g. after editing team.yaml)
  python run_all.py --epochs 30  shorter budget (applied to every model, so the comparison stays fair)
"""
import argparse
import os
import runpy
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--report", action="store_true")
ap.add_argument("--epochs", type=int)
ap.add_argument("--only", nargs="*", help="config names to train, e.g. --only cnn crnn_attention")
a = ap.parse_args()
if a.quick:
    os.environ["ESC50_QUICK"] = "1"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import analysis, evaluate                                        # noqa: E402
from src.datasets.esc50 import EXP, download, load_manifest, waveforms   # noqa: E402
from src.datasets.noise import self_test                                  # noqa: E402
from src.train import all_configs, hardware, train                        # noqa: E402

ORDER = ["svm_clean", "svm_noise", "ast_probe", "cnn", "crnn_attention", "crnn_no_attention", "crnn_no_noise",
         "crnn_no_specaug", "crnn_lr3e-4", "crnn_lr3e-3", "efficientnet_scratch", "efficientnet_pretrained"]


def step(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


t0 = time.time()
if not a.report:
    step("1/6  Dataset: download ESC-50 + leakage audit of the team split")
    download()
    df = load_manifest()
    print(df.split.value_counts().to_string(), "\nNo source file is shared between splits; all 50 classes in every split.")
    step("2/6  Waveform cache + custom noise-mixer self-test")
    waveforms(); self_test()
    print("Hardware:", hardware())
    step("3/6  Training (validation-only model selection; test set untouched)")
    cfgs = {c["name"]: c for c in all_configs()}
    for name in [n for n in ORDER if not a.only or n in a.only]:
        for seed in ([42] if a.quick else cfgs[name]["seeds"]):
            train(cfgs[name], seed, 1 if a.quick else a.epochs)
    step("4/6  Final test evaluation (same clips, same noise, same metrics for every model)")
    evaluate.main()
step("5/6  Tables, hypothesis verdicts, error analysis, figures")
analysis.main()
import json, shutil  # noqa: E401,E402  export the validation-chosen checkpoint of each main model
dep = json.loads((EXP / "results" / "deploy.json").read_text())["seeds"]
for name, seed in ({} if a.quick else dep).items():
    for f in (EXP / "runs" / name / str(seed)).glob("model.*"):
        (ROOT / "checkpoints" / "final").mkdir(parents=True, exist_ok=True)
        shutil.copy(f, ROOT / "checkpoints" / "final" / f"{name}_seed{seed}{f.suffix}")
step("6/6  Technical report (.md + .docx) and slides (.pptx)")
runpy.run_path(str(ROOT / "report" / "build_report.py"), run_name="__main__")
print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results: {EXP}\nLive demo:  python demo/app.py")
