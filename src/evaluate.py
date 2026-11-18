"""Shared final evaluation for ALL models: same 400 test clips, same 6 fixed corruptions, same metrics.
Runs once after all training is finished; nothing here is used for model or hyper-parameter selection."""
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .datasets.esc50 import CONDITIONS, EXP, ROOT, class_names, condition_set, waveforms
from .models import load_trained
from .models.baseline_svm import mfcc_features
from .train import all_configs, device, hardware, run_dir

_AST = {}


def ast_model(dev):
    if "m" not in _AST:
        from .models.ast_probe import ASTProbe
        _AST["m"] = ASTProbe(dev)
    return _AST["m"]


def ast_embeddings(dev):
    """Frozen backbone => embed each test condition once and reuse for all 3 heads."""
    path = EXP / "runs" / "ast_probe" / "test_embeddings.pt"
    if path.exists():
        return torch.load(path)
    m, emb = ast_model(dev), {}
    for c in CONDITIONS:
        x, _ = condition_set("test", c)
        emb[c] = torch.cat([m.embed(x[i:i + 8]) for i in range(0, len(x), 8)])
        print(f"AST test embeddings {c}: done", flush=True)
    torch.save(emb, path)
    return emb


def make_scorer(cfg, seed, dev, labels):
    out = run_dir(cfg["name"], seed)
    if cfg["model"] == "svm":
        clf = joblib.load(out / "model.joblib")
        return lambda x: clf.decision_function(mfcc_features(x.float().numpy()))
    if cfg["model"] == "ast":
        m = ast_model(dev).load_head(ROOT / "checkpoints" / "ast_probe" / str(seed) / "best_head.pth", labels)
        return lambda x: m(x).numpy()
    model = load_trained(cfg, out / "model.pt", dev)

    @torch.no_grad()
    def score(x, bs=64):
        return torch.cat([model(x[i:i + bs].float().to(dev)).cpu() for i in range(0, len(x), bs)]).numpy()
    return score


def time_inference(fn, X, n=20):
    """End-to-end latency from raw 5 s waveform, batch size 1, warmed up."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for i in range(3):
        fn(X[i:i + 1])
    times = []
    for i in range(n):
        sync(); t = time.perf_counter(); fn(X[i:i + 1]); sync()
        times.append(time.perf_counter() - t)
    return 1000 * float(np.mean(times))


def metrics(y, scores):
    pred = scores.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(50)), average="macro", zero_division=0)
    top5 = float(np.mean([t in s for t, s in zip(y, np.argsort(-scores, 1)[:, :5])]))
    return pred, {"accuracy": accuracy_score(y, pred), "macro_f1": f, "macro_precision": p, "macro_recall": r, "top5_accuracy": top5}


def evaluate_run(cfg, seed, dev, labels, df):
    out = run_dir(cfg["name"], seed)
    if (out / "test_metrics.csv").exists():
        return
    score = make_scorer(cfg, seed, dev, labels)
    emb = ast_embeddings(dev) if cfg["model"] == "ast" else None
    preds, rows, per_class = {}, [], []
    for c in CONDITIONS:
        x, idx = condition_set("test", c)
        y = df.target.values[idx]
        s = _AST["m"].head(emb[c]).numpy() if emb else score(x)
        preds[c], m = metrics(y, s)
        rows.append({"model": cfg["name"], "seed": seed, "condition": c, **m})
        f1 = precision_recall_fscore_support(y, preds[c], labels=list(range(50)), zero_division=0)[2]
        per_class += [{"model": cfg["name"], "seed": seed, "condition": c, "class": labels[k], "f1": f1[k]} for k in range(50)]
    pd.DataFrame({"filename": df.filename.values[idx], "true": y, **{f"pred_{c}": p for c, p in preds.items()}}).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "test_per_class.csv", index=False)
    cost = json.loads((out / "cost.json").read_text())
    cost["infer_ms_per_clip"] = time_inference(score, condition_set("test", "clean")[0])
    cost["infer_hardware"] = hardware()
    (out / "cost.json").write_text(json.dumps(cost, indent=2))
    print(f"[test] {cfg['name']} seed {seed}: " + ", ".join(f"{r['condition']} F1={r['macro_f1']:.3f}" for r in rows), flush=True)


def main():
    X, df = waveforms()
    labels, dev = class_names(df), device()
    for cfg in all_configs():
        for seed in cfg["seeds"]:
            if (run_dir(cfg["name"], seed) / "cost.json").exists():
                try:
                    evaluate_run(cfg, seed, dev, labels, df)
                except Exception as e:  # e.g. no internet for AST download: report it, keep going
                    print(f"[warn] evaluation failed for {cfg['name']} seed {seed}: {e}")
    res = EXP / "results"
    res.mkdir(parents=True, exist_ok=True)
    for name in ["test_metrics", "test_per_class"]:
        files = sorted((EXP / "runs").glob(f"*/*/{name}.csv"))
        if files:
            pd.concat(map(pd.read_csv, files)).to_csv(res / f"{name}.csv", index=False)
    print("Test results ->", res)


if __name__ == "__main__":
    main()
