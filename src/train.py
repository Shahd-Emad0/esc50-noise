"""Train one config for one seed (test set never touched).  python -m src.train configs/cnn.yaml --seed 42"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from .datasets.esc50 import EXP, QUICK, ROOT, TRAIN_SNRS, VAL_CONDITIONS, condition_set, waveforms
from .datasets.noise import mix_noise
from .models import build_model
from .models.baseline_svm import make_svm, mfcc_features


def load_config(path):
    cfg = {**yaml.safe_load(open(ROOT / "configs" / "_defaults.yaml")), **yaml.safe_load(open(path))}
    cfg["name"] = Path(path).stem
    return cfg


def all_configs():
    return [load_config(p) for p in sorted((ROOT / "configs").glob("[!_]*.yaml"))]


def run_dir(name, seed):
    return EXP / "runs" / name / str(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True


def macro_f1(y, pred):
    return f1_score(y, pred, labels=list(range(50)), average="macro", zero_division=0)


@torch.no_grad()
def predict(model, X, bs=64):
    model.eval()
    dev = next(model.parameters()).device
    return torch.cat([model(X[i:i + bs].float().to(dev)).float().cpu() for i in range(0, len(X), bs)])


def validate(score_fn, df):
    """Model selection score = mean macro-F1 over clean + 4 white-noise SNRs on the VALIDATION split."""
    res = {}
    for c in VAL_CONDITIONS:
        x, idx = condition_set("val", c)
        res[c] = macro_f1(df.target.values[idx], np.asarray(score_fn(x)).argmax(1))
    res["selection"] = float(np.mean(list(res.values())))
    return res


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def train_svm(cfg, seed, out, X, df, tr):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    waves, ys = [X[i].float() for i in tr], df.target.values[tr]
    feats, labels = [mfcc_features([w.numpy() for w in waves])], [ys]
    if cfg["noise_aug"]:  # Member 1 strategy: 2 noisy copies per training clip
        for _ in range(2):
            noisy = [mix_noise(w, int(rng.choice(TRAIN_SNRS)), generator=torch.Generator().manual_seed(int(rng.integers(2**62))))
                     for w in waves]
            feats.append(mfcc_features([w.numpy() for w in noisy])); labels.append(ys)
    clf = make_svm().fit(np.concatenate(feats), np.concatenate(labels))
    train_s = time.perf_counter() - t0
    joblib.dump(clf, out / "model.joblib")
    val = validate(lambda x: clf.decision_function(mfcc_features(x.float().numpy())), df)
    save_json({**val, "epoch": None}, out / "val_metrics.json")
    save_json({"params": int(clf[-1].support_vectors_.size + clf[-1].dual_coef_.size), "trainable": "n/a (convex)",
               "size_mb": (out / "model.joblib").stat().st_size / 2**20, "train_seconds": train_s,
               "peak_mem_mb": None, "hardware": "CPU", "train_samples": int(sum(map(len, labels)))}, out / "cost.json")


def register_ast(seed, out):
    """AST heads were trained by Member 3 with the same split / seeds / selection rule; reuse them unchanged."""
    src = ROOT / "checkpoints" / "ast_probe" / str(seed)
    v = pd.read_csv(src / "val_metrics.csv").set_index("condition").macro_f1.to_dict()
    save_json({**v, "selection": float(np.mean(list(v.values()))), "epoch": None}, out / "val_metrics.json")
    c = json.loads((src / "cost.json").read_text())
    save_json({"params": c["total_parameters"], "trainable": c["trainable_parameters"],
               "size_mb": 344_909_000 / 2**20, "train_seconds": c["feature_stage_wall_seconds"] + c["head_training_and_validation_seconds"],
               "peak_mem_mb": None, "hardware": c["hardware"], "head_file": str(src / "best_head.pth")}, out / "cost.json")
    shutil.copy(ROOT / "legacy/member3_ast/experiments/ast_probe" / str(seed) / "history.csv", out / "history.csv")


def train(cfg, seed, epochs=None):
    out = run_dir(cfg["name"], seed)
    if (out / "cost.json").exists():
        return print(f"[skip] {cfg['name']} seed {seed} already trained")
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    X, df = waveforms()
    tr = np.where(df.split == "train")[0]
    if QUICK:
        tr = tr[::10]  # 128 clips spread over all classes
    print(f"\n=== {cfg['name']} | seed {seed} ===")
    if cfg["model"] == "ast":
        return register_ast(seed, out)
    if cfg["model"] == "svm":
        return train_svm(cfg, seed, out, X, df, tr)

    dev, bs = device(), cfg["batch_size"]
    epochs = epochs or cfg["epochs"]
    model = build_model(cfg).to(dev)
    y_all = torch.tensor(df.target.values)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=epochs * int(np.ceil(len(tr) / bs)), pct_start=0.1)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    snrs = torch.tensor(TRAIN_SNRS, dtype=torch.float32, device=dev)
    best, hist, t0 = -1.0, [], time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm, total = tr[torch.randperm(len(tr)).numpy()], 0.0
        for i in range(0, len(perm), bs):
            ids = perm[i:i + bs]
            x, y = X[ids].float().to(dev), y_all[ids].to(dev)
            if cfg["noise_aug"]:
                m = torch.rand(len(x), device=dev) < cfg["noise_p"]
                if m.any():
                    x[m] = mix_noise(x[m], snrs[torch.randint(len(snrs), (int(m.sum()),), device=dev)])
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(ids)
        row = {"epoch": ep, "loss": total / len(tr)}
        if ep > epochs // 4 or ep == epochs:  # checkpoint selection on validation only
            v = validate(lambda x: predict(model, x), df)
            row.update(v)
            if v["selection"] > best:
                best = v["selection"]
                torch.save(model.state_dict(), out / "model.pt")
                save_json({**v, "epoch": ep}, out / "val_metrics.json")
        hist.append(row)
        print(f"ep {ep:3d} loss {row['loss']:.3f}" + (f" val {row['selection']:.4f} (best {best:.4f})" if "selection" in row else ""), flush=True)
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    save_json({"params": sum(p.numel() for p in model.parameters()),
               "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "size_mb": (out / "model.pt").stat().st_size / 2**20, "train_seconds": time.perf_counter() - t0,
               "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
               "hardware": hardware(), "train_samples": len(tr), "epochs": epochs}, out / "cost.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    train(load_config(a.config), a.seed, a.epochs)
