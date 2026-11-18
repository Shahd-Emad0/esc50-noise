"""Builds every report table (1-7 + studies), hypothesis verdicts, error analysis and figures from saved runs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from .datasets.esc50 import CONDITIONS, EXP, ROOT, LogMel, class_names, condition_set, waveforms
from .train import all_configs, run_dir

TAB, FIG, RES = EXP / "tables", EXP / "figures", EXP / "results"
WHITE = CONDITIONS[:5]
BROADBAND = {"wind", "rain", "sea_waves", "crackling_fire", "vacuum_cleaner", "pouring_water", "airplane",
             "helicopter", "engine", "train", "washing_machine", "thunderstorm", "water_drops", "toilet_flush"}


def fmt(v, sd=None, pct=True):
    k = 100 if pct else 1
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{k * v:.1f}" + ("" if sd is None or np.isnan(sd) else f" ± {k * sd:.1f}")


def md(df, name, title):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    text = f"**{title}**\n\n| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    text += "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))
    (TAB / f"{name}.md").write_text(text, encoding="utf-8")
    return text


def ci(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return d.mean(), np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.std(ddof=1), d.mean() - h, d.mean() + h


def verdict(d):
    m, sd, lo, hi = ci(d)
    if np.isnan(lo):
        return "Inconclusive (one seed)"
    return "Supported" if lo > 0 else "Rejected (opposite effect)" if hi < 0 else "Not supported (95% CI includes 0)"


class Results:
    def __init__(self):
        self.cfgs = {c["name"]: c for c in all_configs()}
        self.test = pd.read_csv(RES / "test_metrics.csv")
        self.pc = pd.read_csv(RES / "test_per_class.csv")
        self.val, self.cost = {}, {}
        for n, c in self.cfgs.items():
            for s in c["seeds"]:
                d = run_dir(n, s)
                if (d / "val_metrics.json").exists():
                    self.val[n, s] = json.loads((d / "val_metrics.json").read_text())
                    self.cost[n, s] = json.loads((d / "cost.json").read_text())
        self.have = [n for n in self.cfgs if n in set(self.test.model)]
        self.main = [n for n in self.have if self.cfgs[n]["role"] == "main"]

    def per_seed(self, model, cond):
        """Macro-F1 per seed; cond='overall' = mean over clean + 4 white SNRs."""
        t = self.test[self.test.model == model]
        conds = WHITE if cond == "overall" else [cond]
        return t[t.condition.isin(conds)].groupby("seed").macro_f1.mean()

    def stat(self, model, cond):
        s = self.per_seed(model, cond)
        return s.mean(), (s.std(ddof=1) if len(s) > 1 else np.nan)

    def paired(self, a, b, cond):
        sa, sb = self.per_seed(a, cond), self.per_seed(b, cond)
        common = sa.index.intersection(sb.index)
        return (sa[common] - sb[common]).values

    def val_score(self, model):
        return np.mean([v["selection"] for (n, _), v in self.val.items() if n == model])

    def disp(self, n):
        return self.cfgs[n]["display"]


def table1():
    rows = [["ESC-50 (Piczak 2015)", "github.com/karolpiczak/ESC-50", "2,000 clips (1,280 / 320 / 400)", "5 s WAV, 44.1 kHz mono",
             "50 (40 clips each)", "CC BY-NC (ESC-10: CC BY)", "train / val / test (group split on src_file)"],
            ["Synthetic white noise (ours)", "src/datasets/noise.py", "generated on the fly", "Gaussian, SNR 10/5/0/-5 dB",
             "–", "n/a", "training augmentation + test corruption"],
            ["Synthetic pink noise (ours)", "src/datasets/noise.py", "400 test clips", "1/f noise, SNR 0 dB", "–", "n/a",
             "test only (unseen noise type)"],
            ["AudioSet (Gemmeke et al. 2017)", "research.google.com/audioset", "~2M clips", "10 s YouTube audio", "527",
             "CC BY 4.0 (labels)", "external: AST pretraining only"],
            ["ImageNet-1k (Deng et al. 2009)", "image-net.org", "1.28M images", "RGB images", "1,000",
             "research licence", "external: EfficientNet pretraining only"]]
    return md(pd.DataFrame(rows, columns=["Dataset", "Source", "Size", "Format", "Classes", "Licence", "Role"]),
              "table1_datasets", "Table 1 — Dataset comparison")


def table2(R):
    inp = {"svm": "MFCC 80-d (22.05 kHz)", "ast": "log-mel 128×1024 frames (16 kHz)"}
    pre = {"ast": "Yes (AudioSet)"}
    rows = []
    for n in R.main:
        c = R.cfgs[n]
        cost = next(v for (m, _), v in R.cost.items() if m == n)
        rows.append([c["display"], c["category"], inp.get(c["model"], "log-mel 128×216 (22.05 kHz)"), c["layers"],
                     f"{cost['params']:,}", pre.get(c["model"], "Yes (ImageNet)" if c["pretrained"] else "No")])
    return md(pd.DataFrame(rows, columns=["Model", "Family", "Input", "Main layers", "Parameters", "Pretrained"]),
              "table2_architectures", "Table 2 — Model architecture comparison")


def table3(R):
    rows = []
    for n in R.have:
        c = R.cfgs[n]
        if c["role"] == "hp":
            continue
        if c["model"] == "svm":
            rows.append([c["display"], "libsvm (SMO)", "–", "–", "–", "C = 10, RBF γ = scale", "hinge (multi-class OvO)",
                         "2 noisy copies / clip" if c["noise_aug"] else "none"])
            continue
        aug = ", ".join([f"white noise p={c['noise_p']} SNR∈{{10,5,0,−5}}"] * c["noise_aug"] + ["SpecAugment"] * c["specaug"])
        if c["model"] == "ast":
            aug = "1 white-noise copy / clip (SNR∈{10,5,0,−5})"
        reg = f"wd {c['weight_decay']}" + (", dropout 0.3" if c["model"] != "ast" else ", frozen encoder")
        ep = next((v["epochs"] for (m, _), v in R.cost.items() if m == n and "epochs" in v), c["epochs"])
        rows.append([c["display"], c["optimizer"] + (" + OneCycle" if c["model"] != "ast" else ""), c["lr"], c["batch_size"],
                     ep, reg, c["loss"], aug or "none"])
    return md(pd.DataFrame(rows, columns=["Model", "Optimizer", "LR", "Batch", "Epochs", "Regularisation", "Loss", "Augmentation"]),
              "table3_hyperparameters", "Table 3 — Hyper-parameter configuration")


def table4(R):
    rows, raw = [], []
    for n in R.main:
        row = [R.disp(n)] + [fmt(*R.stat(n, c)) for c in CONDITIONS]
        m, sd, lo, hi = ci(R.per_seed(n, "overall"))
        acc = R.test[(R.test.model == n) & (R.test.condition == "clean")].accuracy
        row += [f"{fmt(m, sd)} [{fmt(lo)}, {fmt(hi)}]", fmt(acc.mean(), acc.std(ddof=1) if len(acc) > 1 else None)]
        rows.append(row)
        raw.append({"model": n, "overall_mean": m, "overall_sd": sd, "ci_low": lo, "ci_high": hi,
                    **{c: R.stat(n, c)[0] for c in CONDITIONS}})
    pd.DataFrame(raw).to_csv(TAB / "table4_raw.csv", index=False)
    cols = ["Model"] + [f"F1 {c}" for c in CONDITIONS] + ["Mean F1 clean+white [95% CI]", "Clean accuracy"]
    return md(pd.DataFrame(rows, columns=cols), "table4_main_results",
              "Table 4 — Test macro-F1 (%) per condition, mean ± SD over 3 seeds (primary metric: macro-F1; secondary: accuracy)")


def table5(R):
    pairs = [("crnn_no_attention", "crnn_attention", "Attention pooling (→ mean pooling)"),
             ("crnn_no_noise", "crnn_attention", "Noise augmentation"),
             ("crnn_no_specaug", "crnn_attention", "SpecAugment"),
             ("efficientnet_scratch", "efficientnet_pretrained", "ImageNet pretraining"),
             ("svm_clean", "svm_noise", "Noise augmentation (SVM baseline)")]
    rows = []
    for abl, full, comp in pairs:
        if abl not in R.have or full not in R.have:
            continue
        r = [comp, R.disp(full)]
        for c in ["clean", "-5dB", "pink_0dB", "overall"]:
            d = ci(R.paired(abl, full, c))
            r.append(f"{fmt(R.stat(abl, c)[0])} ({'+' if d[0] >= 0 else ''}{fmt(d[0], d[1])})")
        rows.append(r)
    leg = ROOT / "legacy/member3_ast/experiments/ast_probe/supplementary_studies/overall_macro_f1_summary.csv"
    if leg.exists():
        s = pd.read_csv(leg).set_index("study")["mean"]
        rows.append(["Noise augmentation (AST head, validation, Roqaya)", "AST + head", "–", "–", "–",
                     f"{fmt(s['no_noise'])} ({fmt(s['no_noise'] - s['reference'])})"])
    return md(pd.DataFrame(rows, columns=["Component removed", "Full model", "Clean F1 (Δ)", "−5 dB F1 (Δ)", "Pink 0 dB F1 (Δ)", "Mean F1 (Δ)"]),
              "table5_ablation", "Table 5 — Ablation study: test macro-F1 (%) without the component, (Δ vs full model, paired by seed, mean ± SD)")


def study_tables(R):
    out = []
    hp = [n for n in ["crnn_lr3e-4", "crnn_attention", "crnn_lr3e-3"] if any(k[0] == n for k in R.val)]
    rows = []
    for n in hp:
        v = pd.DataFrame([x for (m, _), x in R.val.items() if m == n])
        rows.append([R.cfgs[n]["lr"], fmt(v.selection.mean(), v.selection.std()), fmt(v.clean.mean()), fmt(v["-5dB"].mean()),
                     f"{v.epoch.mean():.0f}"])
    if rows:
        out.append(md(pd.DataFrame(rows, columns=["Learning rate", "Val selection F1", "Val clean F1", "Val −5 dB F1", "Best epoch"]),
                      "study_hyperparameters", "Hyper-parameter study (SE-CRNN, validation set only, 3 seeds)"))
    strat = [("crnn_attention", "noise + SpecAugment"), ("crnn_no_specaug", "noise only"), ("crnn_no_noise", "SpecAugment only"),
             ("svm_noise", "SVM: 2 noisy copies"), ("svm_clean", "SVM: none")]
    rows = [[lbl, fmt(R.val_score(n)), fmt(R.stat(n, "clean")[0]), fmt(R.stat(n, "-5dB")[0]), fmt(R.stat(n, "pink_0dB")[0])]
            for n, lbl in strat if n in R.have]
    out.append(md(pd.DataFrame(rows, columns=["Augmentation strategy", "Val selection F1", "Test clean F1", "Test −5 dB F1", "Test pink 0 dB F1"]),
                  "study_data_strategy", "Data-strategy study (augmentation), mean of 3 seeds"))
    if {"efficientnet_pretrained", "efficientnet_scratch"} <= set(R.have):
        rows = [[R.disp(n), fmt(R.val_score(n))] + [fmt(*R.stat(n, c)) for c in ["clean", "-5dB", "pink_0dB", "overall"]]
                for n in ["efficientnet_pretrained", "efficientnet_scratch"]]
        out.append(md(pd.DataFrame(rows, columns=["Model", "Val selection F1", "Clean", "−5 dB", "Pink 0 dB", "Mean"]),
                      "study_pretrained_vs_scratch", "Pretrained vs scratch (same architecture, same budget)"))
    return out


def hypotheses(R, best):
    H = []

    def add(h, text, d, extra=""):
        m, sd, lo, hi = ci(d)
        H.append([h, text, f"Δ = {fmt(m, sd)} pp, 95% CI [{fmt(lo)}, {fmt(hi)}]{extra}", verdict(d)])

    if {"efficientnet_pretrained", "efficientnet_scratch"} <= set(R.have):
        add("H1", "Pretraining improves noisy-sound classification on small data (EfficientNet-B0 ImageNet vs scratch, mean F1)",
            R.paired("efficientnet_pretrained", "efficientnet_scratch", "overall"))
    if {"crnn_attention", "crnn_no_noise"} <= set(R.have):
        clean = ci(R.paired("crnn_attention", "crnn_no_noise", "clean"))[0]
        d = R.paired("crnn_attention", "crnn_no_noise", "-5dB")
        add("H2", "Noise augmentation raises −5 dB macro-F1 while costing < 2 pp on clean audio (SE-CRNN)", d,
            f"; clean Δ = {fmt(clean)} pp")
        if clean < -0.02 and H[-1][3] == "Supported":
            H[-1][3] = "Partly supported (clean cost > 2 pp)"
    if {"crnn_attention", "crnn_no_attention"} <= set(R.have):
        add("H3", "Attention pooling improves robustness at −5 dB vs mean pooling (SE-CRNN)",
            R.paired("crnn_attention", "crnn_no_attention", "-5dB"))
    d = (R.per_seed(best, "0dB") - R.per_seed(best, "pink_0dB")).values
    k = sum(R.stat(n, "0dB")[0] > R.stat(n, "pink_0dB")[0] for n in R.main)
    add("H4", f"Models trained with white noise lose accuracy on unseen pink noise at the same SNR ({R.disp(best)}, white 0 dB − pink 0 dB)",
        d, f"; pink < white in {k}/{len(R.main)} models")
    return md(pd.DataFrame(H, columns=["Hypothesis", "Statement", "Evidence (test, paired by seed)", "Verdict"]),
              "hypotheses", "Hypothesis verdicts"), H


def error_analysis(R, best, seed, labels):
    p = pd.read_csv(run_dir(best, seed) / "test_predictions.csv")
    name = lambda i: labels[i]
    ex = lambda d, c: f"{d.filename.iloc[0]}: {name(d.true.iloc[0])} → {name(d[c].iloc[0])}" if len(d) else "–"
    clean_err = p[p.pred_clean != p.true]
    masked = p[(p.pred_clean == p.true) & (p["pred_-5dB"] != p.true)]
    attract = p[(p["pred_-5dB"] != p.true) & p["pred_-5dB"].map(lambda i: name(i) in BROADBAND)]
    pink = p[(p["pred_0dB"] == p.true) & (p["pred_pink_0dB"] != p.true)]
    pairs = p[p["pred_-5dB"] != p.true].groupby(["true", "pred_-5dB"]).size().sort_values(ascending=False)
    top = pairs.index[0] if len(pairs) else None
    rows = [["Intrinsic confusion (wrong even on clean audio)", len(clean_err), ex(clean_err, "pred_clean"),
             "acoustically similar classes / few (40) examples per class"],
            ["Noise masking (correct clean → wrong at −5 dB)", len(masked), ex(masked, "pred_-5dB"),
             "low-energy or transient events buried below the noise floor"],
            ["Broadband attractor (−5 dB error predicted as a noise-like class)", len(attract), ex(attract, "pred_-5dB"),
             "white noise resembles wind/rain/engine textures, so the model labels the noise, not the event"],
            ["Unseen noise colour (correct at white 0 dB → wrong at pink 0 dB)", len(pink), ex(pink, "pred_pink_0dB"),
             "pink noise concentrates energy at low frequencies, masking bands not covered by training noise"]]
    if top is not None:
        rows.append([f"Most frequent −5 dB confusion pair", int(pairs.iloc[0]), f"{name(top[0])} → {name(top[1])}",
                     "overlapping spectral envelope once fine structure is masked"])
    pairs.head(10).rename("count").reset_index().assign(true=lambda d: d.true.map(name), predicted=lambda d: d["pred_-5dB"].map(name))[
        ["true", "predicted", "count"]].to_csv(TAB / "error_pairs_-5dB.csv", index=False)
    t = md(pd.DataFrame(rows, columns=["Error type", "Count (of 400)", "Example", "Likely cause"]), "table6_error_analysis",
           f"Table 6 — Error analysis: {R.disp(best)}, seed {seed}, test set")
    # spectrogram examples of noise-masking failures
    if len(masked):
        X, df = waveforms()
        fe, sel = LogMel(), masked.head(3)
        fig, ax = plt.subplots(len(sel), 2, figsize=(10, 2.6 * len(sel)), squeeze=False)
        test_files = list(df.filename.values[condition_set("test", "clean")[1]])
        for r, (_, row) in enumerate(sel.iterrows()):
            i = test_files.index(row.filename)
            for c, cond in enumerate(["clean", "-5dB"]):
                z = fe(condition_set("test", cond)[0][i:i + 1].float())[0, 0].numpy()
                ax[r, c].imshow(z, origin="lower", aspect="auto", cmap="magma")
                pred = row[f"pred_{cond}"]
                ax[r, c].set_title(f"{row.filename} [{cond}] true={name(row.true)} pred={name(pred)}", fontsize=8)
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        fig.tight_layout(); fig.savefig(FIG / "error_examples.png", dpi=130); plt.close(fig)
    return t, rows


def table7(R):
    rows = []
    for n in R.main:
        cs = [v for (m, _), v in R.cost.items() if m == n]
        c0 = cs[0]
        mem = c0.get("peak_mem_mb")
        rows.append([R.disp(n), f"{c0['params']:,}", f"{c0['size_mb']:.1f}", f"{np.mean([c['train_seconds'] for c in cs]) / 60:.1f}",
                     f"{np.mean([c.get('infer_ms_per_clip', np.nan) for c in cs]):.1f}", f"{mem:.0f}" if mem else "–",
                     c0["hardware"] + (f" (inference: {c0['infer_hardware']})" if c0.get("infer_hardware") not in (None, c0["hardware"]) and c0["hardware"] != "CPU" else "")])
    return md(pd.DataFrame(rows, columns=["Model", "Parameters", "Size (MB)", "Train time (min/seed)", "Inference (ms/clip)",
                                          "Peak GPU mem (MB)", "Hardware"]), "table7_cost",
              "Table 7 — Computational cost (inference = raw 5 s waveform → logits, batch 1, warmed)")


def figures(R, best, seed, labels):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in R.main:
        st = [R.stat(n, c) for c in CONDITIONS]
        ax.errorbar(range(6), [s[0] for s in st], yerr=[0 if np.isnan(s[1]) else s[1] for s in st], marker="o", capsize=3, label=R.disp(n))
    ax.set_xticks(range(6), ["clean", "10 dB", "5 dB", "0 dB", "−5 dB", "pink 0 dB\n(unseen)"]); ax.axvline(4.5, ls="--", c="gray")
    ax.set_ylabel("Test macro-F1"); ax.set_title("Robustness to noise (mean ± SD, 3 seeds)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "snr_curves.png", dpi=150); plt.close(fig)

    pc = R.pc[(R.pc.model == best) & (R.pc.condition.isin(["clean", "-5dB"]))].groupby(["class", "condition"]).f1.mean().unstack()
    pc = pc.sort_values("-5dB")
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.barh(pc.index, pc["clean"], color="#9ecae1", label="clean"); ax.barh(pc.index, pc["-5dB"], color="#08519c", label="−5 dB")
    ax.set_xlabel("F1"); ax.set_title(f"Per-class F1 — {R.disp(best)}"); ax.legend(); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / "per_class_f1.png", dpi=150); plt.close(fig)
    worst = pc.head(10).reset_index()
    md(worst.assign(clean=worst["clean"].map(fmt), **{"-5dB": worst["-5dB"].map(fmt)})[["class", "clean", "-5dB"]],
       "per_class_worst10", f"Ten hardest classes at −5 dB — {R.disp(best)} (F1 %, mean of seeds)")

    from sklearn.metrics import confusion_matrix
    p = pd.read_csv(run_dir(best, seed) / "test_predictions.csv")
    cm = confusion_matrix(p.true, p["pred_-5dB"], labels=range(50))
    fig, ax = plt.subplots(figsize=(11, 10)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(50), labels, rotation=90, fontsize=6); ax.set_yticks(range(50), labels, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix at −5 dB — {R.disp(best)} (seed {seed})")
    fig.tight_layout(); fig.savefig(FIG / "confusion_-5dB.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in R.main + [m for m in R.have if R.cfgs[m]["role"] == "hp"]:
        h = run_dir(n, R.cfgs[n]["seeds"][0]) / "history.csv"
        if not h.exists() or R.cfgs[n]["model"] in ("svm", "ast"):
            continue
        h = pd.read_csv(h)
        ax[0].plot(h.epoch, h.loss, label=R.disp(n)); ax[1].plot(h.epoch, h.selection, label=R.disp(n))
    ax[0].set(title="Training loss (seed 42)", xlabel="epoch"); ax[1].set(title="Validation selection macro-F1", xlabel="epoch")
    ax[1].legend(fontsize=7); [a.grid(alpha=.3) for a in ax]
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=150); plt.close(fig)


def main():
    R = Results()
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    X, df = waveforms()
    labels = class_names(df)
    best = max(R.main, key=R.val_score)                       # chosen on VALIDATION, never on test
    seed = max((s for (n, s) in R.val if n == best), key=lambda s: R.val[best, s]["selection"])
    deploy = {n: max((s for (m, s) in R.val if m == n), key=lambda s: R.val[n, s]["selection"]) for n in R.have}
    (RES / "deploy.json").write_text(json.dumps({"best": best, "seeds": deploy}, indent=2))
    figures(R, best, seed, labels)
    parts = [table1(), table2(R), table3(R), table4(R), table5(R)]
    t6, err_rows = error_analysis(R, best, seed, labels)
    parts += [t6, table7(R)] + study_tables(R)
    ht, H = hypotheses(R, best)
    parts.append(ht)
    ranking = sorted(R.main, key=lambda n: -R.stat(n, "overall")[0])
    summary = {"best": best, "best_display": R.disp(best), "deploy_seed": seed,
               "ranking": [[R.disp(n), R.stat(n, "overall")[0]] for n in ranking],
               "stats": {n: {c: R.stat(n, c) for c in CONDITIONS + ["overall"]} for n in R.have},
               "display": {n: R.disp(n) for n in R.have}, "hypotheses": H, "errors": err_rows,
               "n_seeds": int(R.test.groupby("model").seed.nunique().min())}
    (TAB / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (TAB / "ALL_TABLES.md").write_text("\n\n".join(parts), encoding="utf-8")
    print("\n\n".join(parts))
    print("\nTables ->", TAB, "\nFigures ->", FIG)


if __name__ == "__main__":
    main()
