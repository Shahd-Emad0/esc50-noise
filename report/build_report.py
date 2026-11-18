"""Fills report_template.md with the real results -> <Team>_Report.md/.docx(/.pdf if LibreOffice exists) + <Team>_Slides.pptx."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from pptx import Presentation
from pptx.util import Inches as PI, Pt as PP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.esc50 import EXP  # noqa: E402

TAB, FIG = EXP / "tables", EXP / "figures"
OUT = ROOT / "report" / ("_quick" if EXP.name == "_quick" else "")
TEAM = yaml.safe_load(open(ROOT / "team.yaml", encoding="utf-8"))
S = json.loads((TAB / "summary.json").read_text())


def t(name):
    p = TAB / f"{name}.md"
    return p.read_text(encoding="utf-8") if p.exists() else "_(not available — run the corresponding experiment)_"


def pct(model, cond):
    return f"{100 * S['stats'][model][cond][0]:.1f}" if model in S["stats"] else "–"


def findings():
    st, d = S["stats"], S["display"]
    f = [f"- Best model on validation: **{S['best_display']}** (deployed seed {S['deploy_seed']})."]
    for n, c in [(S["best"], "best model"), ("svm_noise", "noise-augmented baseline"), ("cnn", "classic CNN")]:
        if n in st:
            f.append(f"- {d[n]} ({c}): clean {pct(n, 'clean')} % → −5 dB {pct(n, '-5dB')} % "
                     f"(drop {100 * (st[n]['clean'][0] - st[n]['-5dB'][0]):.1f} pp); unseen pink 0 dB {pct(n, 'pink_0dB')} %.")
    if {"efficientnet_pretrained", "efficientnet_scratch"} <= set(st):
        g = st["efficientnet_pretrained"]["overall"][0] - st["efficientnet_scratch"]["overall"][0]
        f.append(f"- ImageNet pretraining changes EfficientNet-B0's mean macro-F1 by {100 * g:+.1f} pp.")
    if {"crnn_attention", "crnn_no_noise"} <= set(st):
        g = st["crnn_attention"]["-5dB"][0] - st["crnn_no_noise"]["-5dB"][0]
        f.append(f"- Noise augmentation changes the SE-CRNN's −5 dB macro-F1 by {100 * g:+.1f} pp.")
    if {"crnn_attention", "crnn_no_attention"} <= set(st):
        g = st["crnn_attention"]["-5dB"][0] - st["crnn_no_attention"]["-5dB"][0]
        f.append(f"- Attention pooling vs mean pooling at −5 dB: {100 * g:+.1f} pp.")
    for h in S["hypotheses"]:
        f.append(f"- {h[0]}: **{h[3]}** — {h[2]}.")
    return "\n".join(f)


def fill():
    members = "| Name | Student ID | Main responsibility |\n|---|---|---|\n" + "".join(
        f"| {m['name']} | {m['id']} | {m['role']} |\n" for m in TEAM["members"])
    pairs = pd.read_csv(TAB / "error_pairs_-5dB.csv") if (TAB / "error_pairs_-5dB.csv").exists() else pd.DataFrame()
    hw = sorted({json.loads(p.read_text()).get("hardware", "") for p in (EXP / "runs").glob("*/*/cost.json")})
    rep = {"TEAM": TEAM["team_name"], "MEMBERS": members, "GITHUB": TEAM.get("github", ""), "N_SEEDS": str(S["n_seeds"]),
           "BEST": S["best_display"], "BEST_CLEAN": pct(S["best"], "clean"), "BEST_M5": pct(S["best"], "-5dB"),
           "BEST_PINK": pct(S["best"], "pink_0dB"), "SVM_CLEAN": pct("svm_noise", "clean"), "SVM_M5": pct("svm_noise", "-5dB"),
           "HARDWARE": ", ".join(h for h in hw if h), "FINDINGS": findings(),
           "RANKING": "\n".join(f"{i + 1}. {n} — {100 * v:.1f} %" for i, (n, v) in enumerate(S["ranking"])),
           "ERR_PAIRS": "\n".join(f"- {r.true} → {r.predicted}: {r['count']}" for _, r in pairs.head(6).iterrows()),
           "T1": t("table1_datasets"), "T2": t("table2_architectures"), "T3": t("table3_hyperparameters"),
           "T4": t("table4_main_results"), "T5": t("table5_ablation"), "T6": t("table6_error_analysis"), "T7": t("table7_cost"),
           "HP": t("study_hyperparameters"), "DATA": t("study_data_strategy"), "PVS": t("study_pretrained_vs_scratch"),
           "HYP": t("hypotheses"), "WORST": t("per_class_worst10")}
    text = (ROOT / "report" / "report_template.md").read_text(encoding="utf-8")
    for k, v in rep.items():
        text = text.replace("{{" + k + "}}", v)
    return re.sub(r"\{\{FIG:([^|]+)\|([^}]+)\}\}",
                  lambda m: f"![{m.group(2)}]({(FIG / m.group(1)).as_posix()})" if (FIG / m.group(1)).exists() else "", text)


def add_runs(par, text):
    for i, part in enumerate(re.split(r"\*\*", text)):
        par.add_run(part).bold = i % 2 == 1


def to_docx(text, path):
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
    lines, i = text.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|(-+\|)+$", lines[i].replace(" ", "")):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            tb = doc.add_table(rows=len(rows), cols=len(rows[0])); tb.style = "Light Grid Accent 1"
            for r, row in enumerate(rows):
                for c, val in enumerate(row[:len(rows[0])]):
                    cell = tb.cell(r, c); cell.text = ""
                    run = cell.paragraphs[0].add_run(val.replace("**", "")); run.font.size = Pt(7.5); run.bold = r == 0
            doc.add_paragraph()
            continue
        m = re.match(r"!\[(.*)\]\((.*)\)", ln)
        if m:
            doc.add_picture(m.group(2), width=Inches(6.2))
            cap = doc.add_paragraph(m.group(1)); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER; cap.runs[0].italic = True
        elif ln.startswith("#"):
            doc.add_heading(ln.lstrip("#").strip(), level=0 if text.startswith(ln) else min(ln.count("#"), 3))
        elif re.match(r"^(- |\d+\. )", ln):
            add_runs(doc.add_paragraph(style="List Bullet" if ln.startswith("- ") else "List Number"), re.sub(r"^(- |\d+\. )", "", ln))
        elif ln.strip():
            add_runs(doc.add_paragraph(), ln)
        i += 1
    doc.save(path)


def slides(path):
    prs = Presentation(); prs.slide_width, prs.slide_height = PI(13.33), PI(7.5)
    ms = [m["name"] for m in TEAM["members"]]
    who = lambda k: f"Presenter: {ms[k % len(ms)]}"

    def slide(title, bullets=(), img=None, table=None, note=""):
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title; s.shapes.title.text_frame.paragraphs[0].font.size = PP(30)
        left = 0.5
        if img and (FIG / img).exists():
            s.shapes.add_picture(str(FIG / img), PI(6.4 if bullets else 2.2), PI(1.4), **({"width": PI(6.7)} if bullets else {"height": PI(5.6)})); left = 0.5
        if table is not None and (TAB / f"{table}.csv").exists():
            df = pd.read_csv(TAB / f"{table}.csv").astype(str)
            shp = s.shapes.add_table(len(df) + 1, len(df.columns), PI(0.4), PI(1.4 if not bullets else 3.6), PI(12.5), PI(0.3 * (len(df) + 1)))
            for c, col in enumerate(df.columns):
                shp.table.cell(0, c).text = col
            for r, row in enumerate(df.values):
                for c, v in enumerate(row):
                    shp.table.cell(r + 1, c).text = v
            for cell in shp.table.iter_cells():
                for p in cell.text_frame.paragraphs:
                    p.font.size = PP(10)
        if bullets:
            tf = s.shapes.add_textbox(PI(left), PI(1.4), PI(5.8 if img else 12.3), PI(2 if table else 5.6)).text_frame; tf.word_wrap = True
            for k, b in enumerate(bullets):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); p.text = "• " + b; p.font.size = PP(16 if img else 18)
        s.notes_slide.notes_text_frame.text = note

    st, best = S["stats"], S["best"]
    slide(f"Environmental Sound Classification under Noise — Team {TEAM['team_name']}",
          ["Deep Learning AI403 · Final Project · Idea 17", *ms, "Pharos University in Alexandria — Dr. Sahar Ghanem"])
    slide("1. Problem & motivation", ["Classify 5-s sounds into 50 everyday classes (ESC-50)", "Real microphones add noise: accuracy must hold at low SNR",
          "Only 40 clips per class → is pretraining worth it?", "Input: waveform → Output: 1 of 50 classes"], note=who(0))
    slide("Hypotheses", [f"{h[0]}: {h[1]}" for h in S["hypotheses"]], note=who(0))
    slide("2. Data: ESC-50 & leakage-free split", ["2,000 clips, 50 balanced classes, CC BY-NC", "Group split on source recording: 1,280 / 320 / 400 (64/16/20 %)",
          "Asserted every run: no shared source, all classes in every split, manifest SHA", "Noise: white 10/5/0/−5 dB (train+test), pink 0 dB (test only)",
          "Our own SNR-exact noise mixer; identical noisy test clips for every model"], note=who(1))
    slide("Models (3 families + baseline)", table="table2_architectures", note=who(1))
    ep = next((json.loads(p.read_text()).get("epochs") for p in (EXP / "runs" / "cnn").glob("*/cost.json")), 60)
    slide("Fair protocol", ["Same split, waveforms, noise samples, metrics for all models", f"AdamW 1e-3, OneCycle, {ep} epochs, batch 32, same augmentation",
          "Checkpoint selection on VALIDATION (mean macro-F1 over 5 conditions)", "Test set read once at the end; 3 seeds (42/43/44) → mean ± SD"], note=who(1))
    slide("3. Main results (test macro-F1 %, mean ± SD)", table="table4_main_results", note=who(2))
    slide("Robustness across noise levels", [f"Best (validation-chosen): {S['best_display']}",
          f"Clean {pct(best, 'clean')} % → −5 dB {pct(best, '-5dB')} %", f"Unseen pink 0 dB: {pct(best, 'pink_0dB')} %",
          f"Baseline SVM (noise-aug): clean {pct('svm_noise', 'clean')} %, −5 dB {pct('svm_noise', '-5dB')} %"], img="snr_curves.png", note=who(2))
    slide("Studies: pretrained vs scratch · learning rate · augmentation", table="study_data_strategy",
          bullets=[l for l in findings().replace("**", "").replace("- ", "").splitlines() if "changes" in l or "vs mean pooling" in l], note=who(2))
    slide("4. Ablation study", table="table5_ablation", note=who(3))
    slide("Error analysis", [f"{r[0]}: {r[1]}" for r in S["errors"]], img="error_examples.png", note=who(3))
    slide("Where the best model fails at −5 dB", img="confusion_-5dB.png", note=who(3))
    slide("Computational cost", table="table7_cost", note=who(4))
    slide("5. Conclusion", [f"{h[0]} → {h[3]}" for h in S["hypotheses"]] + ["Pretrained audio models + noise augmentation = most robust",
          "Limitations: synthetic noise, 3 seeds, frozen AST"], note=who(4))
    slide("6. Live demo", ["python demo/app.py → upload / record a sound", "Choose model, add white or pink noise, set SNR",
          "Top-5 classes + the spectrogram the model sees"], note=who(4))
    prs.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    name = TEAM["team_name"]
    text = fill()
    (OUT / f"{name}_Report.md").write_text(text, encoding="utf-8")
    to_docx(text, OUT / f"{name}_Report.docx")
    slides(OUT / f"{name}_Slides.pptx")
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if office:
        subprocess.run([office, "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(OUT / f"{name}_Report.docx")], capture_output=True)
    pdf = OUT / f"{name}_Report.pdf"
    print("Report :", OUT / f"{name}_Report.docx", "(PDF ready)" if pdf.exists() else "-> open in Word: File > Save As > PDF")
    print("Slides :", OUT / f"{name}_Slides.pptx")


if __name__ == "__main__":
    main()
