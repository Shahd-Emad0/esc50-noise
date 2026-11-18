# Environmental Sound Classification under Noise (ESC-50)

**Deep Learning AI403 — Final Project, Fall 2026 — Pharos University in Alexandria**
**Idea 17:** Environmental sound classification under noise (audio → class)
**Team D07:** Rania Hamdy Gomaa (202302910), Heba Atef Farouk (202301959), Roqaya Ebrahim Elghandour (202302796), Nour Osama Hassan (202301336), Shahd Emad (202301173) · TA: Eng. Mohamed Ashraf · Supervisor: Dr. Sahar Ghanem

## 1. Task and best result

Classify 5-second clips into 50 everyday sound classes (ESC-50) and stay accurate when noise is added. Every model is tested on the same 400 held-out clips under 6 fixed conditions:
- clean audio;
- white noise at 10, 5, 0 and −5 dB SNR;
- pink noise at 0 dB, a noise type that no model saw during training.

Models compared (3 seeds each):

| Category | Model |
|---|---|
| Baseline | MFCC + SVM |
| Classic | CNN trained from scratch |
| Modern | EfficientNet-B0, ImageNet-pretrained and from scratch |
| Modern | AST (AudioSet-pretrained, frozen) + trained head |
| Advanced (ours) | SE-CRNN with team-written attention pooling |

**Final result (3 seeds, chosen on validation):** AST (AudioSet, frozen) + trained head, test macro-F1 96.0 ± 0.2 % clean, 79.4 ± 1.4 % at −5 dB white noise and 87.8 ± 0.1 % on unseen pink noise at 0 dB. The noise-augmented SVM baseline gets 41.5 % / 36.5 % / 21.4 %.

After a run, the best model and all numbers are in `experiments/tables/ALL_TABLES.md` and `report/<Team>_Report.docx`. Member 3's earlier result for AST on clean audio was 95.8 % ± 0.1 test accuracy.

## 2. One-command setup and run

You need Python 3.10–3.12 and internet access for the first run (ESC-50 is about 650 MB, plus the pretrained weights).

```powershell
# Windows (PowerShell, from this folder)
powershell -ExecutionPolicy Bypass -File setup.ps1          # full project
powershell -ExecutionPolicy Bypass -File setup.ps1 -Quick   # 10-min smoke test first (recommended)
```
```bash
# Linux / macOS
bash setup.sh            # or: bash setup.sh --quick
```

The setup script does everything in one go:
- creates `.venv`;
- installs PyTorch 2.5.1 (CUDA 12.1 build, which also works on CPU) and `requirements.txt`;
- downloads and audits the data;
- trains every model and seed;
- evaluates once on the test set;
- builds the tables and figures, the report (`.md`/`.docx`) and the slides (`.pptx`).

The run is resumable: if it stops, run it again and finished runs are skipped. After editing `team.yaml`, run `python run_all.py --report` to rebuild the report and slides with your names.

## 3. Environment

`requirements.txt` pins torch 2.5.1, torchaudio 2.5.1, timm 1.0.11, transformers 4.46.3, scikit-learn, librosa, gradio, python-docx and python-pptx. Activate the environment with `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`.

## 4. Dataset download and preprocessing

```bash
python -c "from src.datasets.esc50 import download, load_manifest, waveforms; download(); load_manifest(); waveforms()"
```

The data files are:
- **Audio:** downloaded from https://github.com/karolpiczak/ESC-50 into `data/ESC-50-master/`. Raw data is not committed.
- **Split:** `data/split_manifest.csv` is the team split (StratifiedGroupKFold on `src_file`, 1,280/320/400). It is verified by SHA-256 on every run, together with a no-source-overlap check and an all-classes-present check. `legacy/member1_data_baseline/scripts/02_make_group_split_v2.py` regenerates it.
- **Cache:** `data/cache/waveforms_22050.npy` holds mono 22.05 kHz audio cut or padded to 5 s, so every model reads identical samples.

See `docs/DATASET_CARD.md` and `docs/EXPERIMENT_PROTOCOL.md`.

## 5. Training (one config file per experiment in `configs/`)

```bash
python -m src.train configs/svm_noise.yaml --seed 42
python -m src.train configs/cnn.yaml --seed 42
python -m src.train configs/crnn_attention.yaml --seed 42
python -m src.train configs/efficientnet_pretrained.yaml --seed 42
python -m src.train configs/efficientnet_scratch.yaml --seed 42
python -m src.train configs/ast_probe.yaml --seed 42    # registers Member 3's trained heads (retrain: legacy/member3_ast)
# ablations / studies: crnn_no_attention, crnn_no_noise, crnn_no_specaug, crnn_lr3e-4, crnn_lr3e-3, svm_clean
python run_all.py --only cnn crnn_attention                 # all seeds of chosen configs
```

## 6. Evaluation (shared by all models)

```bash
python -m src.evaluate      # test set, 6 conditions -> experiments/results/
python -m src.analysis      # Tables 1-7, studies, hypotheses, error analysis, figures -> experiments/tables, experiments/figures
```

## 7. Demo

```bash
python demo/app.py          # open http://127.0.0.1:7860
```

You can upload or record a sound, pick a model, optionally add white or pink noise at any SNR, and see the top-5 classes and the spectrogram. All inference runs locally; no external AI API is used.

## 8. Hardware and expected runtimes

Exact training time, inference time, memory and hardware are measured automatically for every run and reported in Table 7 (`experiments/tables/table7_cost.md`). Approximate full-run times on a 4 GB laptop GPU (Quadro P1000 class):

| Step | Time |
|---|---|
| SVMs | ~5 min |
| CNN and SE-CRNN runs | ~3–5 min each |
| EfficientNet runs | ~10–15 min each |
| AST test embeddings | ~20 min |
| **Total** | **~3 h** |

Measured final runs (Google Colab, Tesla T4, 60 epochs): about 6 min per CNN seed, 7.4 min per SE-CRNN seed and 5.5 min per EfficientNet seed. Training all trainable configurations took about 3.1 h in total (SVMs on CPU included).

CPU-only machines are much slower. There you can use `python run_all.py --epochs 30`, which is still fair because every model gets the same budget.

## 9. Credits

| Item | Source and licence |
|---|---|
| Dataset | ESC-50, K. J. Piczak (2015), CC BY-NC |
| AST | Gong et al. (2021), `MIT/ast-finetuned-audioset-10-10-0.4593` via Hugging Face Transformers (Wolf et al. 2020) |
| EfficientNet-B0 | Tan & Le (2019), ImageNet weights via timm (Wightman) |
| CRNN / SE / attention pooling | Designs follow Cakir et al. (2017), Hu et al. (2018) and Kong et al. (2020) |
| SpecAugment | Park et al. (2019), torchaudio transforms |

Written by the team:
- the noise mixer (`src/datasets/noise.py`);
- the attention pooling and SE-CRNN (`src/models/crnn_attention.py`);
- all training, evaluation and analysis code.

Full references are in the report.

## Repository layout

```
configs/        one YAML per experiment (+ _defaults.yaml = shared fair protocol)
data/           split manifest (download + cache created automatically)
src/datasets/   loading, audit, cache, noise mixer, log-mel
src/models/     baseline_svm, cnn, crnn_attention, efficientnet, ast_probe
src/train.py    training + validation-only selection      src/evaluate.py  shared test evaluation
src/analysis.py tables, hypotheses, error analysis, figures
experiments/    runs/<config>/<seed>/ (history, cost, val/test metrics, predictions), results/, tables/, figures/
checkpoints/    AST heads + final deployed checkpoints (full set: shared-drive link below)
demo/app.py     Gradio demo          report/  report + slides builder
contributions/  individual logs + peer evaluation      legacy/  original Member 1 / Member 3 handoffs
```

**Large files (all checkpoints):** Final checkpoints are in checkpoints/ in this repository; every other run can be reproduced with python run_all.py.
