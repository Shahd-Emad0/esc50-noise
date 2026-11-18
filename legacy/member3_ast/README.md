# Environmental Sound Classification under Noise — Member 3

Final modern approach: frozen AudioSet AST with trained 50-class head. Test clean accuracy: 95.83% ± 0.14 percentage points across 3 seeds. This package excludes demo work by request.

## Use this package

Extract ESC50_Member3_AST. Open its modern/ folder in VS Code. This is a standalone handoff, not a request to overwrite your current working folder. Raw audio, .venv and pretrained encoder are excluded. Head checkpoints, cached embeddings and all received results are included.

Python 3.10, PyTorch/torchaudio 2.5.1+cu121, torchvision 0.20.1+cu121, Transformers 4.46.3, timm 1.0.11. Use your existing functioning environment. For a fresh environment install CUDA libraries first from the official PyTorch wheel index, then requirements.txt.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

ESC-50 audio source: https://github.com/karolpiczak/ESC-50 . Keep LICENSE/attributions. Pass the folder of WAV files to --audio. No raw audio is bundled. Read docs/DATASET_CARD.md and team split protocol. The manifest's SHA-256 must match the checkpoint.

## Commands (run from modern/)

Reproduce training, only when needed:

```powershell
foreach ($seed in @(42,43,44)) {
  .\.venv\Scripts\python.exe train_ast_probe.py --audio "C:\path\to\ESC-50\audio" --seed $seed
  if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
}
```

The first feature extraction needs approximately 30 minutes per seed on the measured laptop; cached head training takes a few seconds. Different hardware/cache state changes runtime. Original model resolution is recorded in protocol.json; original training script resolves the model repository's default revision, while the final evaluation explicitly pins checkpoint revisions.

Final evaluation already completed; reproduce only to verify unchanged checkpoints/protocol, never to tune:

```powershell
.\.venv\Scripts\python.exe evaluate_ast_test.py --audio "C:\path\to\ESC-50\audio"
```

Supplementary validation-only studies already completed; reproduce only if necessary (no audio/encoder recomputation required):

```powershell
.\.venv\Scripts\python.exe run_cached_studies.py
```

Read docs/MODERN_APPROACH_REPORT.md for verified results, architecture, costs, limitations and remaining requirements. Study outputs are included and validated for seeds 42, 43 and 44; see docs/SUPPLEMENTARY_STUDIES.md. No AST scratch comparison is claimed. Contributions log is blank intentionally: enter actual hours and obtain teammate co-signature.

## Credits

AST: Gong et al. (2021), MIT AudioSet checkpoint; Hugging Face Transformers implementation. Dataset: Piczak ESC-50. See report references. src/core.py adapts public libraries and provides a custom SNR mixer. Model architecture is library-provided, not claimed as original code.
