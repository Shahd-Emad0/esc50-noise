# AST review and next steps

Verified all 15 test metric rows by recomputing accuracy and macro F1 from prediction CSVs: 400 clips x 5 conditions x 3 seeds. These are repeated predictions of 400 test recordings, not 6000 independent recordings.

## Final test summary

```text
condition  seeds  accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
    clean      3       0.958333     0.001443       0.957261     0.001715
     10dB      3       0.931667     0.003819       0.930681     0.003669
      5dB      3       0.901667     0.011547       0.901373     0.011341
      0dB      3       0.857500     0.013229       0.859263     0.013452
     -5dB      3       0.762500     0.006614       0.771284     0.009034
```

## Deployment

Seed 43 selected using validation mean Macro F1 (0.882099), not test results. Use ast_demo.py from modern/. Command:

```powershell
.\.venv\Scripts\python.exe ast_demo.py
```

Open Gradio's local URL. Upload or record audio. The first five seconds are used. Optional white noise is a demo stress test. Scores are uncalibrated, and unfamiliar sounds may receive confident wrong labels. Backbone download/cache is required; no external prediction API is used.

86,227,250 total parameters; 39,986 trained parameters. This is a frozen pretrained encoder with a trained LayerNorm/linear head. best_head.pth alone is not the whole model. Full state tensors: 344,909,000 bytes (~329 MiB) before serialization overhead.

## Error analysis at -5 dB

Counts pool error predictions across seeds; repeated errors of the same clip are not independent examples.

```text
true_category     predicted_category
door_wood_creaks  engine                10
rain              wind                  10
train             wind                   7
chirping_birds    wind                   6
airplane          helicopter             6
glass_breaking    fireworks              6
crickets          wind                   5
can_opening       crackling_fire         5
engine            helicopter             5
helicopter        engine                 5
```

Acoustic masking and similar patterns are possible explanations, not verified causes without listening to the examples.

## Computational cost

- Seed 42: feature extraction stage 1798.6 s; cached-head training and validation 3.19 s.
- Seed 43: feature extraction stage 1791.1 s; cached-head training and validation 3.29 s.
- Seed 44: feature extraction stage 1817.9 s; cached-head training and validation 8.34 s.

Warmed model-only inference approximately 445 ms/clip, batch size 1. Excludes preprocessing and transfers. Encoder timing shared across identical backbones; not three independent timing measurements.

## Remaining requirements

- Run and verify uploaded-audio and microphone demo on the target laptop.
- Listen to concrete mistakes and write supported error interpretations.
- AST-head learning-rate study and matched augmentation ablation are now completed in supplementary_studies. AST pretrained-versus-scratch remains unperformed. Existing EfficientNet studies must not be relabeled AST scratch experiments.
- Team-wide fair comparison still needs agreed representations, noise protocol and tuning budgets. AST uses 16 kHz and EfficientNet 22.05 kHz; their synthetic noise samples differ.
- Three seeds vary the trained head, shuffling and augmentation with fixed pretrained encoder and fixed evaluation noise.
- Synthetic white noise only; no real-noise evaluation established.
- Complete team report/slides, source citations, contribution log, peer evaluation and classic/advanced model integration.
- Keep final test results for reporting, not further tuning.

## Verification limits

Results recomputed from CSV files; demo Python syntax checked. AST/Gradio runtime was not available here and requires local verification. No model weights were changed.
