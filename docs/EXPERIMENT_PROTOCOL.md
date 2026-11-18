# Final experiment protocol (team-wide)

1. **Split**: `data/split_manifest.csv`, group-aware on `src_file`; test read only by `src/evaluate.py` after all training.
2. **Inputs**: one waveform cache; identical seeded corruptions for validation/test (clean, white 10/5/0/−5 dB, pink 0 dB).
3. **Budget** (`configs/_defaults.yaml`): AdamW, OneCycle peak 1e-3, wd 0.01, batch 32, 60 epochs, CE loss, noise aug p=0.5 + SpecAugment. No per-model tuning. AST head (Member 3): AdamW 1e-3, 30 epochs, batch 64 on cached frozen embeddings.
4. **Selection**: checkpoint with best validation mean macro-F1 over clean + 4 white SNRs (from epoch > 25 %). Best model and demo seed also chosen on validation.
5. **Seeds**: 42, 43, 44 for every final configuration; report mean ± SD and paired 95 % t-intervals.
6. **Studies**: LR {3e-4, 1e-3, 3e-3} (validation); augmentation strategy {noise+SpecAug, noise, SpecAug} and SVM {clean, noisy copies}; pretrained vs scratch (EfficientNet-B0).
7. **Ablation**: attention→mean pooling, −noise aug, −SpecAugment, −pretraining.
8. **Metrics**: macro-F1 (primary), accuracy, macro P/R, top-5, per-class F1, confusion matrices; cost: params, size, train time, inference ms/clip (batch 1, end-to-end), peak GPU memory, hardware.
