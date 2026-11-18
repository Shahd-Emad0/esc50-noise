## Completed supplementary AST-head studies

These validation-only exploratory studies were conducted after final test results had been viewed. They do not modify the reported final model, and do not justify further tuning on the final test. All four arms use the same frozen cached AST features, 30 head epochs, batch size 64, 2560 feature examples and 40 optimizer steps per epoch. Three paired seeds are used. In the no-noise arm, each noisy partner is replaced by its clean partner, preserving the example/update budget. All arms share the same explicit head initialization per seed; this differs from the random-number consumption of the original Hugging Face model initialization, so the supplementary reference is not the original final head.

### Learning-rate study and overall ablation

Equally weighted mean validation Macro F1 over the five conditions, then mean and sample SD over three seeds; values are percentages/percentage points.

| Study | Mean Macro F1 (%) | Seed SD (pp) |
|---|---|---|
| reference | 87.65 | 0.76 |
| lr_3e4 | 87.30 | 0.23 |
| lr_3e3 | 87.22 | 0.79 |
| no_noise | 73.54 | 1.85 |

The reference learning rate 0.001 has the highest observed mean. Differences between learning-rate settings are small; no statistical superiority is claimed from three seeds. This supports retaining the already frozen configuration rather than changing it after test evaluation.

### Noise augmentation ablation

| Condition | Augmented accuracy (%) | No-noise accuracy (%) | Accuracy gain (pp) | Paired Macro F1 gain ± SD (pp) |
|---|---|---|---|---|
| clean | 94.69 | 95.00 | -0.31 | -0.37 ± 0.60 |
| 10dB | 91.88 | 85.00 | +6.88 | +6.88 ± 0.36 |
| 5dB | 90.31 | 76.98 | +13.33 | +13.25 ± 0.75 |
| 0dB | 85.31 | 63.54 | +21.77 | +19.31 ± 1.67 |
| -5dB | 77.29 | 43.02 | +34.27 | +31.46 ± 3.68 |

At -5 dB the augmentation strategy improves mean validation accuracy by 34.27 percentage points, while clean accuracy decreases by 0.31 points. This is evidence for noise augmentation under this fixed-embedding and fixed-noise protocol, not proof of real-world noise robustness. Paired deltas are computed per seed before averaging. The number of clean recordings is unchanged between arms.

This completes the implemented AST-head learning-rate, data-strategy and augmentation-ablation studies. It does not test random AST encoder training or remove AudioSet pretraining. No AST pretrained-versus-scratch result is available.
