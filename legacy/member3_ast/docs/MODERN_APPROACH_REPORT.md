# Modern Deep Learning Approach: Audio Spectrogram Transformer

Project: Environmental Sound Classification under Noise  
Author: Roqaya Ebrahim Elghandour  
Scope: Member 3 modern approach; excludes the team's demo, classic/advanced approaches and complete team report.

## Methodology

The final modern model is an Audio Spectrogram Transformer (AST) initialized from the MIT AudioSet checkpoint, revision f826b80d28226b62986cc218e5cec390b1096902. Its encoder remains frozen. A LayerNorm and a new 768-to-50 linear classifier are trained using cross entropy on ESC-50. There are 86,227,250 total parameters and 39,986 trainable parameters. This is pretrained feature transfer with a trained classification head, not full-transformer fine-tuning or a transformer implemented from scratch [1,2].

Audio is read as mono, resampled to 16 kHz, and cropped or zero-padded to five seconds. ASTFeatureExtractor produces 1024-by-128 features using the checkpoint's AudioSet normalization (mean -4.2677393, standard deviation 4.5689974). Five-second inputs are padded to the pretrained extractor's time length. The input contract is held constant between AST training and evaluation.

The manifest contains 1280 training, 320 validation and 400 test clips. Source IDs and filenames are audited for isolation/uniqueness. This establishes isolation within the provided ESC-50 split; overlap with pretraining data has not been independently audited. Parent documentation calls the split a proposal: team adoption must be confirmed for team comparisons [3].

## Custom component and noise strategy

The custom waveform mixer scales white Gaussian noise to a target SNR using signal and noise mean-square powers. Its scale is sqrt(P_signal / (P_noise * 10^(SNR/10))). No waveform clipping follows mixing, because clipping would alter the intended ratio. Silent inputs are left unchanged by this mixer.

Each training recording contributes a clean example and one deterministic noisy example at a selected SNR in {10,5,0,-5} dB, giving 2560 cached training embeddings per seed. The noisy copy is fixed within a run; new noise is not regenerated every epoch. All validation/test conditions use fixed filename/SNR-derived noise seed 2026, shared between the three AST runs. This tests controlled synthetic noise, not a real-noise corpus.

## Experimental setup

The classifier is trained with AdamW, learning rate 0.001, weight decay 0.001, cached-feature batch size 64, 30 epochs and no scheduler. Frozen encoder extraction uses one waveform at a time. Seeds 42, 43 and 44 vary initialization, shuffling and training noise. All runs retain the same data partition and evaluation noise. Each checkpoint is selected on the equally weighted mean validation Macro F1 over clean, 10, 5, 0 and -5 dB. Best epochs are 9, 26 and 10. Test labels are not used in selection.

## Final test results

Values below are percentages, mean ± sample standard deviation across three training seeds. SD is expressed in percentage points, not a confidence interval. There are 400 unique test clips, evaluated repeatedly under five conditions and three models.

| Condition | Accuracy (%) | Macro F1 (%) |
|---|---|---|
| clean | 95.83 ± 0.14 | 95.73 ± 0.17 |
| 10dB | 93.17 ± 0.38 | 93.07 ± 0.37 |
| 5dB | 90.17 ± 1.15 | 90.14 ± 1.13 |
| 0dB | 85.75 ± 1.32 | 85.93 ± 1.35 |
| -5dB | 76.25 ± 0.66 | 77.13 ± 0.90 |

Performance declines with stronger noise. The fixed-noise seed variation is small but does not establish robustness to different real noise types, devices or new acoustic domains. High clean accuracy cannot be interpreted as an open-world reliability guarantee.

## Error analysis

Error prediction records and confusion matrices are included for all seeds and conditions. The most frequent pairs at -5 dB include wood-door creaks -> engine (10 predictions across seeds), rain -> wind (10), and train -> wind (7). These counts combine repeated predictions of the same test clips; they are not counts of independent recordings. Broad-band masking and similar acoustic patterns are plausible hypotheses. Audio listening is still required to confirm case-specific interpretations.

## Computational cost

The full model state occupies 344,909,000 tensor bytes (~329 MiB), excluding serialization overhead. A 163,744-byte head checkpoint is only an adapter to the pinned pretrained encoder, not the complete model. Warmed model-only inference on Quadro P1000 is approximately 445 ms per clip at batch size one, excluding loading, waveform preprocessing and device transfers. Backbone timings were shared between identical frozen encoders; the three reported times are not independent encoder benchmarks.

- Seed 42: feature stage 1798.6 s; head optimization plus cached validation 3.19 s.
- Seed 43: feature stage 1791.1 s; head optimization plus cached validation 3.29 s.
- Seed 44: feature stage 1817.9 s; head optimization plus cached validation 8.34 s.

## Studies and fair-comparison limits

Prior EfficientNet studies tested learning rate, weight decay, augmentation and BatchNorm behavior. They are exploratory evidence about EfficientNet and must not be described as AST ablations. Its 22.05 kHz representation and noise realizations differ from AST's 16 kHz pipeline; training and tuning budgets also differ. Consequently, prior cross-model differences cannot isolate architecture as the sole cause.

The supplied run_cached_studies.py has been executed for all three seeds. It provides supplementary validation-only matched AST-head learning-rate and augmentation-removal studies, detailed below. These analyses were performed after final test evaluation and did not change the final model. They do not constitute a pretrained-versus-scratch AST comparison.

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


## Limitations and remaining work

- No AST-specific scratch-training comparison has been performed. Cover the assignment's team-level requirement using an explicitly documented applicable study, or design a separately budgeted experiment without using the final test to tune it.
- AST-head augmentation ablation and learning-rate studies are complete; they are supplementary validation-only analyses.
- Concrete error causes need listening-based review; no audio was available in this review environment.
- The shared model comparison, classic and advanced models, full technical report, slides and peer evaluation belong to team integration. Demo is outside this contribution.
- Pretraining overlap with the downstream dataset was not independently checked.
- Raw audio and full encoder weights are not bundled. Preserve dataset license and clip attribution; cache/download the pinned backbone for reproduction [2,3].

## References

[1] Gong, Y., Chung, Y.-A., and Glass, J. AST: Audio Spectrogram Transformer. Interspeech, 2021. https://arxiv.org/abs/2104.01778

[2] MIT AudioSet AST checkpoint. https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 . Transformers AST implementation and feature extraction: https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer

[3] Piczak, K. J. ESC: Dataset for Environmental Sound Classification. ACM Multimedia, 2015. https://doi.org/10.1145/2733373.2806390 . Dataset and license: https://github.com/karolpiczak/ESC-50
