**Table 1 — Dataset comparison**

| Dataset | Source | Size | Format | Classes | Licence | Role |
|---|---|---|---|---|---|---|
| ESC-50 (Piczak 2015) | github.com/karolpiczak/ESC-50 | 2,000 clips (1,280 / 320 / 400) | 5 s WAV, 44.1 kHz mono | 50 (40 clips each) | CC BY-NC (ESC-10: CC BY) | train / val / test (group split on src_file) |
| Synthetic white noise (ours) | src/datasets/noise.py | generated on the fly | Gaussian, SNR 10/5/0/-5 dB | – | n/a | training augmentation + test corruption |
| Synthetic pink noise (ours) | src/datasets/noise.py | 400 test clips | 1/f noise, SNR 0 dB | – | n/a | test only (unseen noise type) |
| AudioSet (Gemmeke et al. 2017) | research.google.com/audioset | ~2M clips | 10 s YouTube audio | 527 | CC BY 4.0 (labels) | external: AST pretraining only |
| ImageNet-1k (Deng et al. 2009) | image-net.org | 1.28M images | RGB images | 1,000 | research licence | external: EfficientNet pretraining only |


**Table 2 — Model architecture comparison**

| Model | Family | Input | Main layers | Parameters | Pretrained |
|---|---|---|---|---|---|
| AST (AudioSet, frozen) + trained head | modern | log-mel 128×1024 frames (16 kHz) | 16x16 patch embedding -> 12 Transformer layers (768-d, frozen) -> LayerNorm -> FC | 86,227,250 | Yes (AudioSet) |
| Classic CNN (scratch) | classic | log-mel 128×216 (22.05 kHz) | 4 x [Conv3x3-BN-ReLU x2, MaxPool] (32-256 ch) -> GAP -> Dropout -> FC | 1,185,490 | No |
| SE-CRNN + attention pooling | advanced | log-mel 128×216 (22.05 kHz) | 4 residual SE conv blocks -> FC 256 -> BiGRU (2x128) -> 4-head attention pooling -> FC | 1,323,842 | No |
| EfficientNet-B0 (ImageNet pretrained) | modern | log-mel 128×216 (22.05 kHz) | EfficientNet-B0 MBConv+SE stages -> GAP -> FC | 4,071,022 | Yes (ImageNet) |
| EfficientNet-B0 (scratch) | modern | log-mel 128×216 (22.05 kHz) | EfficientNet-B0 MBConv+SE stages -> GAP -> FC | 4,071,022 | No |
| SVM baseline (MFCC, clean training) | baseline | MFCC 80-d (22.05 kHz) | 40 MFCC mean+std (80-d) -> StandardScaler -> RBF SVM (C=10) | 163,701 | No |
| SVM baseline (MFCC, noise-augmented) | baseline | MFCC 80-d (22.05 kHz) | 40 MFCC mean+std (80-d) -> StandardScaler -> RBF SVM (C=10) | 455,499 | No |


**Table 3 — Hyper-parameter configuration**

| Model | Optimizer | LR | Batch | Epochs | Regularisation | Loss | Augmentation |
|---|---|---|---|---|---|---|---|
| AST (AudioSet, frozen) + trained head | AdamW | 0.001 | 64 | 30 | wd 0.001, frozen encoder | cross-entropy | 1 white-noise copy / clip (SNR∈{10,5,0,−5}) |
| Classic CNN (scratch) | AdamW + OneCycle | 0.001 | 32 | 60 | wd 0.01, dropout 0.3 | cross-entropy | white noise p=0.5 SNR∈{10,5,0,−5}, SpecAugment |
| SE-CRNN + attention pooling | AdamW + OneCycle | 0.001 | 32 | 60 | wd 0.01, dropout 0.3 | cross-entropy | white noise p=0.5 SNR∈{10,5,0,−5}, SpecAugment |
| SE-CRNN, attention -> mean pooling | AdamW + OneCycle | 0.001 | 32 | 60 | wd 0.01, dropout 0.3 | cross-entropy | white noise p=0.5 SNR∈{10,5,0,−5}, SpecAugment |
| SE-CRNN, no noise augmentation | AdamW + OneCycle | 0.001 | 32 | 60 | wd 0.01, dropout 0.3 | cross-entropy | SpecAugment |
| SE-CRNN, no SpecAugment | AdamW + OneCycle | 0.001 | 32 | 60 | wd 0.01, dropout 0.3 | cross-entropy | white noise p=0.5 SNR∈{10,5,0,−5} |
| EfficientNet-B0 (ImageNet pretrained) | AdamW + OneCycle | 0.001 | 32 | 60 | wd 0.01, dropout 0.3 | cross-entropy | white noise p=0.5 SNR∈{10,5,0,−5}, SpecAugment |
| EfficientNet-B0 (scratch) | AdamW + OneCycle | 0.001 | 32 | 60 | wd 0.01, dropout 0.3 | cross-entropy | white noise p=0.5 SNR∈{10,5,0,−5}, SpecAugment |
| SVM baseline (MFCC, clean training) | libsvm (SMO) | – | – | – | C = 10, RBF γ = scale | hinge (multi-class OvO) | none |
| SVM baseline (MFCC, noise-augmented) | libsvm (SMO) | – | – | – | C = 10, RBF γ = scale | hinge (multi-class OvO) | 2 noisy copies / clip |


**Table 4 — Test macro-F1 (%) per condition, mean ± SD over 3 seeds (primary metric: macro-F1; secondary: accuracy)**

| Model | F1 clean | F1 10dB | F1 5dB | F1 0dB | F1 -5dB | F1 pink_0dB | Mean F1 clean+white [95% CI] | Clean accuracy |
|---|---|---|---|---|---|---|---|---|
| AST (AudioSet, frozen) + trained head | 96.0 ± 0.2 | 93.0 ± 0.3 | 91.1 ± 0.6 | 87.6 ± 0.6 | 79.4 ± 1.4 | 87.8 ± 0.1 | 89.4 ± 0.5 [88.2, 90.6] | 96.1 ± 0.1 |
| Classic CNN (scratch) | 81.9 ± 0.8 | 83.2 ± 1.4 | 81.7 ± 1.5 | 78.7 ± 0.9 | 72.5 ± 2.4 | 53.4 ± 4.7 | 79.6 ± 1.1 [76.9, 82.3] | 82.3 ± 0.8 |
| SE-CRNN + attention pooling | 79.3 ± 2.9 | 78.1 ± 0.8 | 75.6 ± 1.2 | 74.1 ± 0.3 | 69.3 ± 1.0 | 57.0 ± 0.4 | 75.3 ± 0.6 [73.7, 76.9] | 79.8 ± 2.8 |
| EfficientNet-B0 (ImageNet pretrained) | 87.4 ± 0.6 | 86.8 ± 1.0 | 85.3 ± 1.3 | 82.7 ± 1.3 | 77.9 ± 2.4 | 69.4 ± 1.9 | 84.1 ± 1.0 [81.5, 86.6] | 87.8 ± 0.7 |
| EfficientNet-B0 (scratch) | 59.7 ± 2.5 | 58.9 ± 1.2 | 58.2 ± 1.2 | 55.3 ± 1.5 | 50.5 ± 1.1 | 36.6 ± 2.4 | 56.5 ± 1.1 [53.7, 59.3] | 60.9 ± 2.0 |
| SVM baseline (MFCC, clean training) | 52.0 ± 0.0 | 13.6 ± 0.0 | 6.3 ± 0.0 | 1.5 ± 0.0 | 0.1 ± 0.0 | 1.9 ± 0.0 | 14.7 ± 0.0 [14.7, 14.7] | 53.8 ± 0.0 |
| SVM baseline (MFCC, noise-augmented) | 41.5 ± 0.5 | 45.2 ± 1.8 | 45.5 ± 1.3 | 43.3 ± 1.4 | 36.5 ± 2.1 | 21.4 ± 0.3 | 42.4 ± 1.2 [39.5, 45.3] | 42.5 ± 0.9 |


**Table 5 — Ablation study: test macro-F1 (%) without the component, (Δ vs full model, paired by seed, mean ± SD)**

| Component removed | Full model | Clean F1 (Δ) | −5 dB F1 (Δ) | Pink 0 dB F1 (Δ) | Mean F1 (Δ) |
|---|---|---|---|---|---|
| Attention pooling (→ mean pooling) | SE-CRNN + attention pooling | 76.7 (-2.7 ± 2.1) | 68.1 (-1.2 ± 1.4) | 53.2 (-3.8 ± 2.0) | 74.3 (-1.0 ± 0.6) |
| Noise augmentation | SE-CRNN + attention pooling | 73.7 (-5.6 ± 8.5) | 2.8 (-66.5 ± 2.3) | 26.7 (-30.3 ± 1.3) | 29.6 (-45.6 ± 1.1) |
| SpecAugment | SE-CRNN + attention pooling | 75.5 (-3.8 ± 2.3) | 65.4 (-3.9 ± 2.0) | 56.8 (-0.2 ± 2.9) | 72.1 (-3.2 ± 1.0) |
| ImageNet pretraining | EfficientNet-B0 (ImageNet pretrained) | 59.7 (-27.8 ± 2.5) | 50.5 (-27.4 ± 3.1) | 36.6 (-32.9 ± 2.7) | 56.5 (-27.5 ± 0.2) |
| Noise augmentation (SVM baseline) | SVM baseline (MFCC, noise-augmented) | 52.0 (+10.5 ± 0.5) | 0.1 (-36.4 ± 2.1) | 1.9 (-19.5 ± 0.3) | 14.7 (-27.7 ± 1.2) |
| Noise augmentation (AST head, validation, Roqaya) | AST + head | – | – | – | 73.5 (-14.1) |


**Table 6 — Error analysis: AST (AudioSet, frozen) + trained head, seed 43, test set**

| Error type | Count (of 400) | Example | Likely cause |
|---|---|---|---|
| Intrinsic confusion (wrong even on clean audio) | 15 | 1-50454-A-44.wav: engine → helicopter | acoustically similar classes / few (40) examples per class |
| Noise masking (correct clean → wrong at −5 dB) | 73 | 1-202111-A-3.wav: cow → snoring | low-energy or transient events buried below the noise floor |
| Broadband attractor (−5 dB error predicted as a noise-like class) | 46 | 1-208757-A-2.wav: pig → train | white noise resembles wind/rain/engine textures, so the model labels the noise, not the event |
| Unseen noise colour (correct at white 0 dB → wrong at pink 0 dB) | 18 | 1-208757-A-2.wav: pig → train | pink noise concentrates energy at low frequencies, masking bands not covered by training noise |
| Most frequent −5 dB confusion pair | 3 | door_wood_creaks → engine | overlapping spectral envelope once fine structure is masked |


**Table 7 — Computational cost (inference = raw 5 s waveform → logits, batch 1, warmed)**

| Model | Parameters | Size (MB) | Train time (min/seed) | Inference (ms/clip) | Peak GPU mem (MB) | Hardware |
|---|---|---|---|---|---|---|
| AST (AudioSet, frozen) + trained head | 86,227,250 | 328.9 | 30.1 | 101.6 | – | Quadro P1000 (inference: Tesla T4) |
| Classic CNN (scratch) | 1,185,490 | 5.1 | 5.9 | 2.2 | 1076 | Tesla T4 |
| SE-CRNN + attention pooling | 1,323,842 | 5.6 | 7.4 | 4.6 | 1513 | Tesla T4 |
| EfficientNet-B0 (ImageNet pretrained) | 4,071,022 | 16.3 | 5.5 | 11.5 | 1601 | Tesla T4 |
| EfficientNet-B0 (scratch) | 4,071,022 | 16.3 | 5.5 | 9.4 | 1601 | Tesla T4 |
| SVM baseline (MFCC, clean training) | 163,701 | 1.8 | 0.4 | 59.1 | – | CPU |
| SVM baseline (MFCC, noise-augmented) | 455,499 | 4.8 | 0.9 | 57.0 | – | CPU |


**Hyper-parameter study (SE-CRNN, validation set only, 3 seeds)**

| Learning rate | Val selection F1 | Val clean F1 | Val −5 dB F1 | Best epoch |
|---|---|---|---|---|
| 0.0003 | 66.8 ± 1.4 | 70.1 | 61.5 | 54 |
| 0.001 | 66.6 ± 1.9 | 71.1 | 60.9 | 45 |
| 0.003 | 66.7 ± 2.2 | 70.4 | 62.3 | 54 |


**Data-strategy study (augmentation), mean of 3 seeds**

| Augmentation strategy | Val selection F1 | Test clean F1 | Test −5 dB F1 | Test pink 0 dB F1 |
|---|---|---|---|---|
| noise + SpecAugment | 66.6 | 79.3 | 69.3 | 57.0 |
| noise only | 67.0 | 75.5 | 65.4 | 56.8 |
| SpecAugment only | 27.9 | 73.7 | 2.8 | 26.7 |
| SVM: 2 noisy copies | 38.6 | 41.5 | 36.5 | 21.4 |
| SVM: none | 13.8 | 52.0 | 0.1 | 1.9 |


**Pretrained vs scratch (same architecture, same budget)**

| Model | Val selection F1 | Clean | −5 dB | Pink 0 dB | Mean |
|---|---|---|---|---|---|
| EfficientNet-B0 (ImageNet pretrained) | 78.9 | 87.4 ± 0.6 | 77.9 ± 2.4 | 69.4 ± 1.9 | 84.1 ± 1.0 |
| EfficientNet-B0 (scratch) | 50.9 | 59.7 ± 2.5 | 50.5 ± 1.1 | 36.6 ± 2.4 | 56.5 ± 1.1 |


**Hypothesis verdicts**

| Hypothesis | Statement | Evidence (test, paired by seed) | Verdict |
|---|---|---|---|
| H1 | Pretraining improves noisy-sound classification on small data (EfficientNet-B0 ImageNet vs scratch, mean F1) | Δ = 27.5 ± 0.2 pp, 95% CI [27.1, 28.0] | Supported |
| H2 | Noise augmentation raises −5 dB macro-F1 while costing < 2 pp on clean audio (SE-CRNN) | Δ = 66.5 ± 2.3 pp, 95% CI [60.9, 72.2]; clean Δ = 5.6 pp | Supported |
| H3 | Attention pooling improves robustness at −5 dB vs mean pooling (SE-CRNN) | Δ = 1.2 ± 1.4 pp, 95% CI [-2.1, 4.6] | Not supported (95% CI includes 0) |
| H4 | Models trained with white noise lose accuracy on unseen pink noise at the same SNR (AST (AudioSet, frozen) + trained head, white 0 dB − pink 0 dB) | Δ = -0.3 ± 0.7 pp, 95% CI [-2.1, 1.6]; pink < white in 5/7 models | Not supported (95% CI includes 0) |
