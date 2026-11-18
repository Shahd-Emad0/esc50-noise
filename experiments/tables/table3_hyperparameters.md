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
