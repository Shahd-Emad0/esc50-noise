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
