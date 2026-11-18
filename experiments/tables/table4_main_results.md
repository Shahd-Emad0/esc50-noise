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
