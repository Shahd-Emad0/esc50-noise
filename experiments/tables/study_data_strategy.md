**Data-strategy study (augmentation), mean of 3 seeds**

| Augmentation strategy | Val selection F1 | Test clean F1 | Test −5 dB F1 | Test pink 0 dB F1 |
|---|---|---|---|---|
| noise + SpecAugment | 66.6 | 79.3 | 69.3 | 57.0 |
| noise only | 67.0 | 75.5 | 65.4 | 56.8 |
| SpecAugment only | 27.9 | 73.7 | 2.8 | 26.7 |
| SVM: 2 noisy copies | 38.6 | 41.5 | 36.5 | 21.4 |
| SVM: none | 13.8 | 52.0 | 0.1 | 1.9 |
