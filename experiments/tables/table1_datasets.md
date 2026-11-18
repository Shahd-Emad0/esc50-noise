**Table 1 — Dataset comparison**

| Dataset | Source | Size | Format | Classes | Licence | Role |
|---|---|---|---|---|---|---|
| ESC-50 (Piczak 2015) | github.com/karolpiczak/ESC-50 | 2,000 clips (1,280 / 320 / 400) | 5 s WAV, 44.1 kHz mono | 50 (40 clips each) | CC BY-NC (ESC-10: CC BY) | train / val / test (group split on src_file) |
| Synthetic white noise (ours) | src/datasets/noise.py | generated on the fly | Gaussian, SNR 10/5/0/-5 dB | – | n/a | training augmentation + test corruption |
| Synthetic pink noise (ours) | src/datasets/noise.py | 400 test clips | 1/f noise, SNR 0 dB | – | n/a | test only (unseen noise type) |
| AudioSet (Gemmeke et al. 2017) | research.google.com/audioset | ~2M clips | 10 s YouTube audio | 527 | CC BY 4.0 (labels) | external: AST pretraining only |
| ImageNet-1k (Deng et al. 2009) | image-net.org | 1.28M images | RGB images | 1,000 | research licence | external: EfficientNet pretraining only |
