# Environmental Sound Classification under Noise

**Deep Learning (AI403) — Final Project, Fall 2026 — Pharos University in Alexandria**

**Team D07** — Supervisor: Dr. Sahar Ghanem — TA: Eng. Mohamed Ashraf

| Name | Student ID | Main responsibility |
|---|---|---|
| Rania Hamdy Gomaa | 202302910 | Data audit, group split, MFCC + SVM baseline, noise protocol |
| Heba Atef Farouk | 202301959 | Classic CNN, EfficientNet-B0 pretrained vs scratch |
| Roqaya Ebrahim Elghandour | 202302796 | AST modern model, AST learning-rate study and augmentation ablation |
| Nour Osama Hassan | 202301336 | SE-CRNN attention model, ablation study, error analysis |
| Shahd Emad | 202301173 | Shared evaluation, cost report, live demo, report and slides |


Repository: https://github.com/Shahd-Emad0/esc50-noise

# 1. Abstract

We study environmental sound classification (Project Idea 17) on ESC-50, a public benchmark of 2,000 five-second clips in 50 classes, under controlled additive noise. Using a leakage-free, source-grouped split (1,280 train / 320 validation / 400 test clips), we compare a classical MFCC + SVM baseline with four deep learning approaches covering the three required families: a classic convolutional network trained from scratch, two modern pretrained models (an ImageNet-pretrained EfficientNet-B0 and a frozen AudioSet Audio Spectrogram Transformer with a trained head), and an advanced hybrid architecture that we designed: a residual squeeze-and-excitation CNN followed by a bidirectional GRU and a multi-head attention pooling layer written from scratch. All models share the same split, the same waveform pipeline, the same noise-augmentation protocol and the same fixed test corruptions (clean, white noise at 10/5/0/−5 dB SNR, and unseen pink noise at 0 dB), and every final configuration is run with 3 random seeds. The best model selected on the validation set, AST (AudioSet, frozen) + trained head, reaches a test macro-F1 of 96.0 % on clean audio, 79.4 % at −5 dB and 87.8 % on unseen pink noise, compared with 41.5 % / 36.5 % for the noise-augmented SVM baseline. A hyper-parameter study, a data-strategy study, a pretrained-versus-scratch comparison, an ablation study, a categorised error analysis and a full computational-cost report are used to test four explicit hypotheses; two are supported (pretraining, noise augmentation) and two are not (attention pooling, and a pink-noise penalty for the best model). A Gradio demo classifies uploaded or recorded sounds live, with optional noise injection.

# 2. Introduction

Machines that understand everyday sounds are useful in hearing aids, smart homes, assistive technology for deaf and hard-of-hearing people, urban noise monitoring, wildlife monitoring and machine condition monitoring. Unlike speech or music, environmental sounds have no shared grammar: a dog bark, a chainsaw and rain differ in duration, harmonicity, stationarity and bandwidth. In deployment they are also rarely heard in isolation; microphones capture traffic, wind, crowds and electronic hiss together with the event of interest. A classifier that is accurate on studio-quality clips but collapses when the signal-to-noise ratio (SNR) drops is of little practical use.

Deep learning has largely replaced hand-crafted features for this task. Piczak [2] showed that a small CNN on log-mel spectrograms outperforms MFCC-based classifiers on ESC-50, Salamon and Bello [11] showed the value of data augmentation, and large-scale pretraining on AudioSet [4] produced models such as PANNs [10] and the Audio Spectrogram Transformer (AST) [3] that transfer well to small datasets. Most published ESC-50 results, however, are reported on clean audio only. Our project asks how these families behave when the input is corrupted, which design decisions make a model robust, and what remains hard.

Our contributions are: (i) a unified, leakage-free and reproducible protocol in which every model sees exactly the same clips and exactly the same noise samples; (ii) a comparison of five approaches across three deep learning families plus a classical baseline, each with 3 seeds; (iii) a hybrid SE-CRNN with a team-written attention pooling layer and a team-written SNR-exact noise mixer; (iv) hypothesis-driven studies (pretraining, augmentation, attention, unseen noise type) with paired statistics; and (v) a categorised error analysis and a live demonstration.

# 3. Problem Statement

**Task.** Single-label, 50-way classification of a 5-second environmental audio clip.

**Input.** A mono waveform resampled to 22.05 kHz (5 s = 110,250 samples), optionally corrupted by additive noise. Each model converts it to its own representation: MFCC statistics (baseline), 128-band log-mel spectrograms (CNN, EfficientNet, SE-CRNN), or the AST feature extractor's 16 kHz log-mel input.

**Output.** One of the 50 ESC-50 classes (plus a ranked top-5 list in the demo).

**Evaluation conditions.** Clean audio; white Gaussian noise at 10, 5, 0 and −5 dB SNR (the training-noise family); and pink (1/f) noise at 0 dB, a noise colour never used for training or model selection.

**Hypotheses** (each is tested in Section 14 with paired, per-seed differences and a 95 % confidence interval):

- **H1 — Pretraining.** With only 40 examples per class, ImageNet pretraining improves EfficientNet-B0's mean macro-F1 over the same architecture trained from scratch with the same budget.
- **H2 — Noise augmentation.** Training with white-noise augmentation raises macro-F1 at −5 dB substantially while costing less than 2 percentage points on clean audio.
- **H3 — Attention.** Learned attention pooling over time improves robustness at −5 dB compared with mean pooling, because it can weight frames where the event rises above the noise.
- **H4 — Unseen noise.** Robustness learned from white noise does not fully transfer: at the same SNR (0 dB), performance on pink noise is lower than on white noise.

# 4. Motivation

Three observations motivate the design. First, ESC-50 is small (40 clips per class), so the question of whether knowledge transferred from large datasets (ImageNet images or AudioSet audio) beats training from scratch is central and practically important for anyone building a sound classifier with limited labelled data. Second, clean-audio accuracy is a poor proxy for real-world usefulness; reporting performance as a function of SNR exposes failure modes that a single accuracy number hides. Third, the course asks us to reason like researchers: a fair comparison requires that differences between models come from the models, not from different splits, different noise samples, different tuning effort or test-set leakage. We therefore invested heavily in a shared protocol, in a group-aware split, and in seed-level statistics, and we report negative or inconclusive outcomes as they are.

# 5. Related Work

**Datasets and early systems.** ESC-50 was introduced by Piczak [1] with human accuracy of about 81 % and baseline classifiers on MFCC and zero-crossing features. MFCCs [19] combined with SVMs [18] or random forests remain a standard classical reference.

**Convolutional models.** Piczak [2] applied a two-layer CNN to log-mel spectrograms; deeper VGG-style networks [16] with batch normalisation [22] became the dominant design, and Hershey et al. [23] compared image-classification CNNs on large-scale audio. Salamon and Bello [11] showed that time stretching, pitch shifting and background-noise mixing materially improve environmental sound classification. SpecAugment [9] masks frequency bands and time segments of the spectrogram and is now ubiquitous.

**Recurrent and hybrid models.** Convolutional recurrent networks (CRNNs) combine local spectro-temporal feature extraction with recurrent modelling of temporal context; Cakir et al. [8] demonstrated their advantage for sound event detection. Squeeze-and-excitation (SE) blocks [7] re-weight channels using global context, and attention-based pooling over time, used for weakly labelled audio tagging in PANNs [10], lets a clip-level classifier focus on informative frames. GRUs [12] are a lighter alternative to LSTMs.

**Transfer learning and Transformers.** PANNs [10] pretrained CNNs on AudioSet [4] and showed strong transfer to ESC-50. ImageNet-pretrained image models [5, 6] also transfer to spectrograms despite the domain gap. The AST [3] applies a Vision-Transformer-style encoder to spectrogram patches and, after AudioSet pretraining, reports state-of-the-art ESC-50 accuracy. We use public implementations from timm [20] and Hugging Face Transformers [21].

**Robustness and evaluation methodology.** Data leakage between training and evaluation inflates reported results [17]; fragments of the same source recording in both partitions are a known risk on ESC-50. We therefore split by source file and report seed variability rather than single runs.

# 6. Dataset

**Table 1 — Dataset comparison**

| Dataset | Source | Size | Format | Classes | Licence | Role |
|---|---|---|---|---|---|---|
| ESC-50 (Piczak 2015) | github.com/karolpiczak/ESC-50 | 2,000 clips (1,280 / 320 / 400) | 5 s WAV, 44.1 kHz mono | 50 (40 clips each) | CC BY-NC (ESC-10: CC BY) | train / val / test (group split on src_file) |
| Synthetic white noise (ours) | src/datasets/noise.py | generated on the fly | Gaussian, SNR 10/5/0/-5 dB | – | n/a | training augmentation + test corruption |
| Synthetic pink noise (ours) | src/datasets/noise.py | 400 test clips | 1/f noise, SNR 0 dB | – | n/a | test only (unseen noise type) |
| AudioSet (Gemmeke et al. 2017) | research.google.com/audioset | ~2M clips | 10 s YouTube audio | 527 | CC BY 4.0 (labels) | external: AST pretraining only |
| ImageNet-1k (Deng et al. 2009) | image-net.org | 1.28M images | RGB images | 1,000 | research licence | external: EfficientNet pretraining only |


ESC-50 [1] contains 2,000 five-second clips extracted from Freesound recordings, 40 per class, grouped into five broad categories (animals; natural soundscapes and water; human non-speech; domestic/interior; urban/exterior). Clips are mono WAV at 44.1 kHz. The metadata provides the class index, category name, official fold and the identifier of the original Freesound recording (`src_file`). The main dataset is licensed CC BY-NC; the ESC-10 subset is CC BY. Clip-level attributions are in the dataset's LICENSE file, which our download script keeps.

**Split strategy.** The official five folds are designed for cross-validation; our project requires a single train/validation/test split with the test set used only once. Because several clips can be cut from the same original recording, a random split could place near-duplicate fragments in training and test. Rania Hamdy Gomaa's data audit found source identifiers occurring in more than one official fold. We therefore used `StratifiedGroupKFold` twice (outer split for test, inner split for validation; seeds 42 and 43) with `src_file` as the group, producing 1,280 training (64 %), 320 validation (16 %) and 400 test (20 %) clips. The pipeline asserts at every run that no source file appears in two partitions, that all 50 classes appear in every partition, and that the manifest's SHA-256 matches the one recorded with the AST checkpoints, so all members provably used the same split.

**Noise data.** We generate noise synthetically rather than using a second corpus. White noise is used for training augmentation and for the main test conditions; pink noise is held out entirely to measure generalisation to an unseen noise colour. Synthetic noise is fully reproducible and its SNR is exact, but it does not capture all properties of real environmental noise (see Section 18).

# 7. Data Preprocessing

All models start from the same cached waveforms: each WAV is read with `soundfile`, averaged to mono, resampled from 44.1 kHz to 22.05 kHz with polyphase filtering, cropped or zero-padded to exactly 5 s, and stored once as a float16 array (about 440 MB) so that every experiment reads identical samples.

**Noise mixing (team-written component).** For a clean signal x with power P_s and a noise vector n with power P_n, the mixer returns x + g·n with g = sqrt(P_s / (P_n · 10^(SNR/10))), which sets the SNR exactly. Pink noise is produced by dividing the amplitude spectrum of Gaussian noise by sqrt(f). The function is vectorised so that each clip in a GPU batch can receive a different SNR, and a self-test verifies the measured SNR to within 0.001 dB before every run. Test and validation corruptions are deterministic: the noise for each (clip, condition) pair is generated from a seed derived from a SHA-256 hash of the file name and condition, so every model and every seed is evaluated on byte-identical noisy inputs. Pink noise is never used in training or model selection.

**Representations.** The baseline computes 40 MFCCs (n_fft 2048, hop 512) and summarises each coefficient by its mean and standard deviation (80 features), standardised with statistics from the training set only. Spectrogram models compute a 128-band power mel spectrogram (n_fft 2048, hop 512 → 128 × 216), convert it to decibels and standardise each clip to zero mean and unit variance; this runs inside the model on the GPU so that noise augmentation can be applied to the waveform first. AST uses its own feature extractor (16 kHz, 128 mel bins, 1,024 frames, AudioSet normalisation statistics) after resampling the same noisy 22.05 kHz waveform.

**Augmentation.** During training, each clip is corrupted with probability 0.5 by white noise at an SNR drawn uniformly from {10, 5, 0, −5} dB; SpecAugment applies one frequency mask (≤ 24 mel bins) and one time mask (≤ 48 frames) per clip. The SVM baseline cannot use on-the-fly augmentation, so it receives two noisy copies of each training clip (3,840 examples). The AST head was trained on one clean and one noisy embedding per clip because its encoder is frozen and embeddings are cached.

# 8. Methodology

The study follows a fixed protocol designed for a fair comparison:

- **Same data.** One split manifest, one waveform cache, one label mapping, identical validation and test corruptions for all models.
- **Same budget.** All trainable spectrogram models use AdamW, a one-cycle learning-rate schedule with peak 1e-3, weight decay 0.01, batch size 32, 60 epochs, cross-entropy loss and the same augmentation. No model received individual tuning; the learning-rate study (Section 12) was run only to check the sensitivity of the shared value.
- **Same selection rule.** After the first quarter of training, the checkpoint with the highest validation selection score — mean macro-F1 over clean and the four white-noise SNRs on the validation split — is kept. The AST head used exactly the same rule. The test set is read once, by `src/evaluate.py`, after all training has finished, and no decision is taken from test results; the "best model" and the demo checkpoints are also chosen on validation.
- **Seeds.** Every final configuration is trained with seeds 42, 43 and 44 (affecting initialisation, batch order, augmentation and dropout). We report mean ± sample standard deviation and, for comparisons, paired per-seed differences with a two-sided 95 % t-interval (n = 3, t = 4.30). With three seeds these intervals are wide by design; we label an effect "supported" only if its interval excludes zero.
- **Metrics.** Accuracy is misleading only mildly here because classes are balanced, but per-class behaviour matters, so the primary metric is macro-F1. Secondary metrics are accuracy, macro-precision, macro-recall and top-5 accuracy; per-class F1 and confusion matrices are saved for every run and condition.

# 9. Baseline

The baseline reproduces Rania Hamdy Gomaa's classical pipeline: 80 MFCC statistics, a `StandardScaler` fitted on training data and an RBF-kernel SVM with C = 10 and γ = "scale" [18, 19]. We report two variants: trained on clean clips only, and trained on clean clips plus two white-noise copies per clip. The clean-only SVM is deterministic (its three "seeds" are identical); the noise-augmented SVM varies the noise realisation with the seed. The baseline is fast (seconds to train, milliseconds to predict) and sets the reference that every deep model must beat to justify its cost.

# 10. Proposed / Advanced Models

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


**Classic model — VGG-style CNN (scratch).** Four blocks of two 3×3 convolutions with batch normalisation and ReLU followed by 2×2 max-pooling (32, 64, 128, 256 channels), global average pooling, dropout 0.3 and a linear classifier (≈1.2 M parameters). This is the standard design popularised by Piczak [2] and VGG [16], and it isolates what a purely convolutional model learns from 1,280 clips.

**Modern model 1 — EfficientNet-B0.** A compound-scaled network of mobile inverted-bottleneck blocks with SE attention [5], loaded from timm [20] with a one-channel stem. We train it twice under identical conditions: initialised from ImageNet [6] weights and from random weights. This is our pretrained-versus-scratch comparison: architecture, data and budget are held constant, so any difference is due to pretraining.

**Modern model 2 — Audio Spectrogram Transformer (Roqaya Ebrahim Elghandour).** The AST [3] splits a 128 × 1,024 log-mel input into overlapping 16 × 16 patches and processes them with a 12-layer, 768-dimensional Transformer pretrained on AudioSet. We keep the encoder frozen (86.2 M parameters) and train only the LayerNorm + linear head (39,986 parameters) for 30 epochs; the head checkpoints are reused unchanged from Roqaya Ebrahim Elghandour's work and re-evaluated on our shared test corruptions. Freezing makes training cheap and avoids over-fitting a very large model to 1,280 clips, at the cost of not adapting the representation.

**Advanced model — SE-CRNN with attention pooling (ours).** The front end has four residual blocks, each with two 3×3 convolutions, batch normalisation, a squeeze-and-excitation module [7] and a 1×1 projection shortcut; pooling halves frequency in every block but time only in the first two, giving a 128-channel × 8-band × 54-frame map. Each frame is projected to 256 dimensions and processed by a bidirectional GRU [8, 12]. Our attention pooling layer then computes, for each of four heads, softmax weights over the 54 frames and a weighted sum of value vectors, which are concatenated and projected; a dropout + linear layer produces the 50 logits (≈1.3 M parameters, similar to the classic CNN so that architecture rather than size is compared). The motivation is that in noisy clips the informative event often occupies only a few frames; attention can up-weight them, whereas mean pooling dilutes them with noise-only frames. The attention layer and the noise mixer are the components we implemented ourselves rather than calling from a library.

# 11. Experimental Setup

Each experiment is defined by one YAML file in `configs/` that overrides only the studied factor of `configs/_defaults.yaml`. `run_all.py` executes the full pipeline: dataset download and audit, waveform cache, noise self-test, training of every configuration and seed (resumable), a single test evaluation, analysis and report generation. Training and inference were run on: CPU, Quadro P1000, Tesla T4. Software: Python 3.10–3.12, PyTorch 2.5.1, torchaudio 2.5.1, timm 1.0.11, Transformers 4.46.3, scikit-learn and librosa (exact versions in `requirements.txt`).

Experiments run: two baselines; four deep models (classic CNN, EfficientNet-B0 pretrained, EfficientNet-B0 scratch, SE-CRNN) × 3 seeds; AST heads × 3 seeds; three ablations × 3 seeds; two additional learning rates × 3 seeds. Every run stores its validation metrics, training history, cost record, checkpoint, test predictions, per-class F1 and confusion data under `experiments/runs/<config>/<seed>/`.

# 12. Hyperparameters

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


**Hyper-parameter study.** We varied the peak learning rate of the SE-CRNN (3e-4, 1e-3, 3e-3) with three seeds each and measured validation performance only:

**Hyper-parameter study (SE-CRNN, validation set only, 3 seeds)**

| Learning rate | Val selection F1 | Val clean F1 | Val −5 dB F1 | Best epoch |
|---|---|---|---|---|
| 0.0003 | 66.8 ± 1.4 | 70.1 | 61.5 | 54 |
| 0.001 | 66.6 ± 1.9 | 71.1 | 60.9 | 45 |
| 0.003 | 66.7 ± 2.2 | 70.4 | 62.3 | 54 |


Roqaya Ebrahim Elghandour ran the equivalent study for the AST head (learning rates 3e-4, 1e-3 and 3e-3): the reference 1e-3 had the highest mean validation macro-F1 (87.65 %), with differences under 0.5 points between settings, so the shared value 1e-3 was retained for all models. The SE-CRNN study above leads to the same conclusion: the three learning rates differ by at most 0.2 points in mean validation selection F1 (66.6–66.8 %), far less than the seed standard deviation (1.4–2.2 points), so the shared 1e-3 is not a disadvantage for this model.

# 13. Results

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


![Figure 1 — Test macro-F1 versus noise condition (mean ± SD over seeds). The dashed line separates the training-noise family from unseen pink noise.](C:/Users/shahd/Downloads/esc50-noise/experiments/figures/snr_curves.png)

**Ranking by mean macro-F1 (clean + four white-noise SNRs):**

1. AST (AudioSet, frozen) + trained head — 89.4 %
2. EfficientNet-B0 (ImageNet pretrained) — 84.1 %
3. Classic CNN (scratch) — 79.6 %
4. SE-CRNN + attention pooling — 75.3 %
5. EfficientNet-B0 (scratch) — 56.5 %
6. SVM baseline (MFCC, noise-augmented) — 42.4 %
7. SVM baseline (MFCC, clean training) — 14.7 %

**Data-strategy study.** The following table compares augmentation strategies for the SE-CRNN and the baseline (selection on validation, test reported for completeness):

**Data-strategy study (augmentation), mean of 3 seeds**

| Augmentation strategy | Val selection F1 | Test clean F1 | Test −5 dB F1 | Test pink 0 dB F1 |
|---|---|---|---|---|
| noise + SpecAugment | 66.6 | 79.3 | 69.3 | 57.0 |
| noise only | 67.0 | 75.5 | 65.4 | 56.8 |
| SpecAugment only | 27.9 | 73.7 | 2.8 | 26.7 |
| SVM: 2 noisy copies | 38.6 | 41.5 | 36.5 | 21.4 |
| SVM: none | 13.8 | 52.0 | 0.1 | 1.9 |


**Per-class results.** Per-class F1 for every run and condition is stored in `experiments/results/test_per_class.csv`. For the best model, the hardest classes at −5 dB are:

**Ten hardest classes at −5 dB — AST (AudioSet, frozen) + trained head (F1 %, mean of seeds)**

| class | clean | -5dB |
|---|---|---|
| engine | 90.8 | 29.8 |
| wind | 69.0 | 35.2 |
| rain | 95.6 | 44.9 |
| helicopter | 77.1 | 48.8 |
| vacuum_cleaner | 97.8 | 55.5 |
| keyboard_typing | 85.8 | 66.8 |
| footsteps | 93.3 | 66.8 |
| clock_alarm | 97.8 | 67.7 |
| door_wood_creaks | 100.0 | 68.3 |
| chainsaw | 92.4 | 70.0 |


![Figure 2 — Per-class F1 of the best model on clean audio and at −5 dB, sorted by −5 dB performance.](C:/Users/shahd/Downloads/esc50-noise/experiments/figures/per_class_f1.png)

![Figure 3 — Training loss and validation selection score per epoch (seed 42).](C:/Users/shahd/Downloads/esc50-noise/experiments/figures/training_curves.png)

# 14. Model Comparison

The comparison below uses only quantities produced under the identical protocol of Section 8. Key findings, generated directly from the result files:

- Best model on validation: **AST (AudioSet, frozen) + trained head** (deployed seed 43).
- AST (AudioSet, frozen) + trained head (best model): clean 96.0 % → −5 dB 79.4 % (drop 16.6 pp); unseen pink 0 dB 87.8 %.
- SVM baseline (MFCC, noise-augmented) (noise-augmented baseline): clean 41.5 % → −5 dB 36.5 % (drop 5.0 pp); unseen pink 0 dB 21.4 %.
- Classic CNN (scratch) (classic CNN): clean 81.9 % → −5 dB 72.5 % (drop 9.4 pp); unseen pink 0 dB 53.4 %.
- ImageNet pretraining changes EfficientNet-B0's mean macro-F1 by +27.5 pp.
- Noise augmentation changes the SE-CRNN's −5 dB macro-F1 by +66.5 pp.
- Attention pooling vs mean pooling at −5 dB: +1.2 pp.
- H1: **Supported** — Δ = 27.5 ± 0.2 pp, 95% CI [27.1, 28.0].
- H2: **Supported** — Δ = 66.5 ± 2.3 pp, 95% CI [60.9, 72.2]; clean Δ = 5.6 pp.
- H3: **Not supported (95% CI includes 0)** — Δ = 1.2 ± 1.4 pp, 95% CI [-2.1, 4.6].
- H4: **Not supported (95% CI includes 0)** — Δ = -0.3 ± 0.7 pp, 95% CI [-2.1, 1.6]; pink < white in 5/7 models.

**Pretrained versus scratch.** Holding EfficientNet-B0's architecture and budget fixed:

**Pretrained vs scratch (same architecture, same budget)**

| Model | Val selection F1 | Clean | −5 dB | Pink 0 dB | Mean |
|---|---|---|---|---|---|
| EfficientNet-B0 (ImageNet pretrained) | 78.9 | 87.4 ± 0.6 | 77.9 ± 2.4 | 69.4 ± 1.9 | 84.1 ± 1.0 |
| EfficientNet-B0 (scratch) | 50.9 | 59.7 ± 2.5 | 50.5 ± 1.1 | 36.6 ± 2.4 | 56.5 ± 1.1 |


**Hypothesis verdicts.**

**Hypothesis verdicts**

| Hypothesis | Statement | Evidence (test, paired by seed) | Verdict |
|---|---|---|---|
| H1 | Pretraining improves noisy-sound classification on small data (EfficientNet-B0 ImageNet vs scratch, mean F1) | Δ = 27.5 ± 0.2 pp, 95% CI [27.1, 28.0] | Supported |
| H2 | Noise augmentation raises −5 dB macro-F1 while costing < 2 pp on clean audio (SE-CRNN) | Δ = 66.5 ± 2.3 pp, 95% CI [60.9, 72.2]; clean Δ = 5.6 pp | Supported |
| H3 | Attention pooling improves robustness at −5 dB vs mean pooling (SE-CRNN) | Δ = 1.2 ± 1.4 pp, 95% CI [-2.1, 4.6] | Not supported (95% CI includes 0) |
| H4 | Models trained with white noise lose accuracy on unseen pink noise at the same SNR (AST (AudioSet, frozen) + trained head, white 0 dB − pink 0 dB) | Δ = -0.3 ± 0.7 pp, 95% CI [-2.1, 1.6]; pink < white in 5/7 models | Not supported (95% CI includes 0) |


**Why the models differ.** Pretrained models start from features learnt on orders of magnitude more data, which matters most when each class has only 32 training clips after the split. AST additionally benefits from in-domain (audio) pretraining on AudioSet, whose ontology overlaps heavily with ESC-50 classes, whereas EfficientNet transfers generic edge and texture detectors from images. Among from-scratch models, the extra machinery of our SE-CRNN did not pay off under this budget: the plain VGG-style CNN, with a similar parameter count (1.19 M vs 1.32 M), was better in every white-noise condition (mean 79.6 % vs 75.3 %, clean 81.9 % vs 79.3 %, −5 dB 72.5 % vs 69.3 %). A plausible explanation is that the recurrent and attention layers add optimisation difficulty and variance that 1,280 training clips cannot support, whereas the CNN's local time–frequency invariance is sufficient for 5-second clips. The SE-CRNN was better only on unseen pink noise (57.0 % vs 53.4 %), where the CNN also had the largest seed variance (± 4.7). The classical baseline summarises each clip by global MFCC statistics, so additive broadband noise shifts the whole feature vector, which explains its steep degradation unless it is trained on noisy copies. Training cost and accuracy are not proportional; Table 7 shows the trade-off.

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


AST training time is Roqaya Ebrahim Elghandour's measured embedding-extraction stage plus head training. Inference is measured end-to-end from the raw 5-second waveform with batch size 1 after warm-up.

# 15. Ablation Study

**Table 5 — Ablation study: test macro-F1 (%) without the component, (Δ vs full model, paired by seed, mean ± SD)**

| Component removed | Full model | Clean F1 (Δ) | −5 dB F1 (Δ) | Pink 0 dB F1 (Δ) | Mean F1 (Δ) |
|---|---|---|---|---|---|
| Attention pooling (→ mean pooling) | SE-CRNN + attention pooling | 76.7 (-2.7 ± 2.1) | 68.1 (-1.2 ± 1.4) | 53.2 (-3.8 ± 2.0) | 74.3 (-1.0 ± 0.6) |
| Noise augmentation | SE-CRNN + attention pooling | 73.7 (-5.6 ± 8.5) | 2.8 (-66.5 ± 2.3) | 26.7 (-30.3 ± 1.3) | 29.6 (-45.6 ± 1.1) |
| SpecAugment | SE-CRNN + attention pooling | 75.5 (-3.8 ± 2.3) | 65.4 (-3.9 ± 2.0) | 56.8 (-0.2 ± 2.9) | 72.1 (-3.2 ± 1.0) |
| ImageNet pretraining | EfficientNet-B0 (ImageNet pretrained) | 59.7 (-27.8 ± 2.5) | 50.5 (-27.4 ± 3.1) | 36.6 (-32.9 ± 2.7) | 56.5 (-27.5 ± 0.2) |
| Noise augmentation (SVM baseline) | SVM baseline (MFCC, noise-augmented) | 52.0 (+10.5 ± 0.5) | 0.1 (-36.4 ± 2.1) | 1.9 (-19.5 ± 0.3) | 14.7 (-27.7 ± 1.2) |
| Noise augmentation (AST head, validation, Roqaya) | AST + head | – | – | – | 73.5 (-14.1) |


Each row removes exactly one component from the full model and keeps everything else, including seeds, fixed: attention pooling (replaced by mean pooling over time), white-noise augmentation, SpecAugment, or ImageNet pretraining. Δ is the paired difference (ablated − full) per seed, averaged; a negative Δ means the component helps. The last row repeats Roqaya Ebrahim Elghandour's validation-only ablation of noise augmentation for the AST head (reference 87.65 % vs 73.54 % mean validation macro-F1), which shows the same direction of effect on a completely different architecture. Note that removing the attention layer also removes its two small projection layers (about 0.13 M parameters), so that row measures the pooling mechanism together with that capacity.

# 16. Error Analysis

**Table 6 — Error analysis: AST (AudioSet, frozen) + trained head, seed 43, test set**

| Error type | Count (of 400) | Example | Likely cause |
|---|---|---|---|
| Intrinsic confusion (wrong even on clean audio) | 15 | 1-50454-A-44.wav: engine → helicopter | acoustically similar classes / few (40) examples per class |
| Noise masking (correct clean → wrong at −5 dB) | 73 | 1-202111-A-3.wav: cow → snoring | low-energy or transient events buried below the noise floor |
| Broadband attractor (−5 dB error predicted as a noise-like class) | 46 | 1-208757-A-2.wav: pig → train | white noise resembles wind/rain/engine textures, so the model labels the noise, not the event |
| Unseen noise colour (correct at white 0 dB → wrong at pink 0 dB) | 18 | 1-208757-A-2.wav: pig → train | pink noise concentrates energy at low frequencies, masking bands not covered by training noise |
| Most frequent −5 dB confusion pair | 3 | door_wood_creaks → engine | overlapping spectral envelope once fine structure is masked |


![Figure 4 — Confusion matrix of the best model at −5 dB SNR.](C:/Users/shahd/Downloads/esc50-noise/experiments/figures/confusion_-5dB.png)

![Figure 5 — Concrete failures: log-mel spectrograms of test clips classified correctly when clean (left) and misclassified at −5 dB (right).](C:/Users/shahd/Downloads/esc50-noise/experiments/figures/error_examples.png)

The most frequent confusion pairs at −5 dB are listed in `experiments/tables/error_pairs_-5dB.csv`:

- door_wood_creaks → engine: 3
- crickets → wind: 3
- helicopter → engine: 3
- vacuum_cleaner → chainsaw: 2
- can_opening → crackling_fire: 2
- hen → crying_baby: 2

**Interpretation.** Errors fall into distinct mechanisms. *Intrinsic confusions* already occur on clean audio and involve classes with similar spectral envelopes or that are under-represented after the split. *Noise masking* dominates at low SNR (73 of 400 clips for the deployed AST seed, against 15 intrinsic errors): at −5 dB the event's energy lies below the noise in most time–frequency bins (visible in Figure 5 as the event fading into a uniform background). For the best model the hardest classes at −5 dB are not short transients but stationary broadband textures (engine 90.8 → 29.8 %, wind 69.0 → 35.2 %, rain 95.6 → 44.9 %, helicopter, vacuum cleaner), whose spectra are closest to white noise; transient classes such as keyboard typing and footsteps follow (≈ 67 %). A characteristic sub-pattern is the *broadband attractor*: the model assigns noisy clips to classes whose clean examples are themselves noise-like textures (wind, rain, engines, vacuum cleaner), i.e. it recognises the noise rather than the event. Finally, *unseen-noise* errors are rare for AST (18 clips; its pink 0 dB F1 of 87.8 % equals its white 0 dB F1 of 87.6 %) but large for the models trained only on ESC-50 spectrograms (e.g. classic CNN 78.7 % → 53.4 %, SE-CRNN 74.1 % → 57.0 %): for them robustness learnt from white noise is partly specific to that spectrum, since pink noise concentrates energy at low frequencies and changes the spectral tilt the model relied on. The likely causes listed in Table 6 are hypotheses supported by spectrogram inspection; listening to the listed clips (file names given) is the recommended verification step.

# 17. Discussion

The results connect directly to the hypotheses. Pretraining (H1) addresses the main limitation of ESC-50, its size, and in-domain audio pretraining (AST) is the most effective form we tested. ImageNet pretraining alone raised EfficientNet-B0's mean macro-F1 by 27.5 points (56.5 % → 84.1 %). Noise augmentation (H2) is the single most important data strategy for robustness: without it the SE-CRNN falls to 2.8 % at −5 dB (−66.5 points), and the noise-free SVM to 0.1 %; on clean audio it did not cost accuracy for the SE-CRNN (+5.6 points, but with a large seed SD of 8.5), whereas for the SVM it did (−10.5 points), so the < 2-point clean-cost condition holds for the deep model only. Attention pooling (H3) gave +1.2 points at −5 dB with a 95 % CI of [−2.1, 4.6], so we cannot claim an effect; the mean-pooling variant was lower in five of six test conditions (higher at 0 dB), but by amounts within seed variance. H4 is not supported for the best model: the AudioSet-pretrained AST loses nothing on pink noise (−0.3 points), presumably because its pretraining data contains varied real background noise. The hypothesis does describe the models trained only on ESC-50 (5 of 7 models score lower on pink than on white noise at 0 dB, by up to 25 points for the classic CNN), so robustness learnt from synthetic white noise transfers only partly to another noise colour unless large-scale pretraining is used.

There is also a practical trade-off. The frozen AST is the most accurate (mean 89.4 %) but has about 65× more parameters than the SE-CRNN (86.2 M vs 1.3 M) and needs 101.6 ms per clip on a Tesla T4, against 11.5 ms for the pretrained EfficientNet-B0 (mean 84.1 %) and 2.2 ms for the classic CNN (79.6 %); the EfficientNet or the CNN may be the better choice for an embedded device, while AST suits a server. The baseline remains valuable as a sanity check and is orders of magnitude cheaper.

# 18. Limitations

Noise is synthetic (white and pink) and stationary; real environments contain non-stationary, structured noise (speech, traffic, music) and reverberation, so absolute robustness numbers should not be read as field performance. Three seeds give only coarse confidence intervals, and the 400-clip test set gives about ±2.5 percentage points of binomial uncertainty on accuracy. Only one train/validation/test split was used (cross-validation over the five folds would be more precise but conflicts with a single held-out test set). The AST encoder was frozen; fine-tuning it might improve results but was beyond our compute budget, and AST operates at 16 kHz while the other models use 22.05 kHz (all models nevertheless receive the same noisy waveform). The AST heads were trained by Roqaya Ebrahim Elghandour before the full team pipeline existed, with the same split, seeds and selection rule but with cached embeddings and one noisy copy per clip, so their training budget is not identical to the spectrogram models. ESC-50 clips are curated and 5 s long; behaviour on long continuous recordings or unknown classes is untested.

# 19. Ethical Considerations

ESC-50 is licensed CC BY-NC; our work is non-commercial and we cite the dataset and keep its attribution file. Sound recognition can be used for surveillance (e.g. continuous monitoring of homes or public spaces); any deployment should obtain consent, process audio on-device where possible and avoid storing recordings. Failure modes matter: a missed crying baby, glass break or alarm in a safety application can cause harm, and our results show that errors grow with noise, so such systems need human oversight and should not be marketed as reliable in noisy conditions without field testing. Freesound recordings over-represent certain regions and recording devices, so performance may differ for sounds from under-represented environments. Training and evaluating large pretrained models has an energy cost; our frozen-encoder design and embedding caching reduce it. All predictions are produced locally by our own models; no external AI service is used.

# 20. Conclusion

We built and evaluated a complete, reproducible system for environmental sound classification under noise. Under a strictly shared protocol, AST (AudioSet, frozen) + trained head gave the best validation performance and reached 96.0 % test macro-F1 on clean audio and 79.4 % at −5 dB, compared with 41.5 % and 36.5 % for the noise-augmented MFCC + SVM baseline. Pretraining, noise augmentation and architectural choices were tested as explicit hypotheses with paired, multi-seed statistics, and the error analysis identified noise masking and noise-like attractor classes as the dominant failure mechanisms at low SNR. Two of our expectations were not confirmed: the attention-based SE-CRNN did not outperform a plain CNN, and the AudioSet-pretrained AST showed no penalty on unseen pink noise. The whole pipeline, from download to report, runs with one command, and the live demo lets users probe the model's robustness interactively.

# 21. Future Work

Evaluate on real recorded noise (e.g. mixing with public background-noise corpora) and with reverberation; fine-tune the AST encoder or adopt efficient audio transformers; add mixup and background-mixing augmentation from real environments; use curriculum learning over SNR; calibrate the output probabilities and add an "unknown sound" rejection option; distil the best pretrained model into the small SE-CRNN for on-device use; and replace the single split with repeated group-aware splits to tighten confidence intervals.

# 22. References

[1] K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," in Proc. 23rd ACM International Conference on Multimedia, 2015, pp. 1015–1018. doi:10.1145/2733373.2806390

[2] K. J. Piczak, "Environmental sound classification with convolutional neural networks," in Proc. IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2015.

[3] Y. Gong, Y.-A. Chung, J. Glass, "AST: Audio Spectrogram Transformer," in Proc. Interspeech, 2021.

[4] J. F. Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," in Proc. IEEE ICASSP, 2017.

[5] M. Tan, Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. ICML, 2019.

[6] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in Proc. IEEE CVPR, 2009.

[7] J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks," in Proc. IEEE CVPR, 2018.

[8] E. Cakir, G. Parascandolo, T. Heittola, H. Huttunen, T. Virtanen, "Convolutional recurrent neural networks for polyphonic sound event detection," IEEE/ACM Trans. Audio, Speech, and Language Processing, 25(6), 2017.

[9] D. S. Park et al., "SpecAugment: A simple data augmentation method for automatic speech recognition," in Proc. Interspeech, 2019.

[10] Q. Kong et al., "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition," IEEE/ACM Trans. Audio, Speech, and Language Processing, 28, 2020.

[11] J. Salamon, J. P. Bello, "Deep convolutional neural networks and data augmentation for environmental sound classification," IEEE Signal Processing Letters, 24(3), 2017.

[12] K. Cho et al., "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in Proc. EMNLP, 2014.

[13] A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems 30 (NIPS), 2017.

[14] I. Loshchilov, F. Hutter, "Decoupled weight decay regularization," in Proc. ICLR, 2019.

[15] L. N. Smith, N. Topin, "Super-convergence: Very fast training of neural networks using large learning rates," Proc. SPIE 11006, 2019.

[16] K. Simonyan, A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in Proc. ICLR, 2015.

[17] S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," ACM Trans. Knowledge Discovery from Data, 6(4), 2012.

[18] C. Cortes, V. Vapnik, "Support-vector networks," Machine Learning, 20, 1995.

[19] S. Davis, P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," IEEE Trans. Acoustics, Speech, and Signal Processing, 28(4), 1980.

[20] R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019. https://github.com/huggingface/pytorch-image-models

[21] T. Wolf et al., "Transformers: State-of-the-art natural language processing," in Proc. EMNLP: System Demonstrations, 2020.

[22] S. Ioffe, C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in Proc. ICML, 2015.

[23] S. Hershey et al., "CNN architectures for large-scale audio classification," in Proc. IEEE ICASSP, 2017.
