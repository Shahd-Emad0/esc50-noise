# ESC-50 Member 1 — Data, Features, and Classical Baseline

## Project status

This package documents Member 1's data preparation and classical
machine-learning experiments for environmental sound classification
using ESC-50.

The group-aware stratified split in
`results/split_manifest_v2.csv` is a **proposed experimental split**.
It has not yet been approved as the shared split by the full team.
Do not describe it as the final team split until all model owners agree
to use the exact same manifest.

## Dataset

- Dataset: ESC-50 Environmental Sound Classification.
- Official source: https://github.com/karolpiczak/ESC-50
- Dataset paper: K. J. Piczak, "ESC: Dataset for Environmental Sound
  Classification," ACM Multimedia, 2015.
- DOI: https://doi.org/10.1145/2733373.2806390
- Expected size: 2,000 audio clips, 50 classes, 40 clips per class.
- Official evaluation structure: five folds of 400 clips each.
- Metadata: `meta/esc50.csv`
- Audio directory: `audio/`

The main dataset is distributed under CC BY-NC; the ESC-10 subset has
a different license. Consult the original repository's `LICENSE` file
for clip-level attribution details.

## Environment

From the project root, activate the intended Python environment and
install the dependencies:

```bash
python -m pip install -r requirements_member1.txt
```

## Reproduction commands

Run commands from the project root. The commands below use the current
v2 experimental pipeline:

```bash
python scripts/01_data_audit.py
python scripts/02_make_group_split_v2.py
python scripts/03_extract_mfcc_v2.py
python scripts/04_baseline_eval_v2.py
python scripts/05_noise_augmentation.py
```

The split-generation command recreates
`results/split_manifest_v2.csv`. Before rerunning it after team approval,
preserve the agreed manifest or coordinate with the team first.

The feature extraction and evaluation scripts create or update files
under `features/` and `results/`. Keep the environment, configuration,
and dataset version consistent when reproducing results.

## Feature extraction and baseline

The current MFCC baseline uses:

- Mono audio resampled to 22,050 Hz.
- Fixed duration of five seconds, padding short clips and cropping
  longer clips.
- 40 MFCC coefficients with `n_fft=2048` and `hop_length=512`.
- Mean and standard deviation for each coefficient, yielding 80 features.
- `StandardScaler` followed by an RBF SVM (`C=10`, `gamma="scale"`).

The baseline pipeline fits the scaler and classifier on training data
only. Validation data is used for model development; test results should
be reserved for final evaluation decisions.

## Preliminary results

The current single-run results on the proposed v2 split are:

| Model | Clean test accuracy | Clean test macro-F1 |
|---|---:|---:|
| Clean-only SVM | 0.520 | 0.497 |
| Noise-augmented SVM | 0.460 | 0.443 |

Preliminary synthetic-noise test accuracy:

| SNR | Clean-only SVM | Noise-augmented SVM |
|---:|---:|---:|
| 10 dB | 0.220 | 0.475 |
| 5 dB | 0.105 | 0.480 |
| 0 dB | 0.040 | 0.450 |
| -5 dB | 0.025 | 0.390 |

These are results from one experimental run, not multi-seed final
results. The noise-augmented model uses clean training examples plus
two noisy copies per training recording, so the training set increases
from 1,280 to 3,840 examples. Its clean-accuracy trade-off should be
reported alongside the noisy-test improvement.

## Noise experiment and limitations

The stress test mixes synthetic white Gaussian noise into test audio at
10, 5, 0, and -5 dB SNR. This is a controlled corruption experiment,
not evaluation on a real-world noise corpus.

Noise augmentation is generated from training audio only. Validation
and test examples are not used to create augmented training examples.

The final project must still follow the team's agreed multi-seed
protocol and any requirement to evaluate on a public real-noise dataset.
Those items are not completed by the current single-run synthetic-noise
results alone.

## Team coordination and fair comparison

Before adopting the v2 split, all model owners should agree on the same
manifest, label mapping, evaluation files, and noise conditions. Model
owners may use different input representations, but must document them
and follow the assignment's common evaluation protocol.

Do not directly compare results from the old official-fold
cross-validation experiment with results from this proposed
train/validation/test split as if they came from the same protocol.

## Known limitations

- ESC-50 is relatively small, with 40 clips per class.
- Synthetic white noise does not represent all real environmental noise.
- The v2 split is proposed and not yet approved by the full team.
- Results shown above are single-run preliminary results.
- Per-clip attribution and licensing details must be checked in the
  original dataset LICENSE file.
