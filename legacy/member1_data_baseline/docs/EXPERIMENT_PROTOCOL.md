# Experiment Protocol — Member 1 Proposal

## 1. Status and approval

This document describes a proposed experimental protocol for Member 1's
classical baseline and noise-augmentation experiments.

The manifest
`results/split_manifest_v2.csv` is a proposed group-aware split.
It has not yet been approved as the shared split by the full team.
Do not treat it as the final common split until all model owners agree.

If the team chooses a different split, regenerate or adopt the agreed
manifest and rerun the affected experiments. Do not combine metrics
from different split manifests in a single direct comparison.

## 2. Dataset and source grouping

Dataset: ESC-50, from
https://github.com/karolpiczak/ESC-50

The dataset contains 2,000 five-second environmental audio clips across
50 classes, with 40 clips per class. The original dataset provides five
official folds of 400 clips each.

The proposed split uses `src_file` as the grouping variable. The goal is
to keep each source group in exactly one partition and reduce the chance
that fragments from the same source occur across training and evaluation.

The data audit identified some source identifiers that appear across
official folds and some associated with multiple categories. These are
audit exceptions to document and review, not automatic proof of leakage.

## 3. Proposed split construction

Script: `scripts/02_make_group_split_v2.py`

- StratifiedGroupKFold with five folds is used to select the test
  partition.
- A second StratifiedGroupKFold split selects validation from the
  remaining training-and-validation data.
- The random seeds are 42 for the outer split and 43 for the inner split.
- The intended overall proportions are approximately 64% train,
  16% validation, and 20% test.
- Group isolation is prioritized; exact per-class counts may vary.

Output:
- `results/split_manifest_v2.csv`
- `reports/split_class_counts_v2.csv`

The script includes assertions for unique filenames and disjoint
source groups. These checks support group isolation for the generated
manifest but do not by themselves establish that the team has approved it.

## 4. Shared preprocessing contract

For the MFCC baseline:

- Load audio as mono.
- Resample to 22,050 Hz.
- Use five seconds of audio; pad short clips and crop longer clips.
- Extract 40 MFCC coefficients with `n_fft=2048` and `hop_length=512`.
- Represent each recording using coefficient-wise means and standard
  deviations, for 80 features total.
- Use the metadata target labels and preserve their mapping to category.

Other model owners may require spectrogram or waveform inputs. Their
representations should be documented while using the same agreed split,
label mapping, and evaluation examples.

## 5. Classical baseline

Script: `scripts/04_baseline_eval_v2.py`

Pipeline:
`StandardScaler -> SVC(kernel="rbf", C=10, gamma="scale")`

The scaler and classifier are fitted on training features only.
Validation and test data are not used to fit either component.

Report accuracy and macro-F1, plus per-class metrics, confusion matrix,
training time, inference time, and model size where available.

## 6. Noise-augmentation experiment

Script: `scripts/05_noise_augmentation.py`

The experiment compares two SVM pipelines with the same SVM settings:

1. Clean-only: clean training features.
2. Noise-augmented: clean training features plus two noisy copies per
   training recording.

For augmentation, each noisy copy receives an SNR selected from
10, 5, 0, and -5 dB. Synthetic white Gaussian noise is generated from
training audio only.

Both models are evaluated on the same clean validation/test examples
and the same test files at the same SNR levels. Test examples are not
used to create augmented training examples.

Important comparison limitation: the augmented model uses 3,840
training examples, compared with 1,280 for the clean-only model.
Therefore, the comparison measures this augmentation-and-data-volume
strategy, not noise augmentation under a fixed training-example budget.

Synthetic white noise is a controlled stress test, not a substitute for
evaluation on a real-world noise corpus.

## 7. Preliminary results and seed requirement

Current single-run results on the proposed v2 split:

| Model | Clean test accuracy | Clean test macro-F1 |
|---|---:|---:|
| Clean-only SVM | 0.520 | 0.497 |
| Noise-augmented SVM | 0.460 | 0.443 |

Noisy test accuracy:

| SNR | Clean-only SVM | Noise-augmented SVM |
|---:|---:|---:|
| 10 dB | 0.220 | 0.475 |
| 5 dB | 0.105 | 0.480 |
| 0 dB | 0.040 | 0.450 |
| -5 dB | 0.025 | 0.390 |

These values are preliminary and come from one run. The assignment's
multi-seed requirement remains outstanding. The team should agree on a
seed protocol, rerun the relevant final configurations, and report
variation across runs before presenting final results.

For the SVM setup, the classifier fit is generally deterministic.
However, augmentation and noise evaluation depend on generated noise;
independent noise seeds should be used if the agreed protocol measures
variation across noise realizations. Keep the split fixed when studying
noise-seed variation, and state clearly what is varied.

## 8. Fair comparison checklist

Before the team-wide comparison:

- Obtain agreement on one shared split manifest.
- Use the same train/validation/test assignments and label mapping.
- Keep test data out of fitting and hyperparameter selection.
- Document each model's input representation and preprocessing.
- Use comparable tuning budgets and the agreed evaluation metrics.
- Evaluate on the same clean test examples and agreed noise conditions.
- Run the required seeds and report mean and variation.
- Do not combine official-fold cross-validation metrics with metrics
  from the proposed v2 split as if they were the same experiment.

## 9. Public real-noise dataset

This package currently evaluates synthetic white Gaussian noise.
If the assignment requires a public real-noise dataset, the team must
select an appropriate corpus, verify its license and suitability, and
document how noise clips are sampled and mixed. This requirement is not
satisfied by synthetic noise alone.

## 10. Dataset license and citation

The ESC-50 repository describes the main dataset as CC BY-NC and the
ESC-10 subset as CC BY. Consult the original LICENSE file for
clip-level attribution details.

Citation:
K. J. Piczak, "ESC: Dataset for Environmental Sound Classification,"
Proceedings of the 23rd ACM International Conference on Multimedia,
2015. https://doi.org/10.1145/2733373.2806390
