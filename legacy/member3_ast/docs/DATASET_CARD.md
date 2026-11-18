# Dataset Card — ESC-50

## Identity and purpose
ESC-50 is a labeled environmental-sound classification dataset introduced by Karol J. Piczak. It contains 2,000 five-second audio recordings across 50 sound classes, with 40 examples per class. The recordings are WAV, mono, and originally 44.1 kHz. The metadata file is `meta/esc50.csv`.

Official source: https://github.com/karolpiczak/ESC-50  
Paper: K. J. Piczak, “ESC: Dataset for Environmental Sound Classification,” ACM Multimedia, 2015. DOI: https://doi.org/10.1145/2733373.2806390

## License and attribution
The main ESC-50 dataset is distributed under Creative Commons Attribution-NonCommercial (CC BY-NC). The ESC-10 subset is marked separately and has a different CC BY license. Check the original repository's `LICENSE` file for clip-level attributions. Do not assume all clips have the same attribution details.

## Metadata fields
- `filename`: audio file name
- `fold`: official cross-validation fold (1–5)
- `target`: integer class label (0–49)
- `category`: human-readable class
- `esc10`: whether the clip belongs to the ESC-10 subset
- `src_file`: identifier of the original Freesound source
- `take`: fragment/take identifier

## Class and fold structure
Expected totals: 2,000 rows, 50 categories, 40 clips/category, five official folds with 400 clips/fold and eight examples per class per fold. Confirm actual counts from generated audit reports before publication.

The official folds are commonly used for cross-validation. This project also generates a group-aware train/validation/test split using `src_file` as the grouping variable, because some `src_file` values in our audit appear across more than one official fold. Treat these as audit exceptions requiring documentation, not automatic proof of leakage.

## Preprocessing used in Member 1 baseline
- Load audio as mono.
- Resample to 22,050 Hz.
- Use a fixed 5-second duration (pad if shorter; crop if longer).
- Compute 40 MFCC coefficients with `n_fft=2048`, `hop_length=512`.
- Summarize each coefficient using mean and standard deviation across frames.
- Result: 80 features per recording.
- Baseline classifier: `StandardScaler` followed by RBF SVM (`C=10`, `gamma="scale"`).

## Noise experiment
Synthetic white Gaussian noise is mixed into audio at SNR levels 10, 5, 0, and -5 dB. The SNR is set by signal/noise power. This is a controlled stress test and must not be described as a real-noise dataset evaluation. No separate public noise corpus is currently included in this baseline package; the team should decide whether to add one.

## Known limitations
- ESC-50 is small and class count is limited to 40 clips per category.
- Real-world recording conditions vary.
- Some original source recordings may have been preprocessed in ways that correlate with labels; see the dataset repository's caveats.
- Synthetic white noise does not represent all real environmental noise.
