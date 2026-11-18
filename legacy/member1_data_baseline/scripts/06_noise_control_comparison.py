
from pathlib import Path
import time

import joblib
import librosa
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "audio"
FEATURES = ROOT / "features" / "mfcc_features_v2.csv"
SPLIT = ROOT / "results" / "split_manifest_v2.csv"
RESULTS = ROOT / "results"

SAMPLE_RATE = 22050
DURATION_SECONDS = 5
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
SNR_LEVELS_DB = [10, 5, 0, -5]
NOISE_SEED = 42

# Number of noisy copies per clean training recording.
# Each copy gets a randomly selected SNR from this list.
AUGMENTATION_COPIES = 2


def load_audio(path):
    y, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION_SECONDS,
    )

    expected = SAMPLE_RATE * DURATION_SECONDS

    if len(y) < expected:
        y = np.pad(y, (0, expected - len(y)))

    return y[:expected].astype(np.float32)


def add_white_noise_at_snr(audio, snr_db, seed):
    audio = np.asarray(audio, dtype=np.float32)
    signal_power = float(np.mean(audio ** 2))

    if signal_power < 1e-12:
        return audio.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, size=audio.shape).astype(np.float32)
    noise_power = float(np.mean(noise ** 2))

    desired_noise_power = signal_power / (10 ** (snr_db / 10))

    noise *= np.sqrt(
        desired_noise_power / max(noise_power, 1e-12)
    )

    return audio + noise


def extract_feature_from_audio(y):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    return np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1)]
    ).astype(np.float32)


def main():
    if not FEATURES.exists() or not SPLIT.exists():
        raise FileNotFoundError(
            "Missing MFCC features or v2 split manifest."
        )

    manifest = pd.read_csv(SPLIT)
    feats = pd.read_csv(FEATURES)

    feature_cols = (
        sorted(c for c in feats.columns if c.startswith("mfcc_mean_"))
        + sorted(c for c in feats.columns if c.startswith("mfcc_std_"))
    )

    if len(feature_cols) != 80:
        raise ValueError(f"Expected 80 features, found {len(feature_cols)}.")

    if manifest["filename"].duplicated().any():
        raise ValueError("Duplicate filenames in manifest.")

    if feats["filename"].duplicated().any():
        raise ValueError("Duplicate filenames in feature file.")

    if set(manifest["filename"]) != set(feats["filename"]):
        raise ValueError("Feature filenames and manifest do not match.")

    df = manifest.merge(
        feats[["filename"] + feature_cols],
        on="filename",
        validate="one_to_one",
    )

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    # Baseline training data: clean training features only.
    X_train_clean = train[feature_cols].to_numpy(dtype=np.float32)
    y_train = train["target"].to_numpy()

    # Validation and test features are never used for fitting.
    X_val = val[feature_cols].to_numpy(dtype=np.float32)
    y_val = val["target"].to_numpy()
    X_test = test[feature_cols].to_numpy(dtype=np.float32)
    y_test = test["target"].to_numpy()

    # Build augmentation from training audio only.
    rng = np.random.default_rng(NOISE_SEED)
    noisy_features = []
    noisy_labels = []

    print("Creating noisy training copies...")

    for row in train.itertuples(index=False):
        audio = load_audio(AUDIO / row.filename)

        for copy_index in range(AUGMENTATION_COPIES):
            snr_db = int(rng.choice(SNR_LEVELS_DB))
            seed = NOISE_SEED + copy_index * 100000 + sum(
                (i + 1) * ord(char)
                for i, char in enumerate(str(row.filename))
            )

            noisy_audio = add_white_noise_at_snr(
                audio, snr_db, seed
            )

            noisy_features.append(
                extract_feature_from_audio(noisy_audio)
            )
            noisy_labels.append(row.target)

    X_noisy = np.asarray(noisy_features, dtype=np.float32)
    y_noisy = np.asarray(noisy_labels)

    # Train three models with the same SVM configuration.
    models = {
        "clean_only": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale"),
        ),
        "clean_upsampled": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale"),
        ),
        "noise_augmented": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale"),
        ),
    }

    # Match the augmented dataset size without introducing noisy audio.
    X_train_up = np.tile(X_train_clean, (3, 1))
    y_train_up = np.tile(y_train, 3)

    X_train_aug = np.vstack([X_train_clean, X_noisy])
    y_train_aug = np.concatenate([y_train, y_noisy])

    records = []
    noisy_test_records = []

    for model_name, model in models.items():
        if model_name == "clean_only":
            X_fit, y_fit = X_train_clean, y_train
        elif model_name == "clean_upsampled":
            X_fit, y_fit = X_train_up, y_train_up
        else:
            X_fit, y_fit = X_train_aug, y_train_aug

        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        train_seconds = time.perf_counter() - start

        for split_name, X_eval, y_eval in [
            ("validation_clean", X_val, y_val),
            ("test_clean", X_test, y_test),
        ]:
            start = time.perf_counter()
            predictions = model.predict(X_eval)
            inference_seconds = time.perf_counter() - start

            records.append({
                "model": model_name,
                "split": split_name,
                "n_samples": len(y_eval),
                "accuracy": accuracy_score(y_eval, predictions),
                "macro_f1": f1_score(
                    y_eval, predictions,
                    average="macro", zero_division=0
                ),
                "train_samples": len(y_fit),
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
            })

        # Evaluate on exactly the same test files and SNR levels.
        for snr_index, snr_db in enumerate(SNR_LEVELS_DB):
            truths = []
            predictions = []

            for row in test.itertuples(index=False):
                audio = load_audio(AUDIO / row.filename)

                filename_seed = sum(
                    (i + 1) * ord(char)
                    for i, char in enumerate(str(row.filename))
                )

                noisy_audio = add_white_noise_at_snr(
                    audio,
                    snr_db,
                    NOISE_SEED + filename_seed + snr_index * 100000,
                )

                feature = extract_feature_from_audio(
                    noisy_audio
                ).reshape(1, -1)

                predictions.append(model.predict(feature)[0])
                truths.append(row.target)

            noisy_test_records.append({
                "model": model_name,
                "split": "test_noisy",
                "snr_db": snr_db,
                "n_samples": len(truths),
                "accuracy": accuracy_score(truths, predictions),
                "macro_f1": f1_score(
                    truths, predictions,
                    average="macro", zero_division=0
                ),
            })

        model_path = RESULTS / f"{model_name}_noise_control_v2.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")

    pd.DataFrame(records).to_csv(
        RESULTS / "noise_control_clean_metrics_v2.csv",
        index=False,
    )

    pd.DataFrame(noisy_test_records).to_csv(
        RESULTS / "noise_control_noisy_test_metrics_v2.csv",
        index=False,
    )

    print("\nNOISE AUGMENTATION EXPERIMENT COMPLETE")
    print("\nClean validation/test metrics:")
    print(pd.DataFrame(records).to_string(index=False))
    print("\nNoisy test metrics:")
    print(pd.DataFrame(noisy_test_records).to_string(index=False))
    print("\nResults saved in:", RESULTS)


if __name__ == "__main__":
    main()
