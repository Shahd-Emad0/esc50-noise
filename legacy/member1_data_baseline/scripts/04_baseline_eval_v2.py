
from pathlib import Path
import time
import platform
import sys

import joblib
import librosa
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
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

    return np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
    ]).astype(np.float32)


def main():
    RESULTS.mkdir(exist_ok=True)

    if not FEATURES.exists() or not SPLIT.exists():
        raise FileNotFoundError(
            "Required v2 feature or split file is missing."
        )

    manifest = pd.read_csv(SPLIT)
    feats = pd.read_csv(FEATURES)

    if manifest["filename"].duplicated().any():
        raise ValueError("Duplicate filenames in split manifest.")

    if feats["filename"].duplicated().any():
        raise ValueError("Duplicate filenames in feature file.")

    # Keep only features here: split labels come from the manifest.
    feature_cols = (
        sorted(c for c in feats.columns if c.startswith("mfcc_mean_"))
        + sorted(c for c in feats.columns if c.startswith("mfcc_std_"))
    )  

    if len(feature_cols) != 80:
        raise ValueError(
            f"Expected 80 MFCC features, found {len(feature_cols)}."
        )

    if set(manifest["filename"]) != set(feats["filename"]):
        raise ValueError("Manifest and feature filenames do not match.")

    df = manifest.merge(
        feats[["filename"] + feature_cols],
        on="filename",
        how="inner",
        validate="one_to_one",
    )

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Train, validation, or test split is empty.")

    X_train = train[feature_cols].to_numpy()
    y_train = train["target"].to_numpy()
    X_val = val[feature_cols].to_numpy()
    y_val = val["target"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["target"].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale"),
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    def evaluate(name, X, y):
        start = time.perf_counter()
        pred = model.predict(X)
        elapsed = time.perf_counter() - start

        metrics = {
            "split": name,
            "n_samples": len(y),
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(
                y, pred, average="macro", zero_division=0
            ),
            "train_seconds": train_seconds,
            "inference_seconds": elapsed,
            "inference_ms_per_sample": (
                elapsed * 1000 / max(len(y), 1)
            ),
        }
        return pred, metrics

    val_pred, val_metrics = evaluate(
        "validation_clean", X_val, y_val
    )
    test_pred, test_metrics = evaluate(
        "test_clean", X_test, y_test
    )

    pd.DataFrame([val_metrics, test_metrics]).to_csv(
        RESULTS / "baseline_metrics_v2.csv", index=False
    )

    pd.DataFrame({
        "filename": val["filename"].to_numpy(),
        "y_true": y_val,
        "y_pred": val_pred,
    }).to_csv(
        RESULTS / "baseline_validation_predictions_v2.csv",
        index=False,
    )

    pd.DataFrame({
        "filename": test["filename"].to_numpy(),
        "y_true": y_test,
        "y_pred": test_pred,
    }).to_csv(
        RESULTS / "baseline_test_predictions_v2.csv",
        index=False,
    )

    class_table = (
        manifest[["target", "category"]]
        .drop_duplicates()
        .sort_values("target")
    )
    classes = class_table["target"].tolist()
    class_names = class_table["category"].tolist()

    report = classification_report(
        y_test,
        test_pred,
        labels=classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULTS / "baseline_test_classification_report_v2.csv"
    )

    cm = confusion_matrix(y_test, test_pred, labels=classes)
    pd.DataFrame(
        cm, index=class_names, columns=class_names
    ).to_csv(
        RESULTS / "baseline_test_confusion_matrix_v2.csv"
    )

    # Test noise robustness without retraining on noisy test data.
    noisy_metrics = []
    noisy_predictions = []

    for snr_index, snr_db in enumerate(SNR_LEVELS_DB):
        preds = []
        truths = []
        filenames = []

        start = time.perf_counter()

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

            prediction = model.predict(feature)[0]

            preds.append(prediction)
            truths.append(row.target)
            filenames.append(row.filename)

        elapsed = time.perf_counter() - start

        noisy_metrics.append({
            "split": "test_noisy",
            "snr_db": snr_db,
            "n_samples": len(truths),
            "accuracy": accuracy_score(truths, preds),
            "macro_f1": f1_score(
                truths, preds, average="macro", zero_division=0
            ),
            "elapsed_seconds_including_audio_and_feature_extraction":
                elapsed,
        })

        for filename, truth, prediction in zip(
            filenames, truths, preds
        ):
            noisy_predictions.append({
                "filename": filename,
                "snr_db": snr_db,
                "y_true": truth,
                "y_pred": prediction,
            })

    pd.DataFrame(noisy_metrics).to_csv(
        RESULTS / "baseline_noisy_test_metrics_v2.csv",
        index=False,
    )
    pd.DataFrame(noisy_predictions).to_csv(
        RESULTS / "baseline_noisy_test_predictions_v2.csv",
        index=False,
    )

    model_path = RESULTS / "baseline_svm_v2.joblib"
    joblib.dump(model, model_path)

    pd.DataFrame([{
        "model": "StandardScaler + RBF SVC",
        "features": len(feature_cols),
        "kernel": "rbf",
        "C": 10,
        "gamma": "scale",
        "train_samples": len(train),
        "validation_samples": len(val),
        "test_samples": len(test),
        "train_seconds": train_seconds,
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }]).to_csv(
        RESULTS / "baseline_cost_metrics_v2.csv", index=False
    )

    print("\nBASELINE V2 COMPLETE")
    print("\nClean metrics:")
    print(pd.DataFrame([val_metrics, test_metrics]).to_string(index=False))
    print("\nNoisy test metrics:")
    print(pd.DataFrame(noisy_metrics).to_string(index=False))
    print(f"\nModel saved: {model_path}")
    print(f"Results folder: {RESULTS}")


if __name__ == "__main__":
    main()
