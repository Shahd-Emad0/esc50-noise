
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "audio"
MANIFEST = ROOT / "results" / "split_manifest_v2.csv"
FEATURES = ROOT / "features"

SAMPLE_RATE = 22050
DURATION_SECONDS = 5
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512


def extract_one(path):
    y, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION_SECONDS
    )

    expected = SAMPLE_RATE * DURATION_SECONDS

    if len(y) < expected:
        y = np.pad(y, (0, expected - len(y)))
    elif len(y) > expected:
        y = y[:expected]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    feat = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1)
    ])

    return feat.astype(np.float32)


def main():
    FEATURES.mkdir(exist_ok=True)

    df = pd.read_csv(MANIFEST)

    required = {
        "filename", "target", "category",
        "src_file", "fold", "split"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found in split manifest.")

    rows = []
    errors = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        path = AUDIO / row.filename

        try:
            feat = extract_one(path)

            item = {
                "filename": row.filename,
                "target": row.target,
                "category": row.category,
                "src_file": row.src_file,
                "fold": row.fold,
                "split": row.split
            }

            for j in range(N_MFCC):
                item[f"mfcc_mean_{j+1:02d}"] = feat[j]
                item[f"mfcc_std_{j+1:02d}"] = feat[N_MFCC + j]

            rows.append(item)

        except Exception as e:
            errors.append({
                "filename": row.filename,
                "error": repr(e)
            })

        if i % 100 == 0:
            print(f"Processed {i}/{len(df)}")

    out = pd.DataFrame(rows)

    out.to_csv(FEATURES / "mfcc_features_v2.csv", index=False)

    pd.DataFrame(
        errors, columns=["filename", "error"]
    ).to_csv(
        FEATURES / "mfcc_extraction_errors_v2.csv",
        index=False
    )

    print(f"\nExtracted: {len(out)} / {len(df)}")
    print(f"Errors: {len(errors)}")
    print(f"Features saved to: {FEATURES / 'mfcc_features_v2.csv'}")

    if errors:
        raise RuntimeError(
            "Some files failed. Review the extraction errors CSV."
        )


if __name__ == "__main__":
    main()
