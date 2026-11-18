from pathlib import Path
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "audio"
META = ROOT / "meta" / "esc50.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

def main():
    if not META.exists():
        raise FileNotFoundError(f"Metadata not found: {META}")
    if not AUDIO.exists():
        raise FileNotFoundError(f"Audio folder not found: {AUDIO}")
    df = pd.read_csv(META)
    required = {"filename", "fold", "target", "category", "src_file", "take"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    summary = {
        "metadata_rows": len(df),
        "unique_filenames": df["filename"].nunique(),
        "duplicate_filenames": int(df["filename"].duplicated().sum()),
        "classes": df["category"].nunique(),
        "folds": df["fold"].nunique(),
        "src_file_unique": df["src_file"].nunique(),
    }
    pd.DataFrame([{"measure": k, "value": v} for k, v in summary.items()]).to_csv(
        REPORTS / "metadata_audit.csv", index=False
    )
    df.groupby(["target", "category"]).size().rename("count").reset_index().to_csv(
        REPORTS / "class_counts.csv", index=False
    )
    df.groupby("fold").agg(n_files=("filename", "size"), n_classes=("category", "nunique")).reset_index().to_csv(
        REPORTS / "fold_counts.csv", index=False
    )

    # Identify source IDs that appear in multiple official folds; report only, do not assume these are proven leakage.
    source_fold_counts = df.groupby("src_file")["fold"].nunique()
    cross = source_fold_counts[source_fold_counts > 1].rename("n_folds").reset_index()
    if len(cross):
        cross = cross.merge(df[["src_file", "filename", "fold", "category", "take"]], on="src_file", how="left")
        cross.to_csv(REPORTS / "cross_fold_source_exceptions.csv", index=False)
    else:
        pd.DataFrame(columns=["src_file", "n_folds"]).to_csv(REPORTS / "cross_fold_source_exceptions.csv", index=False)

    inspection = []
    missing_files = []
    for row in df.itertuples(index=False):
        path = AUDIO / row.filename
        if not path.exists():
            missing_files.append(row.filename)
            continue
        try:
            info = sf.info(str(path))
            inspection.append({
                "filename": row.filename, "samplerate": info.samplerate,
                "channels": info.channels, "duration_seconds": info.duration,
                "frames": info.frames, "format": info.format, "subtype": info.subtype,
                "readable": True
            })
        except Exception as e:
            inspection.append({"filename": row.filename, "readable": False, "error": str(e)})
    pd.DataFrame(inspection).to_csv(REPORTS / "audio_file_inspection.csv", index=False)
    pd.DataFrame({"missing_filename": missing_files}).to_csv(REPORTS / "missing_audio_files.csv", index=False)

    print("AUDIT COMPLETE")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Missing audio files: {len(missing_files)}")
    print(f"Reports: {REPORTS}")

if __name__ == "__main__":
    main()
