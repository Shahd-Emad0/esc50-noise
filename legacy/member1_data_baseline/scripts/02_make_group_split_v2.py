
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "meta" / "esc50.csv"
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"

SEED = 42

def main():
    df = pd.read_csv(META)

    required = {"filename", "target", "category", "src_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["src_file"].isna().any():
        raise ValueError("Missing src_file values.")

    # First: reserve approximately 20% for test.
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    trainval_idx, test_idx = next(
        outer.split(df, y=df["target"], groups=df["src_file"])
    )

    trainval = df.iloc[trainval_idx].copy()
    test = df.iloc[test_idx].copy()

    # Then: reserve approximately 20% of trainval for validation.
    # This aims for roughly 64/16/20 overall.
    inner = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED + 1
    )
    train_rel_idx, val_rel_idx = next(
        inner.split(
            trainval,
            y=trainval["target"],
            groups=trainval["src_file"]
        )
    )

    train = trainval.iloc[train_rel_idx].copy()
    val = trainval.iloc[val_rel_idx].copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    manifest = pd.concat([train, val, test], ignore_index=True)
    manifest = manifest[
        ["filename", "target", "category", "src_file", "fold", "take", "split"]
    ]

    # Verify every file appears once and source groups never overlap.
    assert len(manifest) == len(df)
    assert manifest["filename"].nunique() == len(df)

    source_sets = {
        split: set(manifest.loc[manifest["split"] == split, "src_file"])
        for split in ["train", "validation", "test"]
    }
    assert source_sets["train"].isdisjoint(source_sets["validation"])
    assert source_sets["train"].isdisjoint(source_sets["test"])
    assert source_sets["validation"].isdisjoint(source_sets["test"])

    RESULTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    manifest.to_csv(RESULTS / "split_manifest_v2.csv", index=False)

    class_counts = (
        manifest.groupby(["split", "category"])
        .size().rename("count").reset_index()
    )
    class_counts.to_csv(REPORTS / "split_class_counts_v2.csv", index=False)

    sizes = (
        manifest.groupby("split")
        .agg(n_files=("filename", "size"),
             n_sources=("src_file", "nunique"))
        .reset_index()
    )

    print("GROUP-AWARE STRATIFIED SPLIT — EXPERIMENT ONLY")
    print(sizes.to_string(index=False))
    print("\nClass counts per split:")
    print(class_counts.pivot(
        index="category", columns="split", values="count"
    ).fillna(0).to_string())
    print("\nNo source-group overlaps detected.")
    print(f"\nManifest: {RESULTS / 'split_manifest_v2.csv'}")
    print(f"Class counts: {REPORTS / 'split_class_counts_v2.csv'}")
    print("\nReview the results before adopting this split.")

if __name__ == "__main__":
    main()
