"""
Reproducible Titanic data acquisition, quality assessment, and preprocessing.
Run from the repository root: python src/preprocess.py
"""
from pathlib import Path
import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
IMG_DIR = ROOT / "images"
REPORT_DIR = ROOT / "report"
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
RAW_FILE = RAW_DIR / "titanic.csv"

for directory in (RAW_DIR, OUT_DIR, IMG_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def acquire_data():
    """Download the public CSV once; preserve an existing local raw copy."""
    if not RAW_FILE.exists():
        print(f"Downloading dataset from: {DATA_URL}")
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Python data-preprocessing educational project"})
        with urllib.request.urlopen(request, timeout=30) as response:
            RAW_FILE.write_bytes(response.read())
    return pd.read_csv(RAW_FILE)

def iqr_outlier_counts(df, columns):
    """Return IQR outlier counts and bounds; counts are diagnostic, not deletion instructions."""
    results = {}
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        results[col] = {
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower_bound": float(low), "upper_bound": float(high),
            "outlier_count": int(((values < low) | (values > high)).sum())
        }
    return results

def main():
    raw = acquire_data()
    df = raw.copy()

    # Normalize whitespace in text values, without changing the original raw file.
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].replace("", np.nan)

    initial_shape = list(df.shape)
    missing_before = df.isna().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    outliers = iqr_outlier_counts(df, [c for c in ["age", "fare"] if c in df.columns])

    # Domain checks: flag impossible values, do not silently delete them.
    domain_issues = {}
    if "age" in df.columns:
        domain_issues["age_negative_count"] = int((df["age"] < 0).sum())
    if "fare" in df.columns:
        domain_issues["fare_negative_count"] = int((df["fare"] < 0).sum())
    if "survived" in df.columns:
        domain_issues["survived_invalid_count"] = int((~df["survived"].isin([0, 1]) & df["survived"].notna()).sum())
    if "pclass" in df.columns:
        domain_issues["pclass_invalid_count"] = int((~df["pclass"].isin([1, 2, 3]) & df["pclass"].notna()).sum())

    # Remove exact duplicate rows if present, recording the number removed.
    df = df.drop_duplicates().copy()

    # Treat missingness with column-aware rules. Preserve "deck" as Unknown because
    # missing cabin/deck often means the information was not recorded.
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        if df[col].isna().any():
            if col in {"deck", "embark_town", "embarked", "who", "class", "adult_male", "alone"}:
                if col == "deck":
                    df[col] = df[col].astype("object").fillna("Unknown")
                else:
                    mode = df[col].mode(dropna=True)
                    df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
            else:
                df[col] = df[col].fillna("Unknown")

    # Boolean columns may be read as bool and are not missing in this dataset;
    # convert to explicit strings only if missing values need filling.
    missing_after = df.isna().sum().to_dict()
    df.to_csv(OUT_DIR / "titanic_cleaned.csv", index=False)

    # Quality chart: only columns with missing values before cleaning.
    miss = pd.Series(missing_before, dtype="int64")
    miss = miss[miss > 0].sort_values(ascending=False)
    if not miss.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(x=miss.values, y=miss.index)
        plt.title("Missing values by column — before cleaning")
        plt.xlabel("Number of missing values")
        plt.ylabel("Column")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "missing_values_before.png", dpi=160)
        plt.close()

    # Outlier chart for continuous variables.
    plot_cols = [c for c in ["age", "fare"] if c in raw.columns]
    if plot_cols:
        fig, axes = plt.subplots(1, len(plot_cols), figsize=(5 * len(plot_cols), 4))
        if len(plot_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, plot_cols):
            sns.boxplot(y=pd.to_numeric(raw[col], errors="coerce"), ax=ax)
            ax.set_title(f"{col.title()} — raw distribution")
        fig.tight_layout()
        fig.savefig(IMG_DIR / "numeric_outliers_before.png", dpi=160)
        plt.close(fig)

    summary = {
        "source_url": DATA_URL,
        "initial_rows": initial_shape[0],
        "initial_columns": initial_shape[1],
        "columns": list(raw.columns),
        "data_types": dtypes,
        "missing_values_before": {k: int(v) for k, v in missing_before.items()},
        "exact_duplicate_rows_before": duplicate_rows,
        "duplicate_rows_removed": duplicate_rows,
        "numeric_iqr_outliers": outliers,
        "domain_validation_issue_counts": domain_issues,
        "missing_values_after": {k: int(v) for k, v in missing_after.items()},
        "final_rows": int(df.shape[0]),
        "final_columns": int(df.shape[1]),
        "cleaned_file": "data/processed/titanic_cleaned.csv",
        "notes": [
            "IQR outliers are flagged for review, not automatically deleted.",
            "Median/mode imputation is a simple baseline; it may reduce variability.",
            "For model evaluation, fit preprocessing steps on training folds only."
        ]
    }
    (OUT_DIR / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "run_summary.txt").write_text(
        "RUN RESULTS — generated from the downloaded local dataset\n\n" +
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nDone. Review data/processed/, images/, and report/run_summary.txt.")

if __name__ == "__main__":
    main()
