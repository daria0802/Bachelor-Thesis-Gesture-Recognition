# File:        clean_dataset.py
# Author:      Daria Kinash
# Description: Prepares raw gesture CSV files for training.
#              Reads the activity label and subject ID from the filename, adds them as columns, removes rows with missing values, and writes the result to an output folder.

import argparse
from pathlib import Path
import pandas as pd

GESTURE_ACTIVITY_NAMES = ("idle", "accept", "left", "right", "cancel")

def label_from_filename(filename, activity_names):
    """Return the activity label found in the filename, or None if not found."""
    name_low = filename.lower()
    for word in activity_names:
        if word in name_low:
            return word
    return None

def subject_from_filename(filename):
    """Return the subject ID from the last underscore-separated part of the filename.

    Example: 'accept_d1.CSV' -> 'd1'
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[-1] if parts else stem

def list_csvs(root):
    """Return a sorted list of all CSV files under root."""
    root = Path(root)
    out = []
    seen = set()
    for p in root.rglob("*.csv"):
        if p.is_file() and p.name not in seen:
            seen.add(p.name)
            out.append(p)
    for p in root.rglob("*.CSV"):
        if p.is_file() and p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return sorted(out, key=lambda p: p.name)

def clean_and_add_label(input_path, output_path, activity_names):
    """Add label and subject columns to one CSV file and drop rows with NaN values. Returns the number of rows in the cleaned file."""
    df = pd.read_csv(input_path)
    df.columns = [str(c).strip() for c in df.columns]

    label = label_from_filename(input_path.name, activity_names)
    if label is None:
        raise ValueError("No label found in filename: " + input_path.name)
    df["label"] = label
    df["subject"] = subject_from_filename(input_path.name)

    df = df.dropna(subset=df.columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return len(df)


def main():
    p = argparse.ArgumentParser(description="Add label and subject columns to gesture CSVs.")
    p.add_argument("--input",  "-i", type=str, default=None, help="Input folder with raw CSVs")
    p.add_argument("--output", "-o", type=str, default=None, help="Output folder for labeled CSVs")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_root  = Path(args.input)  if args.input  else (script_dir / ".." / "data" / "hands").resolve()
    output_root = Path(args.output) if args.output else input_root.parent / (input_root.name + "_labeled")

    files = list_csvs(input_root)
    if not files:
        print("No CSV files found in", input_root)
        return

    for f in files:
        out_path = output_root / f.name
        n = clean_and_add_label(f, out_path, GESTURE_ACTIVITY_NAMES)
        print(f.name, "->", out_path, "(", n, "rows)")

    print("Done. Output folder:", output_root)


if __name__ == "__main__":
    main()
