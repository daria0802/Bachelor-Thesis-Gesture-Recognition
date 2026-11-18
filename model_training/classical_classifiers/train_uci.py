#!/usr/bin/env python3
# File:        train_uci.py
# Author:      Daria Kinash
# Description: Train and evaluate five classical ML classifiers on the UCI HAR dataset.

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = (_SCRIPT_DIR / ".." / "data" / "UCI HAR Dataset").resolve()

LABELS_FILTER = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


# Data loading & feature extraction

def load_labels_map(path: Path) -> dict:
    """Read activity_labels.txt and return a mapping from integer ID to class name."""
    df = pd.read_csv(path, header=None, sep=r"\s+", names=["id", "label"])
    return dict(zip(df.id, df.label))


def read_raw_signals(base_path: Path, subset: str, label_map: dict, labels_filter: list):
    """Load body_acc and body_gyro signal files for one subset (train or test).
       Returns a dict of signal arrays and the corresponding label array."""
    signals = {}
    for c in ["body_acc", "body_gyro"]:
        for ax in ["x", "y", "z"]:
            fname = f"{c}_{ax}_{subset}.txt"
            signals[f"{c}_{ax}"] = pd.read_csv(
                base_path / fname, header=None, sep=r"\s+"
            ).to_numpy()

    y = pd.read_csv(base_path / f"y_{subset}.txt", header=None, sep=r"\s+", names=["activity_id"])
    y["label"] = y["activity_id"].map(label_map)

    mask = y["label"].isin(labels_filter)
    y_filtered = y[mask]["label"].copy().to_numpy()

    for k in signals:
        signals[k] = signals[k][mask]

    return signals, y_filtered


def feature_extraction(signals: dict) -> pd.DataFrame:
    """Compute a 30-dimensional feature vector."""
    n_samples = signals["body_acc_x"].shape[0]
    X_features = []

    for i in range(n_samples):
        feats = {}

        def calc_feats(arr, name):
            mean_val = np.mean(arr)
            feats[f"{name}_mean"] = mean_val
            feats[f"{name}_mad"] = np.mean(np.abs(arr - mean_val))
            feats[f"{name}_min"] = np.min(arr)
            feats[f"{name}_max"] = np.max(arr)
            centered = arr - mean_val
            feats[f"{name}_zcr"] = int(((centered[:-1] * centered[1:]) < 0).sum())

        calc_feats(signals["body_acc_x"][i],   "ax")
        calc_feats(signals["body_acc_y"][i],   "ay")
        calc_feats(signals["body_acc_z"][i],   "az")
        calc_feats(signals["body_gyro_x"][i],  "gx")
        calc_feats(signals["body_gyro_y"][i],  "gy")
        calc_feats(signals["body_gyro_z"][i],  "gz")

        X_features.append(feats)

    return pd.DataFrame(X_features)


# Evaluation helpers

def plot_confusion_matrix(y_true, y_pred, model_name, outdir: Path, show: bool):
    """Save a confusion matrix as a PNG file."""
    import matplotlib.pyplot as plt

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = outdir / f"cm_{safe_name}.png"
    fig.savefig(out_path, dpi=120)
    print(f"  Saved confusion matrix -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def evaluate_classifier(clf, X_train, X_test, y_train, y_test, model_name, outdir, show):
    """Train a classifier, evaluate on the test set, and print the classification report."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, model_name, outdir, show)
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(y_test, y_pred))
    return clf, acc


def plot_accuracy_bar(results: dict, outdir: Path, show: bool):
    """Save a bar chart comparing accuracy across all classifiers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdae6b"]
    ax.bar(results.keys(), results.values(), color=colors[: len(results)])
    ax.set_ylim(0.0, 1.05)
    ax.set_title("UCI HAR — Algorithm Comparison")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(results.values()):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center")
    fig.tight_layout()
    out_path = outdir / "accuracy_bar.png"
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train classical classifiers on UCI HAR dataset.")
    p.add_argument("--labels", nargs="+", default=LABELS_FILTER,
                   help=f"Activity labels to include")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt-max-depth", type=int, default=10,
                   help="DecisionTree max_depth; -1 means unlimited (default: 10).")
    p.add_argument("--rf-n-estimators", type=int, default=120,
                   help="RandomForest n_estimators.")
    p.add_argument("--rf-max-depth", type=int, default=-1,
                   help="RandomForest max_depth per tree; -1 means unlimited.")
    p.add_argument("--show-plots", action="store_true",
                   help="Display plots")
    return p.parse_args()


def main():
    """Load UCI HAR data, extract features, train all classifiers, and save evaluation plots."""
    args = parse_args()

    if not args.show_plots:
        matplotlib.use("Agg")

    outdir = Path("runs/uci")
    outdir.mkdir(parents=True, exist_ok=True)

    label_map = load_labels_map(_DEFAULT_DATA / "activity_labels.txt")

    raw_train, y_train = read_raw_signals(_DEFAULT_DATA / "train", "train", label_map, args.labels)
    raw_test,  y_test  = read_raw_signals(_DEFAULT_DATA / "test",  "test",  label_map, args.labels)

    print("Extracting features…")
    X_train_df = feature_extraction(raw_train)
    X_test_df  = feature_extraction(raw_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_df)
    X_test_s  = scaler.transform(X_test_df)

    classifiers = [
        ("Logistic Regression",            LogisticRegression(max_iter=1000, solver="lbfgs")),
        ("Support Vector Machine (Linear)", LinearSVC(multi_class="ovr", C=1.0, random_state=args.seed)),
        ("Decision Tree",                  DecisionTreeClassifier(
            random_state=args.seed,
            max_depth=None if args.dt_max_depth == -1 else args.dt_max_depth,
        )),
        ("K-Nearest Neighbors (k=3)",      KNeighborsClassifier(n_neighbors=3)),
        ("Random Forest",                  RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            max_depth=None if args.rf_max_depth == -1 else args.rf_max_depth,
        )),
    ]

    results = {}
    for name, clf in classifiers:
        _, acc = evaluate_classifier(clf, X_train_s, X_test_s, y_train, y_test,
                                     name, outdir, args.show_plots)
        results[name] = acc

    print("\n=== Summary ===")
    for k, v in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v * 100:.2f}%")

    plot_accuracy_bar(results, outdir, args.show_plots)


if __name__ == "__main__":
    main()
