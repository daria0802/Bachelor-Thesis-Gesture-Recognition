#!/usr/bin/env python3
# File:        train_gesture.py
# Author:      Daria Kinash
# Description: Train and evaluate five classical ML classifiers on the hand gesture dataset.

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = (_SCRIPT_DIR / ".." / "data" / "hands_labeled").resolve()

GESTURE_LABEL_ORDER = ("idle", "accept", "left", "right", "cancel")
VALID_LABELS = set(GESTURE_LABEL_ORDER)

_CHAN = ("ax", "ay", "az", "gx", "gy", "gz")
_STAT = ("mean", "mad", "min", "max", "zcr")
FEATURE_NAMES = [f"{c}_{s}" for c in _CHAN for s in _STAT]  # 30 features total


# Data loading & feature extraction

def read_data(path: str) -> pd.DataFrame:
    """Load one labeled CSV file and return a cleaned DataFrame with renamed channel columns."""
    df = pd.read_csv(Path(path))
    df = df[[
        "timestamp_ms", "ax_g", "ay_g", "az_g",
        "gx_dps", "gy_dps", "gz_dps", "label", "subject",
    ]].copy()
    df["ax"] = df["ax_g"].astype(float)
    df["ay"] = df["ay_g"].astype(float)
    df["az"] = df["az_g"].astype(float)
    df["gx"] = df["gx_dps"].astype(float)
    df["gy"] = df["gy_dps"].astype(float)
    df["gz"] = df["gz_dps"].astype(float)
    df["time_s"] = df["timestamp_ms"].astype(float) / 1000.0
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)].copy()
    return df[["time_s", "ax", "ay", "az", "gx", "gy", "gz", "label", "subject"]].copy()


def extract_feats_from_window(ax, ay, az, gx, gy, gz) -> np.ndarray:
    """Compute a 30-dimensional feature vector from one IMU window.

    For each of the 6 channels: mean, MAD, min, max, and mean-centred ZCR.
    """
    X_f = []
    for sig in [ax, ay, az, gx, gy, gz]:
        sig = sig.astype(float)
        m = np.mean(sig)
        X_f.append(float(m))
        X_f.append(float(np.mean(np.abs(sig - m))))
        X_f.append(float(np.min(sig)))
        X_f.append(float(np.max(sig)))
        centered = sig - m
        X_f.append(float(int(((centered[:-1] * centered[1:]) < 0).sum())))
    return np.array(X_f, dtype=np.float32)


def make_windows(df: pd.DataFrame, win: int, hop: int):
    """Split the DataFrame into overlapping windows and extract features."""
    X, y, groups = [], [], []
    N = len(df)
    if N < win:
        return (
            np.empty((0, 30), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )
    for start in range(0, N - win + 1, hop):
        chunk = df.iloc[start: start + win]
        xf = extract_feats_from_window(
            chunk["ax"].to_numpy(), chunk["ay"].to_numpy(), chunk["az"].to_numpy(),
            chunk["gx"].to_numpy(), chunk["gy"].to_numpy(), chunk["gz"].to_numpy(),
        )
        X.append(xf)
        y.append(chunk["label"].mode().iloc[0])
        groups.append(chunk["subject"].mode().iloc[0])
    return (
        np.vstack(X).astype(np.float32),
        np.array(y, dtype=object),
        np.array(groups, dtype=object),
    )


def process_file(path: str, win_sec: float, overlap: float):
    """Read one CSV file and return feature windows, labels, and group IDs."""
    try:
        df = read_data(path)
    except Exception as e:
        print(f"  Skipping {path}: {e}")
        return (
            np.empty((0, 30), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )
    if len(df) < 5:
        return (
            np.empty((0, 30), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )
    t = df["time_s"].to_numpy()
    dt = np.diff(t)
    if dt.size == 0:
        return (
            np.empty((0, 30), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )
    med_dt = float(np.median(dt))
    fs_hz = 1.0 / med_dt if med_dt > 0 else 50.0
    win = int(round(win_sec * fs_hz))
    if win < 1:
        win = 1
    hop = int(round(win * (1.0 - overlap)))
    if hop < 1:
        hop = 1
    return make_windows(df, win=win, hop=hop)


def load_dataset(data_dir: Path, win_sec: float, overlap: float):
    """Load all CSV files from data_dir, extract features, and encode labels as integers."""
    data_files = sorted(
        [str(p) for p in data_dir.rglob("*.csv")]
        + [str(p) for p in data_dir.rglob("*.CSV")]
    )
    if not data_files:
        raise RuntimeError(f"No CSV files found in {data_dir}")

    all_X, all_y, all_groups = [], [], []
    for fp in data_files:
        Xw, yw, gw = process_file(fp, win_sec, overlap)
        if Xw.shape[0] == 0:
            continue
        all_X.append(Xw)
        all_y.append(yw)
        all_groups.append(gw)

    if not all_X:
        raise RuntimeError("No windows were generated from any file.")

    X = np.vstack(all_X).astype(np.float32)
    y_str = np.concatenate(all_y).astype(str)
    groups = np.concatenate(all_groups)

    classes_present = [c for c in GESTURE_LABEL_ORDER if c in set(y_str)]
    label_to_id = {c: i for i, c in enumerate(classes_present)}
    y_enc = np.array([label_to_id[s] for s in y_str], dtype=np.int64)
    return X, y_enc, groups, classes_present


# Evaluation helpers

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, outdir: Path, show: bool):
    """Save a confusion matrix heatmap as a PNG file."""
    import matplotlib.pyplot as plt

    labels_idx = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, ax=ax)
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


def evaluate_classifier(clf, X_train, X_test, y_train, y_test,
                        model_name, class_names, outdir, show):
    """Train a classifier, evaluate on the test set, and print the classification report."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, class_names, model_name, outdir, show)
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(
        y_test, y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    ))
    return clf, acc


def plot_accuracy_bar(results: dict, title: str, outdir: Path, show: bool):
    """Save a bar chart comparing accuracy across all classifiers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdae6b"]
    ax.bar(results.keys(), results.values(), color=colors[: len(results)])
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(results.values()):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center")
    fig.tight_layout()
    out_path = outdir / "accuracy_bar.png"
    fig.savefig(out_path, dpi=120)
    print(f"  Saved accuracy bar chart -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# C header export

def make_c_array(name: str, arr, dim=None) -> str:
    """Format a float array (1D or 2D) as a C static initializer."""
    if dim:
        txt = f"static const float {name}[{dim[0]}][{dim[1]}] = {{\n"
        arr_2d = np.asarray(arr).reshape(dim[0], dim[1])
        for row in arr_2d:
            txt += "  {" + ", ".join([f"{float(v):.6e}f" for v in row]) + "},\n"
    else:
        flat = np.asarray(arr).ravel()
        txt = f"static const float {name}[{len(flat)}] = {{\n"
        for v in flat:
            if np.asarray(v).dtype.kind in "iu":
                txt += f"  {int(v)},\n"
            else:
                txt += f"  {float(v):.6e}f,\n"
    txt += "};\n"
    return txt


def make_c_int_array(name: str, arr) -> str:
    """Format an int array as a C static initializer."""
    txt = f"static const int {name}[{len(arr)}] = {{\n"
    for v in arr:
        txt += f"  {int(v)},\n"
    txt += "};\n"
    return txt


def export_gesture_model_header(scaler, lr_clf, svm_clf, dt_clf, class_names, n_features, out_path: Path):
    """Export Logistic Regression, Linear SVM, and Decision Tree to a C header file."""
    N_FEATURES = int(n_features)
    N_CLASSES = len(class_names)
    lr_w = lr_clf.coef_.astype(np.float32).ravel()
    lr_b = lr_clf.intercept_.astype(np.float32).ravel()
    svm_w = svm_clf.coef_.astype(np.float32).ravel()
    svm_b = svm_clf.intercept_.astype(np.float32).ravel()
    means = scaler.mean_.astype(np.float32)
    scales = (1.0 / (scaler.scale_ + 1e-6)).astype(np.float32)

    tree = dt_clf.tree_
    N_NODES = int(tree.node_count)
    dt_children_left = tree.children_left.astype(np.int32)
    dt_children_right = tree.children_right.astype(np.int32)
    dt_feature = tree.feature.astype(np.int32)
    dt_threshold = tree.threshold.astype(np.float32)
    dt_value = tree.value.astype(np.float32).ravel()

    class_name_lines = ",\n    ".join(json.dumps(str(n)) for n in class_names)

    header = f"""#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <math.h>


#define N_FEATURES {N_FEATURES}
#define N_CLASSES {N_CLASSES}

static const char *CLASS_NAMES[N_CLASSES] = {{
    {class_name_lines}
}};

{make_c_array("MEAN", means)}
{make_c_array("SCALES", scales)}

static inline uint8_t find_argmax_lin(const float *f, const float *weights, const float *biases) {{
    float max_score = -INFINITY;
    uint8_t pred_class = 0;
    for (int k = 0; k < N_CLASSES; k++) {{
        float score = biases[k];
        for (int i = 0; i < N_FEATURES; i++) {{
            float z = (f[i] - MEAN[i]) * SCALES[i];
            score += weights[k * N_FEATURES + i] * z;
        }}
        if (score > max_score) {{ max_score = score; pred_class = (uint8_t)k; }}
    }}
    return pred_class;
}}

{make_c_array("LR_W", lr_w)}
{make_c_array("LR_B", lr_b)}

static inline uint8_t predict_lr(const float *f) {{
    return find_argmax_lin(f, LR_W, LR_B);
}}

{make_c_array("SVM_W", svm_w)}
{make_c_array("SVM_B", svm_b)}

static inline uint8_t predict_svm(const float *f) {{
    return find_argmax_lin(f, SVM_W, SVM_B);
}}

#define DT_N_NODES {N_NODES}
{make_c_int_array("DT_CHILDREN_LEFT", dt_children_left)}
{make_c_int_array("DT_CHILDREN_RIGHT", dt_children_right)}
{make_c_int_array("DT_FEATURE", dt_feature)}
{make_c_array("DT_THRESHOLD", dt_threshold)}
{make_c_array("DT_VALUE", dt_value, dim=[N_NODES, N_CLASSES])}

static inline uint8_t predict_dt(const float *f) {{
    int node_id = 0;
    float f_scaled[N_FEATURES];
    for (int i = 0; i < N_FEATURES; i++) {{
        f_scaled[i] = (f[i] - MEAN[i]) * SCALES[i];
    }}
    while (DT_CHILDREN_LEFT[node_id] != -1) {{
        int feature = DT_FEATURE[node_id];
        if (f_scaled[feature] <= DT_THRESHOLD[node_id])
            node_id = DT_CHILDREN_LEFT[node_id];
        else
            node_id = DT_CHILDREN_RIGHT[node_id];
    }}
    float max_score = -INFINITY; uint8_t pred_class = 0;
    for (int k = 0; k < N_CLASSES; k++) {{
        float score = DT_VALUE[node_id][k];
        if (score > max_score) {{ max_score = score; pred_class = (uint8_t)k; }}
    }}
    return pred_class;
}}

#endif // MODEL_H
"""
    out_path.write_text(header, encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")


def export_gesture_rf_model_header(scaler, rf_clf, class_names, n_features, out_path: Path):
    """Export a Random Forest to a C header file."""
    N_FEATURES = int(n_features)
    N_CLASSES = len(class_names)
    means = scaler.mean_.astype(np.float32)
    scales = (1.0 / (scaler.scale_ + 1e-6)).astype(np.float32)

    rf_all_cl, rf_all_cr, rf_all_feat, rf_all_thr = [], [], [], []
    rf_val_rows, rf_tree_root = [], []
    rf_off = 0
    for est in rf_clf.estimators_:
        tr = est.tree_
        n = int(tr.node_count)
        rf_tree_root.append(rf_off)
        cl = tr.children_left.astype(np.int64).copy()
        cr = tr.children_right.astype(np.int64).copy()
        for i in range(n):
            if cl[i] >= 0:
                cl[i] += rf_off
            if cr[i] >= 0:
                cr[i] += rf_off
        rf_all_cl.extend(cl.astype(np.int32))
        rf_all_cr.extend(cr.astype(np.int32))
        rf_all_feat.extend(tr.feature.astype(np.int32))
        rf_all_thr.extend(tr.threshold.astype(np.float32))
        rf_val_rows.append(tr.value[:, 0, :].astype(np.float32))
        rf_off += n

    N_RF_TREES = len(rf_tree_root)
    N_RF_NODES = rf_off
    rf_children_left = np.array(rf_all_cl, dtype=np.int32)
    rf_children_right = np.array(rf_all_cr, dtype=np.int32)
    rf_feature = np.array(rf_all_feat, dtype=np.int32)
    rf_threshold = np.array(rf_all_thr, dtype=np.float32)
    rf_value = np.vstack(rf_val_rows).astype(np.float32).ravel()
    rf_tree_root_arr = np.array(rf_tree_root, dtype=np.int32)

    class_name_lines = ",\n    ".join(json.dumps(str(n)) for n in class_names)

    header = f"""#ifndef RF_MODEL_H
#define RF_MODEL_H

#include <stdint.h>
#include <math.h>

#define RF_N_FEATURES {N_FEATURES}
#define RF_N_CLASSES {N_CLASSES}

static const char *RF_CLASS_NAMES[RF_N_CLASSES] = {{
    {class_name_lines}
}};

{make_c_array("RF_MEAN", means)}
{make_c_array("RF_SCALES", scales)}

#define RF_N_TREES {N_RF_TREES}
#define RF_N_NODES {N_RF_NODES}
{make_c_int_array("RF_TREE_ROOT", rf_tree_root_arr)}
{make_c_int_array("RF_CHILDREN_LEFT", rf_children_left)}
{make_c_int_array("RF_CHILDREN_RIGHT", rf_children_right)}
{make_c_int_array("RF_FEATURE", rf_feature)}
{make_c_array("RF_THRESHOLD", rf_threshold)}
{make_c_array("RF_VALUE", rf_value, dim=[N_RF_NODES, N_CLASSES])}

static inline uint8_t rf_tree_predict_at(const float *f_scaled, int node_id) {{
    while (RF_CHILDREN_LEFT[node_id] != -1) {{
        int feat = RF_FEATURE[node_id];
        if (f_scaled[feat] <= RF_THRESHOLD[node_id])
            node_id = RF_CHILDREN_LEFT[node_id];
        else
            node_id = RF_CHILDREN_RIGHT[node_id];
    }}
    uint8_t pred_class = 0; float max_score = -INFINITY;
    for (int k = 0; k < RF_N_CLASSES; k++) {{
        float score = RF_VALUE[node_id][k];
        if (score > max_score) {{ max_score = score; pred_class = (uint8_t)k; }}
    }}
    return pred_class;
}}

static inline uint8_t predict_rf(const float *f) {{
    float f_scaled[RF_N_FEATURES];
    for (int i = 0; i < RF_N_FEATURES; i++)
        f_scaled[i] = (f[i] - RF_MEAN[i]) * RF_SCALES[i];
    int votes[RF_N_CLASSES] = {{0}};
    for (int t = 0; t < RF_N_TREES; t++)
        votes[rf_tree_predict_at(f_scaled, RF_TREE_ROOT[t])]++;
    uint8_t best = 0; int best_v = votes[0];
    for (int k = 1; k < RF_N_CLASSES; k++)
        if (votes[k] > best_v) {{ best_v = votes[k]; best = (uint8_t)k; }}
    return best;
}}

#endif // RF_MODEL_H
"""
    out_path.write_text(header, encoding="utf-8")
    print(f"Wrote {out_path.resolve()}  (RF: {N_RF_TREES} trees, {N_RF_NODES} nodes)")


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train classical classifiers for gesture recognition.")
    p.add_argument("--win-sec", type=float, default=2.0, help="Window length in seconds (default: 2.0)")
    p.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction (default: 0.5)")
    p.add_argument("--dt-max-depth", type=int, default=-1,
                   help="DecisionTree max_depth; -1 means unlimited.")
    p.add_argument("--rf-n-estimators", type=int, default=120,
                   help="RandomForest n_estimators.")
    p.add_argument("--rf-max-depth", type=int, default=-1,
                   help="RandomForest max_depth; -1 means unlimited.")
    p.add_argument("--no-cv", action="store_true",
                   help="Use a single hold-out split instead of GroupKFold CV")
    p.add_argument("--cv-splits", type=int, default=5, help="Number of GroupKFold folds (default: 5)")
    p.add_argument("--test-size", type=float, default=0.3,
                   help="Test fraction for hold-out split (--no-cv only, default: 0.3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-header", action="store_true",
                   help="Export C headers")
    p.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    return p.parse_args()


def main():
    """Load data, run cross-validation or hold-out evaluation, and optionally export C headers."""
    args = parse_args()

    if not args.show_plots:
        matplotlib.use("Agg")

    outdir = Path("runs/gesture")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data")
    X, y_enc, groups, class_names = load_dataset(_DEFAULT_DATA, args.win_sec, args.overlap)
    N_FEATURES = X.shape[1]

    CLASSIFIER_FACTORIES = [
        ("Logistic Regression",            lambda: LogisticRegression(max_iter=1000, solver="lbfgs")),
        ("Support Vector Machine (Linear)", lambda: LinearSVC(multi_class="ovr", C=1.0, random_state=args.seed)),
        ("Decision Tree",                  lambda: DecisionTreeClassifier(
            random_state=args.seed,
            max_depth=None if args.dt_max_depth == -1 else args.dt_max_depth,
        )),
        ("K-Nearest Neighbors (k=3)",      lambda: KNeighborsClassifier(n_neighbors=3)),
        ("Random Forest",                  lambda: RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            max_depth=None if args.rf_max_depth == -1 else args.rf_max_depth,
        )),
    ]

    results = {}

    if not args.no_cv:
        gkf = GroupKFold(n_splits=args.cv_splits)
        cv_scores = {name: [] for name, _ in CLASSIFIER_FACTORIES}
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y_enc, groups=groups)):
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X[tr_idx])
            X_te_s = sc.transform(X[te_idx])
            y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]
            for name, factory in CLASSIFIER_FACTORIES:
                clf = factory()
                clf.fit(X_tr_s, y_tr)
                cv_scores[name].append(accuracy_score(y_te, clf.predict(X_te_s)))
            print(f"  Fold {fold + 1}/{args.cv_splits} done")

        print("\n=== GroupKFold CV (mean ± std accuracy) ===")
        for name, scores in cv_scores.items():
            a = np.asarray(scores, dtype=float)
            print(f"  {name}: {a.mean():.4f} ± {a.std():.4f}")
        results = {name: float(np.mean(cv_scores[name])) for name in cv_scores}

    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X[train_idx])
        X_test_s = sc.transform(X[test_idx])
        y_train_enc, y_test_enc = y_enc[train_idx], y_enc[test_idx]

        trained_models = {}
        for name, factory in CLASSIFIER_FACTORIES:
            clf = factory()
            trained_clf, acc = evaluate_classifier(
                clf, X_train_s, X_test_s, y_train_enc, y_test_enc,
                name, class_names, outdir, args.show_plots,
            )
            results[name] = acc
            trained_models[name] = trained_clf

        print("\n=== Summary (hold-out) ===")
        for k, v in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k}: {v * 100:.2f}%")

    plot_accuracy_bar(results, "Gesture Recognition — Algorithm Comparison", outdir, args.show_plots)

    if args.export_header:
        sc_all = StandardScaler()
        X_all_s = sc_all.fit_transform(X)

        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(X_all_s, y_enc)
        svm = LinearSVC(multi_class="ovr", C=1.0, random_state=args.seed)
        svm.fit(X_all_s, y_enc)
        dt = DecisionTreeClassifier(
            random_state=args.seed,
            max_depth=None if args.dt_max_depth == -1 else args.dt_max_depth,
        )
        dt.fit(X_all_s, y_enc)
        rf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            max_depth=None if args.rf_max_depth == -1 else args.rf_max_depth,
        )
        rf.fit(X_all_s, y_enc)

        export_gesture_model_header(sc_all, lr, svm, dt, class_names, N_FEATURES,
                                  outdir / "model.h")
        export_gesture_rf_model_header(sc_all, rf, class_names, N_FEATURES,
                                     outdir / "rf_model.h")


if __name__ == "__main__":
    main()
