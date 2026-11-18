# File:        train.py
# Author:      Daria Kinash
# Description: Main training script for gesture and UCI HAR recognition.
#              Loads data, builds the CNN model, trains it, evaluates on the test set,
#              and optionally exports to TensorFlow Lite for embedded deployment.

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from datasets import (
    DEFAULT_CHANNELS,
    DataConfig,
    GESTURE_LABEL_ORDER,
    UCI_LABEL_ORDER,
    _make_train_augment_fn,
    prepare_data,
)
from models import CNN1DConfig, build_model

_SCRIPT_DIR = Path(__file__).resolve().parent
_GESTURE_DATA = (_SCRIPT_DIR / ".." / "data" / "hands_labeled").resolve()
_UCI_DATA   = (_SCRIPT_DIR / ".." / "data" / "UCI HAR Dataset").resolve()

_DEFAULT_WINDOW = 150
_DEFAULT_STRIDE = 75


def export_tflite(model, out_path, quant, representative_data=None):
    """Convert a Keras model to TFLite and save it to out_path."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quant == "float32":
        pass
    elif quant == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "int8":
        if representative_data is None:
            raise ValueError("For int8 needs representative_data.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        raise ValueError("quant must be float32, dynamic, or int8")

    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)


def estimate_tflite_ram(tflite_path):
    """Estimate the total RAM used by all tensors in a TFLite model (in bytes)."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    dtype_sizes = {
        np.float32: 4,
        np.float64: 8,
        np.int8: 1,
        np.uint8: 1,
        np.int16: 2,
        np.uint16: 2,
        np.int32: 4,
        np.uint32: 4,
    }

    total_bytes = 0
    for d in details:
        shape = d.get("shape", None)
        if shape is None or -1 in shape:
            continue
        size = int(np.prod(shape))
        dtype = d.get("dtype")
        b = dtype_sizes.get(dtype, None)
        if b is None:
            continue
        total_bytes += size * b

    return total_bytes


def write_normalization_params_h(out_path, mean, std):
    """Write per-channel normalization parameters to a C header file for firmware use."""
    mean = np.asarray(mean).ravel()
    std = np.asarray(std).ravel()
    n = len(mean)
    if n != len(std):
        raise ValueError("mean and std length mismatch")
    mean_str = ", ".join(f"{float(x):.6f}f" for x in mean)
    std_str = ", ".join(f"{float(x):.6f}f" for x in std)
    lines = [
        "",
        "#ifndef NORMALIZATION_PARAMS_H",
        "#define NORMALIZATION_PARAMS_H",
        "",
        f"#define NORM_CHANNELS {n}",
        "",
        f"static const float norm_mean[NORM_CHANNELS] = {{ {mean_str} }};",
        f"static const float norm_std[NORM_CHANNELS] = {{ {std_str} }};",
        "",
        "#endif",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("Normalization header:", out_path.resolve())


def append_results_csv(csv_path, row):
    """Append one result row to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = sorted(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_results_row(args, history, test_dict, tflite_size, tflite_ram_bytes, run_name, cnn_cfg, num_classes, window_length):
    """Build a dictionary of training configuration and results for logging to CSV."""
    final_train_acc = None
    final_val_acc = None
    if "accuracy" in history.history and history.history["accuracy"]:
        final_train_acc = float(history.history["accuracy"][-1])
    if "val_accuracy" in history.history and history.history["val_accuracy"]:
        final_val_acc = float(history.history["val_accuracy"][-1])

    row = {
        "window": window_length,
        "stride": args.stride,
        "label_mode": args.label_mode,
        "channels": ",".join(args.channels),
        "num_classes": num_classes,
        "epochs": len(history.history.get("loss", [])),
        "batch": args.batch,
        "lr": args.lr,
        "cnn_filters": ",".join(map(str, args.cnn_filters)),
        "cnn_kernels": ",".join(map(str, args.cnn_kernels)),
        "cnn_pools": ",".join(map(str, args.cnn_pools)),
        "cnn_dropout": args.cnn_dropout,
        "cnn_dense": args.cnn_dense,
        "cnn_bn": int(args.cnn_bn),
    }

    if final_train_acc is not None:
        row["final_train_accuracy"] = final_train_acc
    if final_val_acc is not None:
        row["final_val_accuracy"] = final_val_acc
    if "test_accuracy" in test_dict:
        row["final_test_accuracy"] = test_dict["test_accuracy"]
    elif "test_compile_metrics" in test_dict:
        row["final_test_accuracy"] = test_dict["test_compile_metrics"]

    if args.export_tflite:
        row["quant"] = args.quant
        row["tflite_bytes"] = int(tflite_size)
        if tflite_ram_bytes is not None:
            row["tflite_ram_bytes"] = int(tflite_ram_bytes)
    else:
        row["quant"] = ""
        row["tflite_bytes"] = ""

    return row


def compute_extended_test_metrics(y_true, y_proba, num_classes, label_names=None):
    """Compute precision, recall, F1 (macro and weighted), ROC AUC, and per-class ROC curves."""
    try:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
        from sklearn.preprocessing import label_binarize
    except ImportError:
        print("Install scikit-learn for precision/recall/F1/ROC: pip install scikit-learn")
        return {}, None

    y_true = np.asarray(y_true, dtype=np.int64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    y_pred = np.argmax(y_proba, axis=1)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=list(range(num_classes))
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=list(range(num_classes))
    )
    p_pc, r_pc, f_pc, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes))
    )

    try:
        roc_macro = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        roc_macro = float("nan")
    try:
        roc_weighted = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    except Exception:
        roc_weighted = float("nan")

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    roc_curves = []
    for c in range(num_classes):
        pos = y_bin[:, c].sum()
        neg = (1 - y_bin[:, c]).sum()
        if pos == 0 or neg == 0:
            roc_curves.append({"class": c, "fpr": [], "tpr": [], "thresholds": []})
            continue
        fpr, tpr, thresholds = roc_curve(y_bin[:, c], y_proba[:, c])
        roc_curves.append(
            {
                "class": c,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
        )

    row = {
        "test_precision_macro": float(p_macro),
        "test_recall_macro": float(r_macro),
        "test_f1_macro": float(f_macro),
        "test_precision_weighted": float(p_w),
        "test_recall_weighted": float(r_w),
        "test_f1_weighted": float(f_w),
        "test_roc_auc_ovr_macro": roc_macro if not np.isnan(roc_macro) else None,
        "test_roc_auc_ovr_weighted": roc_weighted if not np.isnan(roc_weighted) else None,
    }
    plot_data = {"roc_curves": roc_curves, "p_pc": p_pc, "r_pc": r_pc, "f_pc": f_pc, "support": support}
    return row, plot_data


def save_metrics_plots(outdir, cm, plot_data, num_classes, label_names=None):
    """Save confusion matrix, ROC curves, and per-class precision/recall/F1 as PNG files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for metric plots: pip install matplotlib")
        return

    outdir = Path(outdir)
    names = list(label_names) if label_names is not None else [str(i) for i in range(num_classes)]
    if len(names) < num_classes:
        names = names + [str(i) for i in range(len(names), num_classes)]

    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(names[:num_classes], rotation=45, ha="right")
    ax.set_yticklabels(names[:num_classes])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (normalized by row)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{:.0f}\n({:.1f}%)".format(cm[i, j], 100 * cm_norm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    roc_curves = plot_data["roc_curves"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(num_classes):
        rc = roc_curves[c] if c < len(roc_curves) else {"fpr": [], "tpr": []}
        fpr, tpr = rc.get("fpr", []), rc.get("tpr", [])
        if len(fpr) > 0:
            ax.plot(fpr, tpr, lw=2, color="C" + str(c % 10), label=names[c])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "roc_curves.png", dpi=150)
    plt.close(fig)

    p_pc = plot_data["p_pc"]
    r_pc = plot_data["r_pc"]
    f_pc = plot_data["f_pc"]
    x = np.arange(num_classes)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 1.2), 5))
    ax.bar(x - w, p_pc, w, label="Precision")
    ax.bar(x, r_pc, w, label="Recall")
    ax.bar(x + w, f_pc, w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names[:num_classes], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Per-class precision, recall, F1 (test)")
    fig.tight_layout()
    fig.savefig(outdir / "per_class_metrics.png", dpi=150)
    plt.close(fig)

    print("Saved figures:", outdir / "confusion_matrix.png", outdir / "roc_curves.png", outdir / "per_class_metrics.png")


def parse_args():
    """Parse command-line arguments for training configuration."""
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data", type=str, default=None,
                   help="Path to data folder (default: hands_labeled or UCI HAR Dataset).")
    p.add_argument("--uci", action="store_true",
                   help="Use UCI HAR dataset (6 classes) instead of gesture CSV files.")
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS,
                   help="CSV column names to use as input channels.")
    p.add_argument("--window", type=int, default=_DEFAULT_WINDOW,
                   help="Window length in samples (gesture only).")
    p.add_argument("--stride", type=int, default=_DEFAULT_STRIDE,
                   help="Step between windows in samples (gesture only).")
    p.add_argument("--label-mode", type=str, default="last", choices=["last", "mode"],
                   help="Label assignment per window: last sample or majority vote.")
    p.add_argument("--random-split", action="store_true",
                   help="Gesture only: split randomly instead of by subject. Ignored for --uci.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable z-score normalization.")
    p.add_argument("--remove-gravity", action="store_true",
                   help="UCI only: remove gravity component from body_acc channels.")
    p.add_argument("--sample-rate", type=float, default=50.0,
                   help="UCI only: sensor sampling rate in Hz.")
    p.add_argument("--gravity-cutoff-hz", type=float, default=0.5,
                   help="UCI only: high-pass cutoff frequency for gravity removal.")

    # augmentation
    p.add_argument("--augment", action="store_true",
                   help="Add Gaussian noise and random amplitude scaling to training windows.")
    p.add_argument("--aug-noise-std", type=float, default=0.03,
                   help="Std of Gaussian noise added during augmentation.")
    p.add_argument("--aug-scale-min", type=float, default=0.92,
                   help="Min random amplitude scale factor.")
    p.add_argument("--aug-scale-max", type=float, default=1.08,
                   help="Max random amplitude scale factor.")

    # split 
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Fraction of subjects used for training.")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Fraction of subjects used for validation.")
    p.add_argument("--test-subjects", nargs="+", default=None,
                   help="Subject IDs reserved only for the test set.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")

    # model
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate.")
    p.add_argument("--epochs", type=int, default=50,
                   help="Max training epochs (early stopping may end sooner).")
    p.add_argument("--batch", type=int, default=32,
                   help="Batch size.")
    p.add_argument("--cnn-filters", nargs="+", type=int, default=[16, 32],
                   help="Filters per conv layer, e.g. 16 32.")
    p.add_argument("--cnn-kernels", nargs="+", type=int, default=[5, 3],
                   help="Kernel size per conv layer, e.g. 5 3.")
    p.add_argument("--cnn-pools", nargs="+", type=int, default=[2],
                   help="MaxPool size per layer, e.g. 2 2.")
    p.add_argument("--cnn-dropout", type=float, default=0.2,
                   help="Dropout rate after each block.")
    p.add_argument("--cnn-dense", type=int, default=32,
                   help="Units in the dense layer before output.")
    p.add_argument("--cnn-bn", action="store_true",
                   help="Add BatchNormalization after each conv layer.")
    p.add_argument("--cnn-l2", type=float, default=1e-3,
                   help="L2 regularization coefficient.")

    # output
    p.add_argument("--outdir", type=str, default="runs",
                   help="Root directory for all run outputs.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Name for this run (default: timestamp).")
    p.add_argument("--results-csv", type=str, default="results/results.csv",
                   help="CSV file to append results after each run.")
    p.add_argument("--export-tflite", action="store_true",
                   help="Convert model to TFLite after training.")
    p.add_argument("--quant", type=str, default="float32", choices=["float32", "dynamic", "int8"],
                   help="TFLite quantization mode.")

    return p.parse_args()


def main():
    """Run the full training pipeline: load data, build model, train, evaluate, and export."""
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # select dataset and label set 
    if args.uci:
        data_root   = Path(args.data) if args.data else _UCI_DATA
        label_order = UCI_LABEL_ORDER
        window_size = 128          # UCI HAR windows are fixed at 128 samples
    else:
        data_root   = Path(args.data) if args.data else _GESTURE_DATA
        label_order = GESTURE_LABEL_ORDER
        window_size = args.window

    num_classes = len(label_order)

    if args.aug_scale_min > args.aug_scale_max:
        raise ValueError("--aug-scale-min must be <= --aug-scale-max")

    # load and prepare data (windowing, split, normalization)
    aug_fn = _make_train_augment_fn(args.aug_noise_std, args.aug_scale_min, args.aug_scale_max) if args.augment else None

    cfg = DataConfig(
        channels=args.channels,
        label_order=label_order,
        window_size=window_size,
        stride=args.stride,
        window_label_mode=args.label_mode,
        normalize=not args.no_normalize,
        split_by_subject=not args.random_split if not args.uci else True,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_subjects=tuple(args.test_subjects) if args.test_subjects else None,
        seed=args.seed,
        use_uci=args.uci,
        uci_remove_gravity=args.remove_gravity,
        uci_sample_rate_hz=float(args.sample_rate),
        uci_gravity_hp_cutoff_hz=float(args.gravity_cutoff_hz),
    )
    data = prepare_data(data_root, cfg)

    window_length = data.X_train.shape[1]
    n_channels    = data.X_train.shape[2]
    input_shape   = (window_length, n_channels)

    # build model
    cnn_cfg = CNN1DConfig(
        filters=tuple(args.cnn_filters),
        kernel_sizes=tuple(args.cnn_kernels),
        pool_sizes=tuple(args.cnn_pools),
        dropout=args.cnn_dropout,
        dense_units=args.cnn_dense,
        use_batchnorm=args.cnn_bn,
        l2=args.cnn_l2,
    )

    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=args.lr,
        cnn_cfg=cnn_cfg,
    )

    # callbacks: save best checkpoint, early stop, log history 
    ckpt_path = outdir / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(outdir / "history.csv")),
    ]

    # train
    history = model.fit(
        data.train_ds(args.batch, augment_fn=aug_fn),
        validation_data=data.val_ds(args.batch),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # evaluate on the test set
    test_ds = data.test_ds(args.batch)
    test_metrics = model.evaluate(test_ds, verbose=0)
    metric_names = model.metrics_names
    test_dict = {}
    for name, value in zip(metric_names, test_metrics):
        test_dict["test_" + name] = float(value)
        print(f"Test {name}: {value:.4f}")

    y_true = []
    y_proba_list = []
    for batch_x, batch_y in test_ds:
        logits = model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy().tolist())
        y_proba_list.append(np.asarray(logits, dtype=np.float32))

    # compute and save evaluation plots
    extra_metrics = {}
    plot_data = None
    if y_true:
        y_proba = np.concatenate(y_proba_list, axis=0)
        y_pred = np.argmax(y_proba, axis=1)
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
        print("\nConfusion matrix on TEST set (rows = true, cols = predicted):")
        print(cm)
        extra_metrics, plot_data = compute_extended_test_metrics(
            y_true, y_proba, num_classes, label_names=label_order
        )
        if plot_data is not None:
            save_metrics_plots(outdir, cm, plot_data, num_classes, label_order)

    # save final model
    model.save(outdir / "final.keras")

    tflite_size = None
    tflite_ram_bytes = None
    if args.export_tflite:
        tflite_path = outdir / "model.tflite"
        rep = data.representative_dataset if args.quant == "int8" else None
        export_tflite(model, tflite_path, quant=args.quant, representative_data=rep)
        tflite_path = tflite_path.resolve()
        tflite_size = tflite_path.stat().st_size
        print("TFLite saved:", tflite_path)
        try:
            tflite_ram_bytes = estimate_tflite_ram(tflite_path)
        except Exception as e:
            print("Could not estimate TFLite RAM usage:", e)
            tflite_ram_bytes = None


    run_row = build_results_row(
        args=args,
        history=history,
        test_dict=test_dict,
        tflite_size=tflite_size,
        tflite_ram_bytes=tflite_ram_bytes,
        run_name=run_name,
        cnn_cfg=cnn_cfg,
        num_classes=num_classes,
        window_length=window_length,
    )
    run_row.update(extra_metrics)
    append_results_csv(Path(args.results_csv), run_row)

    if data.mean is not None and data.std is not None:
        write_normalization_params_h(outdir / "normalization_params.h", data.mean, data.std)

    print("\nDone.")
    print("Run dir:", outdir)
    if args.export_tflite:
        print("TFLite:", tflite_path.resolve(), "(" + str(tflite_size) + " bytes)")
        if tflite_ram_bytes is not None:
            print("Approximate TFLite RAM (tensors):", f"{tflite_ram_bytes / 1024:.1f} kB")


if __name__ == "__main__":
    main()
