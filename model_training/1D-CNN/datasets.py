# File:        datasets.py
# Author:      Daria Kinash
# Description: Data loading, windowing, normalization, splitting, and augmentation for the gesture and UCI HAR training pipelines.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


DEFAULT_CHANNELS = ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")

GESTURE_LABEL_ORDER = ("idle", "accept", "left", "right", "cancel")
UCI_LABEL_ORDER = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)

UCI_BODY_SIGNALS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
)


def _highpass_remove_gravity_1d(x, sample_rate_hz, cutoff_hz):
    """Remove the gravity from one signal channel using a moving-average high-pass filter."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return x.astype(np.float32)
    fc = max(float(cutoff_hz), 0.05)
    fs = max(float(sample_rate_hz), 1.0)
    w = max(3, int(round(fs / fc)))
    if w % 2 == 0:
        w += 1
    w = min(w, n - 2)
    if w < 3:
        return x.astype(np.float32)
    if w % 2 == 0:
        w -= 1
    k = np.ones(w, dtype=np.float64) / w
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    low = np.convolve(xp, k, mode="valid")
    if low.shape[0] != x.shape[0]:
        low = low[: x.shape[0]]
    return (x - low).astype(np.float32)


def remove_gravity_uci_body_acc(X, sample_rate_hz, cutoff_hz):
    """Apply gravity removal to the first three channels (body_acc x/y/z) of every window."""
    X = np.asarray(X, dtype=np.float32).copy()
    for i in range(X.shape[0]):
        for j in range(3):
            X[i, :, j] = _highpass_remove_gravity_1d(X[i, :, j], sample_rate_hz, cutoff_hz)
    return X


@dataclass
class DataConfig:
    """All parameters that control data loading and preprocessing."""
    channels: tuple | list                  # CSV column names to use as input channels
    label_order: tuple = GESTURE_LABEL_ORDER  # defines integer index for each class name
    window_size: int = 128                  # window length in samples
    split_by_subject: bool = True           # True = subject-based split, False = random
    stride: int = 64                        # step between consecutive windows
    window_label_mode: str = "last"         # how to assign a label to a window: 'last' or 'mode'
    normalize: bool = True                  # apply z-score normalization
    eps: float = 1e-6                       # small value to avoid division by zero in normalization
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_subjects: tuple | None = None      # subject IDs reserved for the test set
    seed: int = 42
    use_uci: bool = False
    uci_remove_gravity: bool = False
    uci_sample_rate_hz: float = 50.0
    uci_gravity_hp_cutoff_hz: float = 0.5


def _make_train_augment_fn(noise_std, scale_min, scale_max):
    """Return a tf.data map function that applies noise and amplitude scaling to a window."""
    def aug_xy(x, y):
        x = tf.cast(x, tf.float32)
        if noise_std > 0:
            x = x + tf.random.normal(tf.shape(x), stddev=noise_std)
        if scale_max - scale_min > 1e-6:
            s = tf.random.uniform((), scale_min, scale_max)
            x = x * s
        return x, y
    return aug_xy


def _list_csv_files(root):
    """Return a sorted list of all CSV files under root."""
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".csv":
        return [root]
    result = []
    seen = set()
    for p in root.rglob("*.csv"):
        if p.is_file() and p.name not in seen:
            seen.add(p.name)
            result.append(p)
    for p in root.rglob("*.CSV"):
        if p.is_file() and p.name not in seen:
            seen.add(p.name)
            result.append(p)
    return sorted(result, key=lambda p: p.name)


def _read_csv(path):
    """Read a CSV file and strip whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _window_label(labels, mode):
    """Return a single integer label for a window."""
    if labels.size == 0:
        raise ValueError("Label array is empty.")
    if mode == "last":
        return int(labels[-1])
    if mode == "mode":
        vals, counts = np.unique(labels, return_counts=True)
        return int(vals[np.argmax(counts)])
    raise ValueError("window_label_mode must be 'last' or 'mode'.")


def _make_windows(data, labels, window_size, stride, label_mode):
    """Slice signal data into overlapping windows of shape (window_size, n_channels)."""
    n = data.shape[0]           
    n_channels = data.shape[1]  
    if n < window_size:
        empty_X = np.empty((0, window_size, n_channels), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64) if labels is not None else None
        return empty_X, empty_y

    starts = []
    pos = 0
    while pos + window_size <= n:
        starts.append(pos)
        pos += stride
    starts = np.array(starts) 

    X = np.zeros((len(starts), window_size, n_channels), dtype=np.float32) 
    for i, s in enumerate(starts):
        X[i] = data[s : s + window_size]

    if labels is None:
        return X, None

    # One label per window
    y = np.zeros(len(starts), dtype=np.int64)
    for i, s in enumerate(starts):
        window_labels = labels[s : s + window_size]
        y[i] = _window_label(window_labels, label_mode)
    return X, y


def compute_normalization_stats(X_train, eps=1e-6):
    """Compute per-channel mean and std from the training set."""
    if X_train.size == 0:
        raise ValueError("X_train is empty.")
    n_channels = X_train.shape[-1]
    flat = X_train.reshape(-1, n_channels)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = np.maximum(flat.std(axis=0).astype(np.float32), eps)
    return mean, std


def apply_normalization(X, mean, std):
    """Apply z-score normalization using pre-computed mean and std."""
    return (X - mean) / std


def _load_windows(root, cfg):
    """Load all CSV files from root, apply windowing, and return (X, y, subjects)."""
    order = list(cfg.label_order)
    name_to_int = {}
    for i, name in enumerate(order):
        name_to_int[name] = i
    files = _list_csv_files(root)
    all_X = []
    all_y = []
    all_subjects = []

    for f in files:
        df = _read_csv(f)
        if "label" not in df.columns:
            raise ValueError(f.name + ": missing column 'label'.")
        data = df[list(cfg.channels)].to_numpy(dtype=np.float32)
        raw_labels = df["label"].astype(str).str.strip().str.lower()
        labels_mapped = raw_labels.map(name_to_int)
        if labels_mapped.isna().any():
            raise ValueError("Unknown label in " + f.name)
        labels = labels_mapped.to_numpy(dtype=np.int64)
        
        Xw, yw = _make_windows(data, labels, cfg.window_size, cfg.stride, cfg.window_label_mode)

        if Xw.shape[0] == 0:
            continue

        subject = df["subject"].iloc[0] # take from first row
        all_X.append(Xw)
        all_y.append(yw)
        for _ in range(Xw.shape[0]):
            all_subjects.append(subject)

    if len(all_X) == 0:
        n_ch = len(cfg.channels)
        X = np.empty((0, cfg.window_size, n_ch), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        return X, y, []

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y, all_subjects


def split_train_val_test(X, y, seed=42, train_ratio=0.7, val_ratio=0.15):
    """Split windows randomly into train, validation, and test sets."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_by_subject(X, y, subjects, train_ratio=0.6, val_ratio=0.2, test_subjects=None):
    """Split windows by subject so no subject appears in more than one partition."""
    all_unique = sorted(set(subjects))

    test_subs = set(test_subjects or [])
    unique = [s for s in all_unique if s not in test_subs]

    n = len(unique)
    n_train = max(1, int(n * train_ratio))

    if test_subs:
        n_val = n - n_train
    else:
        n_val = max(0, int(n * val_ratio))
        n_leftover = n - n_train - n_val
        if n_leftover < 0:
            n_val += n_leftover  # trim val to avoid overlap
        else:
            test_subs = set(unique[n_train + n_val:])

    train_subs = set(unique[:n_train])
    val_subs = set(unique[n_train : n_train + n_val])

    train_idx = [i for i in range(len(subjects)) if subjects[i] in train_subs]
    val_idx = [i for i in range(len(subjects)) if subjects[i] in val_subs]
    test_idx = [i for i in range(len(subjects)) if subjects[i] in test_subs]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


@dataclass
class PreparedData:
    """All data arrays and parameters needed for training and evaluation."""
    X_train: np.ndarray
    y_train: np.ndarray | None
    X_val: np.ndarray
    y_val: np.ndarray | None
    X_test: np.ndarray
    y_test: np.ndarray | None
    mean: np.ndarray | None
    std: np.ndarray | None
    
    def train_ds(self, batch_size, shuffle_buffer=4096, seed=42, augment_fn=None):
        """Dataset for training, with optional shuffling and augmentation."""
        if self.y_train is not None:
            ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        else:
            ds = tf.data.Dataset.from_tensor_slices(self.X_train)
        ds = ds.shuffle(shuffle_buffer, seed=seed)
        if augment_fn is not None and self.y_train is not None:
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def val_ds(self, batch_size):
        """Dataset for validation."""
        if self.y_val is not None:
            ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        else:
            ds = tf.data.Dataset.from_tensor_slices(self.X_val)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def test_ds(self, batch_size):
        """Dataset for testing."""
        if self.y_test is not None:
            ds = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        else:
            ds = tf.data.Dataset.from_tensor_slices(self.X_test)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def representative_dataset(self):
        """Representative dataset for post-training quantization calibration. Returns a generator of batches."""
        X = self.X_train
        if X.shape[0] == 0:
            return
        n = min(200, X.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=n, replace=False)
        for i in idx:
            yield [X[i : i + 1].astype(np.float32)]


def _load_uci_har(root):
    """Load the UCI HAR dataset from root and return train and test arrays."""
    root = Path(root)
    train_ins = root / "train" 
    test_ins = root / "test" 
    if not train_ins.is_dir() or not test_ins.is_dir():
        raise FileNotFoundError("UCI HAR root must contain train and test directories")

    def load_part(part):
        X_list = []
        for name in UCI_BODY_SIGNALS:
            path = train_ins if part == "train" else test_ins
            path = path / f"{name}_{part}.txt"
            if not path.exists():
                raise FileNotFoundError(str(path))
            arr = np.loadtxt(path, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X_list.append(arr)
        X = np.stack(X_list, axis=-1)
        y_path = root / part / f"y_{part}.txt"
        y = np.loadtxt(y_path, dtype=np.int64).ravel()
        y = y - 1 
        subj_path = root / part / f"subject_{part}.txt"
        subjects = np.loadtxt(subj_path, dtype=object).ravel().tolist()
        return X, y, subjects

    X_train, y_train, subjects_train = load_part("train")
    X_test, y_test, subjects_test = load_part("test")
    return X_train, y_train, subjects_train, X_test, y_test, subjects_test


def prepare_data(root, cfg):
    """Load data, split, normalize, and return a PreparedData object ready for training."""
    test_subjects = list(cfg.test_subjects) if cfg.test_subjects else None
    if cfg.use_uci:
        X_train_all, y_train_all, subjects_train, X_test, y_test, subjects_test = _load_uci_har(root)
        if cfg.uci_remove_gravity:
            X_train_all = remove_gravity_uci_body_acc(
                X_train_all, cfg.uci_sample_rate_hz, cfg.uci_gravity_hp_cutoff_hz
            )
            X_test = remove_gravity_uci_body_acc(
                X_test, cfg.uci_sample_rate_hz, cfg.uci_gravity_hp_cutoff_hz
            )
        (X_train, y_train), (X_val, y_val), (_, _) = split_by_subject(
            X_train_all, y_train_all, subjects_train,
            train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, test_subjects=test_subjects,
        )
    else:
        X, y, subjects = _load_windows(root, cfg)
        if cfg.split_by_subject:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_subject(
                X, y, subjects,
                train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, test_subjects=test_subjects,
            )
        else:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(
                X, y, seed=cfg.seed, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
            )

    mean = None
    std = None
    if cfg.normalize and X_train.size > 0:
        mean, std = compute_normalization_stats(X_train, eps=cfg.eps)
        X_train = apply_normalization(X_train, mean, std)
        X_val = apply_normalization(X_val, mean, std)
        X_test = apply_normalization(X_test, mean, std)

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        mean=mean,
        std=std,
    )
