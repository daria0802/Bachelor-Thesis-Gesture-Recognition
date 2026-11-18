# Hand Gesture Recognition on NXP i.MX RT1052

This repository contains the pipeline developed for the bachelor thesis *Implementation of ML Outputs on Embedded Platforms* (FIT VUT, 2026). It covers training gesture recognition models on a PC and deploying them to an embedded microcontroller for real-time classification. Two model types are provided — a 1D convolutional neural network (via TensorFlow Lite Micro) and classical machine learning classifiers.

---

## Repository layout

```
bachelor-thesis/
├── documentation/                  LaTeX source and figures for the thesis
├── model_training/
│   ├── data/
│   │   ├── hands/                  raw gesture CSVs (no label column)
│   │   └── UCI HAR Dataset/        public UCI HAR dataset (download separately, see 3.1)
│   ├── 1D-CNN/
│   │   ├── clean_dataset.py        adds label/subject columns to raw CSVs
│   │   ├── datasets.py             windowing, normalization, data splits
│   │   ├── models.py               1D CNN architecture
│   │   └── train.py                training, evaluation, TFLite export
│   └── classical_classifiers/
│       ├── train_gesture.py        classical classifiers on gesture data
│       └── train_uci.py            classical classifiers on UCI HAR data
└── inference/
    ├── 1D_CNN_model/               MCUXpresso project – TFLite Micro inference
    └── classical_models/           MCUXpresso project – classical model inference
```

---

## 1. Hardware

### Board and sensor

| Component | Details |
|---|---|
| MCU | NXP MIMXRT1052 (IMXRT1050-EVKB) |
| IMU | MPU-6050 (6-axis: 3-axis accelerometer + 3-axis gyroscope) |
| Button | Onboard SW8 (GPIO5\_IO00) – starts/stops recognition |
| USB | Micro-USB J28 – power and debug/flash |

### Wiring the MPU-6050 to the IMXRT1050-EVKB

The sensor communicates over I2C1 (LPI2C1). Connect it as follows:

| MPU-6050 pin | EVKB connector pin | Note |
|---|---|---|
| VCC | 3.3 V (J23-4 or J17-1) | Do not connect to 5 V |
| GND | GND (J23-3 or J17-2) | Common ground |
| SDA | J22-9 | LPI2C1\_SDA (GPIO\_AD\_B1\_01) |
| SCL | J22-11 | LPI2C1\_SCL (GPIO\_AD\_B1\_00) |
| AD0 | GND | Sets I2C address to 0x68 |
| INT | J22-8 | Not used by firmware (polling mode) |

The firmware configures the sensor with:
- Accelerometer range: ±8 g
- Gyroscope range: ±500 °/s
- On-chip digital low-pass filter: 42 Hz
- Polling rate: 50 Hz (20 ms period via `vTaskDelayUntil`)

---

## 2. Requirements

Python 3.10 or newer is required. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

For firmware deployment, the following tools are required:

- **MCUXpresso IDE** 25.6 or compatible (used during development)
- **NXP SDK** for MIMXRT1050-EVKB (imported into the IDE)
- **LinkServer** debug probe driver (bundled with the IDE) for flashing over USB

---

## 3. Data preparation (gesture dataset)

Raw CSV files in `data/hands/` have no label or subject columns. Run `clean_dataset.py` once before training:

```bash
cd model_training/1D-CNN
python clean_dataset.py
```

This reads every CSV from `data/hands/`, extracts the class label and subject ID from the filename (e.g. `accept_d1.CSV` → label `accept`, subject `d1`), drops rows with missing values, and writes the result to `data/hands_labeled/`.

Optional arguments:

```
--input   path to raw CSV folder   (default: data/hands)
--output  path for labeled output   (default: data/hands_labeled)
```

### 3.1 UCI HAR dataset (optional)

The public UCI HAR dataset is not included in this repository because of its size. Download it from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
unzip it and place the `UCI HAR Dataset/` folder into `model_training/data/`.

---

## 4. Training

### 4.1 1D CNN

**Gesture dataset (5 classes: idle, accept, left, right, cancel)**

```bash
cd model_training/1D-CNN
python train.py --export-tflite --quant float32
```

**UCI HAR dataset (6 classes: walking, upstairs, downstairs, sitting, standing, laying)**

```bash
python train.py --uci --export-tflite --quant float32
```

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--data` | auto | Path to data folder (defaults to `hands_labeled` or `UCI HAR Dataset`) |
| `--uci` | off | Use UCI HAR dataset instead of gesture CSVs |
| `--channels` | ax_g ay_g az_g gx_dps gy_dps gz_dps | CSV column names to use as input channels |
| `--window` | 150 | Window length in samples (gesture only) |
| `--stride` | 75 | Step between windows in samples (gesture only) |
| `--label-mode` | last | Label per window: `last` sample or majority `mode` |
| `--random-split` | off | Split randomly instead of by subject (gesture only) |
| `--no-normalize` | off | Disable z-score normalization |
| `--remove-gravity` | off | Remove gravity from body_acc channels (UCI only) |
| `--sample-rate` | 50.0 | Sensor sampling rate in Hz (UCI only) |
| `--gravity-cutoff-hz` | 0.5 | High-pass cutoff for gravity removal in Hz (UCI only) |
| `--augment` | off | Gaussian noise + amplitude scaling on training set |
| `--aug-noise-std` | 0.03 | Std of Gaussian noise added during augmentation |
| `--aug-scale-min` | 0.92 | Min random amplitude scale factor |
| `--aug-scale-max` | 1.08 | Max random amplitude scale factor |
| `--train-ratio` | 0.8 | Fraction of subjects used for training |
| `--val-ratio` | 0.2 | Fraction of subjects used for validation |
| `--test-subjects` | — | Subject IDs reserved only for the test set |
| `--seed` | 42 | Random seed |
| `--lr` | 0.001 | Adam learning rate |
| `--epochs` | 50 | Maximum training epochs (early stopping applies) |
| `--batch` | 32 | Batch size |
| `--cnn-filters` | 16 32 | Filters per convolutional layer |
| `--cnn-kernels` | 5 3 | Kernel size per layer |
| `--cnn-pools` | 2 | MaxPool size per layer |
| `--cnn-dropout` | 0.2 | Dropout rate after each block |
| `--cnn-dense` | 32 | Units in the dense layer before output |
| `--cnn-bn` | off | Add BatchNormalization after each conv layer |
| `--cnn-l2` | 0.001 | L2 regularization coefficient |
| `--outdir` | runs | Root directory for all run outputs |
| `--run-name` | timestamp | Name for this run |
| `--results-csv` | results/results.csv | CSV file to append results after each run |
| `--export-tflite` | off | Convert to TFLite after training |
| `--quant` | float32 | Quantization: `float32`, `dynamic`, or `int8` |

Each run creates a timestamped directory under `--outdir`:

```
runs/<timestamp>/
├── best.keras             best checkpoint by validation accuracy
├── final.keras            model saved after training completes
├── model.tflite           TFLite model (if --export-tflite)
├── normalization_params.h per-channel mean and std as C constants
├── history.csv            loss and accuracy per epoch
├── confusion_matrix.png
├── roc_curves.png
└── per_class_metrics.png
```

`model.tflite` and `normalization_params.h` are the two files needed for firmware deployment.

### 4.2 Classical classifiers – gesture dataset

```bash
cd model_training/classical_classifiers
python train_gesture.py
```

By default this runs 5-fold GroupKFold cross-validation (subjects do not appear in more than one fold). To train on all data and export C headers for firmware use:

```bash
python train_gesture.py --no-cv --export-header
```

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--win-sec` | 2.0 | Window length in seconds |
| `--overlap` | 0.5 | Window overlap fraction |
| `--no-cv` | off | Use a single hold-out split instead of GroupKFold CV |
| `--cv-splits` | 5 | Number of GroupKFold folds |
| `--test-size` | 0.3 | Test fraction for hold-out split (`--no-cv` only) |
| `--seed` | 42 | Random seed |
| `--dt-max-depth` | -1 | Decision Tree max depth (-1 = unlimited) |
| `--rf-n-estimators` | 120 | Number of trees in the Random Forest |
| `--rf-max-depth` | -1 | Random Forest max depth (-1 = unlimited) |
| `--export-header` | off | Write C headers (`model.h`, `rf_model.h`) |
| `--show-plots` | off | Display plots interactively |

With `--export-header`, two files are written to `runs/gesture/`:
- `model.h` – Logistic Regression, Linear SVM, Decision Tree as C inline functions
- `rf_model.h` – Random Forest as C inline functions

### 4.3 Classical classifiers – UCI HAR dataset

```bash
cd model_training/classical_classifiers
python train_uci.py
```

This script uses the pre-split train/test partition provided by UCI HAR and evaluates the same five classifiers. It does not produce C headers.

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--labels` | all 6 classes | Activity labels to include |
| `--seed` | 42 | Random seed |
| `--dt-max-depth` | 10 | Decision Tree max depth (-1 = unlimited) |
| `--rf-n-estimators` | 120 | Number of trees in the Random Forest |
| `--rf-max-depth` | -1 | Random Forest max depth (-1 = unlimited) |
| `--show-plots` | off | Display plots interactively |

---

## 5. Firmware deployment

The deployment workflow: prepare model files -> import the MCUXpresso project -> build and flash the board.

### 5.1 Prepare model files (1D CNN)

Convert the TFLite model to a C byte array and copy it to the firmware project:

```bash
xxd -i model.tflite > model_data.h
```

Then:
1. Copy `model_data.h` to `inference/1D_CNN_model/model/`.
2. Copy `normalization_params.h` (from the training run directory) to `inference/1D_CNN_model/source/`.
3. If the model uses int8 quantization, set `MODEL_USE_INT8 1` in `source/app_config.h`; for float32, set it to `0`.

### 5.2 Import and build

1. Open MCUXpresso IDE.
2. **File -> Import -> Existing Projects into Workspace.**
3. Select `inference/1D_CNN_model` or `inference/classical_models`.
4. Select the **Debug** build configuration and click **Build**.

### 5.3 Flash and run

1. Connect the board via USB (J28).
2. **Run -> Debug** – the IDE flashes the binary over LinkServer.
3. Press **Resume (F8)**.
4. Open a serial terminal at **115200 baud**.
5. Press SW8 to start gesture recognition; press again to stop.

Expected UART output:

```
System ready. Press button to start/stop gesture recognition.
Gesture recognition: STARTED
Gesture: right | Inf: 12888 us | Lat: 12892 us
Stats: n=10 AvgInf=13751 us MaxInf=23737 us AvgLat=13756 us MaxLat=23741 us
Gesture recognition: STOPPED
```

- **Inf** – time spent executing the model (TFLite `Invoke()` only).
- **Lat** – time from the last sensor sample to the result (inference + preprocessing overhead).
- Statistics are printed every 10 windows.

### 5.4 Replacing the classical model

1. Run `python train_gesture.py --no-cv --export-header` to generate updated `model.h` and `rf_model.h`.
2. Copy the generated files to `inference/classical_models/source/`.
3. To select which classifier runs, uncomment the corresponding `predict_*` call in `inference/classical_models/source/task_mpu6050.c`.
4. Rebuild and flash.

---

## License

The source code and the gesture dataset in `model_training/data/hands/` written by the author are released under the [MIT License](LICENSE).

Third-party components keep their original licenses (see the headers of the individual files):

- NXP MCUXpresso SDK (drivers, board, device, startup files) – BSD-3-Clause
- CMSIS – Apache-2.0
- FreeRTOS – MIT
- TensorFlow Lite Micro (`eiq/`) – Apache-2.0
- FatFs – FatFs license (BSD-style)
