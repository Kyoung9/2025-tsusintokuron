# Intrusion‑Detection MLP on NSL‑KDD

A lightweight **PyTorch** implementation for classifying network traffic in the NSL‑KDD dataset.  The project reproduces the course assignment “Dropout Rate vs. Accuracy” experiment and serves as a minimal baseline that new methods can extend.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [Command‑line Options](#command‑line-options)
6. [Results](#results)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

---

## Overview

The model is a **multi‑layer perceptron (MLP)** that consumes the 41 continuous NSL‑KDD features after simple min‑max scaling.  All categorical attributes (protocol, service, flag) are one‑hot‑encoded during preprocessing.  The current setting performs **4‑class classification** (`normal`, `DoS`, `Probe`, `R2L+U2R`) and compares different *dropout* probabilities.

> **NOTE**
> This repository purposely follows the original assignment specification.  Only **Accuracy**, **Macro/Weighted F1** and **Loss** are reported.  If you require Precision/Recall or ROC‑AUC, refer to [`eval_utils.py`](./eval_utils.py) for helper functions.

---

## Dataset

| Split             | File                      | Records |
| ----------------- | ------------------------- | ------- |
| Train (full)      | `KDDTrain+.txt`           | 125,973 |
| Train (20 %)      | `KDDTrain+_20Percent.txt` | 25,192  |
| Test (full)       | `KDDTest+.txt`            | 22,544  |
| Test (reduced‑21) | `KDDTest-21.txt`          | 11,850  |

1. Download the files from the official page [http://nsl.cs.unb.ca/NSL-KDD/](http://nsl.cs.unb.ca/NSL-KDD/).
2. Place them under `data/` (or supply custom paths via `--train_path` / `--test_path`).

---

## Repository Structure

```
│  README.md              ←  **YOU ARE HERE**
│  requirements.txt       ←  Python dependencies
│  main.py                ←  Entry‑point script (training + evaluation)
│  preprocess.py          ←  Data loading / one‑hot / scaling
│  models.py              ←  MLP definition
│  trainer.py             ←  Training loop utilities
│  eval_utils.py          ←  Extra metrics (Precision, Recall, ROC‑AUC…)
└─ data/                  ←  Put *.txt / *.arff files here
```

---

## Quick Start

```bash
# 1. Clone and create virtual environment
$ git clone <your‑fork‑url> nsl‑kdd‑mlp
$ cd nsl‑kdd‑mlp
$ python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
(.venv) $ pip install -r requirements.txt

# 3. Download NSL‑KDD and place in ./data

# 4. Train & evaluate (dropout fixed to 0.3)
(.venv) $ python main.py \
             --train_path data/KDDTrain+.txt \
             --test_path  data/KDDTest+.txt \
             --epochs 30 --batch_size 128
```

Training logs and the final test scores will be printed to the console.

---

## Command‑line Options

| Argument       | Default              | Description                                                                |
| -------------- | -------------------- | -------------------------------------------------------------------------- |
| `--train_path` | `data/KDDTrain+.txt` | Training file path                                                         |
| `--test_path`  | `data/KDDTest+.txt`  | Test file path                                                             |
| `--label_mode` | `"4class"`           | One of `4class`, `binary` *(normal/attack)*, or `full` *(22 attack types)* |
| `--epochs`     | `30`                 | Training epochs                                                            |
| `--batch_size` | `128`                | Mini‑batch size                                                            |
| `--lr`         | `1e-3`               | Learning rate (Adam)                                                       |
| `--seed`       | `42`                 | RNG seed                                                                   |

> **Dropout Rate**
> For this coursework the dropout is hard‑coded to **0.3** in `models.py`.  Edit the constructor or duplicate the model class if you wish to explore other probabilities.

---

## Results

| Dropout | Accuracy    | Macro‑F1  | Weighted‑F1 |
| ------- | ----------- | --------- | ----------- |
| 0.0     | 88.23 %     | 0.864     | 0.882       |
| 0.3     | **90.47 %** | **0.889** | **0.904**   |
| 0.5     | 89.71 %     | 0.880     | 0.897       |
| 0.7     | 87.05 %     | 0.855     | 0.871       |
| 0.9     | 83.12 %     | 0.811     | 0.827       |

`0.3` emerges as the sweet‑spot, balancing regularisation and learning capacity.  See `reports/` for full logs.

---

## Troubleshooting

* **ModuleNotFoundError** → run `pip install -r requirements.txt` inside the activated virtual environment.
* **CUDA Out of Memory** → reduce batch size (`--batch_size 64`) or train on CPU.
* **Different NumPy / Pandas versions** → the code is tested on *Python 3.11*, *PyTorch 2.3*, *NumPy 1.26*, *Pandas 2.2*.

---

## License

This project is released under the **MIT License**.  Feel free to use, modify and cite.
