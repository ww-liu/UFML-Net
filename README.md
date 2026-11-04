# UFML-Net

PyTorch implementation of UFML-Net for multi-label classification of 12-lead ECG recordings. This repository provides the model and training pipeline for the PTB-XL main experiments.

UFML-Net combines a local morphology branch and a global rhythm branch with input-adaptive dynamic group convolution (IADGConv), uncertainty-guided mutual learning (UIML), frequency-domain auxiliary compensation (FreqAux), and reliability-guided prediction fusion.

## Installation

Clone the repository and enter its root directory:

```bash
git clone https://github.com/ww-liu/test.git
cd test
```

Install [PyTorch](https://pytorch.org/get-started/locally/) for your compute platform, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists the core dependencies without locking versions. The training entry uses `segmentation_models_pytorch.losses.FocalLoss` when that package is installed; otherwise, it uses the local multilabel focal-loss implementation in `main_mul.py`. Retain the loss backend and package versions of an existing experiment when reproducing that run.

## PTB-XL data

Obtain PTB-XL from [PhysioNet](https://physionet.org/content/ptb-xl/). Arrange the metadata and low-resolution waveform files as follows:

```text
/path/to/ptbxl/
└── raw/
    ├── ptbxl_database.csv
    ├── scp_statements.csv
    └── records100/
        ├── 00000/
        ├── 01000/
        └── ...
```

The waveform paths in `ptbxl_database.csv` are resolved relative to `raw/`. The default input is a 10-second, 12-lead recording sampled at 100 Hz, with 1,000 time points.

The loader uses the `strat_fold` field provided by PTB-XL:

- Training: folds 1-8
- Validation: fold 9
- Testing: fold 10

Signal standardization is fitted on the training set and applied to validation and test recordings. Labels are generated using the task-specific aggregation in `data_process.py`.

## Train on PTB-XL

Run the following command from the repository root for the All-label task. Replace the dataset path with your own location:

```bash
python main_mul.py \
  --dataset ptbxl \
  --model_name ufml_net \
  --experiment exp0 \
  --datafolder /path/to/ptbxl \
  --seed 1 \
  --run_name ufml_ptbxl_all_seed1
```

The same command can be entered on one line in PowerShell or Command Prompt:

```text
python main_mul.py --dataset ptbxl --model_name ufml_net --experiment exp0 --datafolder "D:/data/ptbxl" --seed 1 --run_name ufml_ptbxl_all_seed1
```

The default All-label configuration uses Adam, a learning rate of `0.001`, a batch size of `64`, and `100` epochs. Task-specific settings are defined in `config.py`; command-line arguments override the corresponding configuration fields.

To inspect supported arguments:

```bash
python main_mul.py --help
```

### Task arguments

| Experiment | PTB-XL task |
|---|---|
| `exp0` | All labels |
| `exp1` | Diagnostic |
| `exp1.1` | Subdiagnostic |
| `exp1.1.1` | Superdiagnostic |
| `exp2` | Form |
| `exp3` | Rhythm |

Change `--experiment` to select a task. Use a distinct `--run_name` for each task and seed.

## Evaluation and outputs

With the default PTB-XL All-label settings, the training pipeline selects the best epoch using validation Macro-AUROC plus Sample-F1 of the fused predictions and evaluates the selected model states on the test set after training. EMA and SWA states participate in evaluation when enabled.

The default `final_eval_mode='ensemble'` uses the best validation states retained during training. `save_ensemble_checkpoints=False` saves a compact checkpoint; set `config.save_ensemble_checkpoints=True` before a run if all selected ensemble states need to be saved together for later evaluation.

The main run artifacts are:

```text
results/ptbxl/<run_name>/
├── config.json
├── final_metrics.csv
└── predictions.npz

checkpoints/
└── ..._best.pth
```

`final_metrics.csv` records branch-specific and fused test metrics, the seed, selected epoch, and checkpoint path. `predictions.npz` contains the final test targets and predictions. Macro-AUROC is computed across classes, and Sample-F1 uses a probability threshold of `0.5` under the default configuration.

## Main source files

| File | Role |
|---|---|
| `main_mul.py` | Training, validation, final evaluation, and checkpoint management |
| `config.py` | Model and experiment settings |
| `dataset.py` | Task selection and data loaders |
| `data_process.py` | PTB-XL loading, label aggregation, and standardization |
| `models/registry.py` | Model construction |
| `models/ufml_net.py` | UFML-Net interface |
| `models/duibi.py` | Dual-branch backbone and FreqAux implementation |
| `models/attention.py` | Convolution and modulation components |
| `utils.py` | Shared utilities and metric functions |

Run the project through `main_mul.py`; auxiliary modules remain in place to preserve the existing import structure and checkpoint compatibility. Historical code identifiers such as `Dui1`, `FDConv1D`, and `FrequencyAuxBranch` are retained.
