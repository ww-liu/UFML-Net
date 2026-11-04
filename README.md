# UFML-Net

PyTorch implementation of UFML-Net for 12-lead ECG classification on PTB-XL and CPSC2018.

The framework combines local morphology and global rhythm branches with input-adaptive dynamic group convolution (IADGConv), uncertainty-guided mutual learning (UIML), frequency-domain auxiliary compensation (FreqAux), and reliability-guided prediction fusion.

## Environment

Main packages in the server's `mrm_net` environment:

| Package | Version |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.1.0+cu121 |
| torchvision | 0.16.0+cu121 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| SciPy | 1.13.1 |
| scikit-learn | 1.6.1 |
| WFDB | 4.3.0 |
| PyWavelets | 1.6.0 |
| segmentation-models-pytorch | 0.5.0 |
| tqdm | 4.67.1 |
| Matplotlib | 3.9.4 |

## Datasets

**PTB-XL.** Obtain the dataset from [PhysioNet](https://physionet.org/content/ptb-xl/). The loader uses the 100 Hz recordings and the provided folds: 1-8 for training, 9 for validation, and 10 for testing.

**CPSC2018.** The loader supports 12-lead ECG recordings with nine diagnostic categories. It reads MATLAB recordings from `TrainingSet1`, `TrainingSet2`, and `TrainingSet3`, together with the `Recording` and `First_label` columns in `cpsc_database.csv`. Existing `data2.npy` and `labels2.npy` caches can also be used.

Arrange the data as follows:

```text
data/
├── ptbxl/
│   └── raw/
│       ├── ptbxl_database.csv
│       ├── scp_statements.csv
│       └── records100/
└── cpsc2018/
    ├── cpsc_database.csv
    ├── TrainingSet1/
    ├── TrainingSet2/
    └── TrainingSet3/
```

Set the dataset path and experiment settings in `config.py`, or override them through the command-line arguments of `main_mul.py`.

## Main files

| File | Description |
|---|---|
| `main_mul.py` | Training and evaluation entry point |
| `config.py` | Model and experiment settings |
| `dataset.py` | Dataset selection and data loaders |
| `data_process.py` | PTB-XL loading and preprocessing |
| `data_load.py` | CPSC2018 loading and preprocessing |
| `models/ufml_net.py` | UFML-Net interface |
| `models/duibi.py` | Dual-branch backbone and FreqAux |
| `models/attention.py` | Convolution and modulation components |
| `utils.py` | Shared utilities and evaluation metrics |
