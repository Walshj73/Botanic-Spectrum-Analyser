# 🌿 BSA Benchmark

**A configuration-driven framework for plant image segmentation benchmarking**

BSA Benchmark is the standalone Python benchmarking package accompanying the Botanic Spectrum Analyser (BSA) research project. It enables researchers to prepare their own image datasets, train and compare selected segmentation methods, perform cross-validation, and export consistent evaluation reports.

The benchmarking package is independent of the BSA graphical application. It does not require the GUI's `Models/` or `BSA-internal/` resources.

## Supported methods

| Method | Configuration ID | Implementation |
|---|---|---|
| BSA / SDA-UNet | `bsa_sda_unet` | Segmentation network with a downsampling stem, residual blocks, attention gates, ASPP and squeeze-and-excitation attention |
| Random Forest | `random_forest` | Pixel classification using colour and texture features |
| SVM | `svm` | Standardised linear support-vector classification using colour and texture features |
| SLIC-RF | `slic_rf_v1` | SLIC superpixels, region features and Random Forest classification |
| FCN-ResNet50 | `fcn_resnet50` | Fully convolutional segmentation network with a ResNet-50 backbone |
| DeepLabV3+ | `deeplabv3plus` | DeepLabV3+ with a ResNet-50 backbone |
| PSPNet | `pspnet` | Pyramid Scene Parsing Network with a ResNet-50 backbone and progressive skip-fusion decoder |

Each method has configurable training and model settings in `configs/methods/`. Run only the methods and experiments you select; the package does not automatically execute the complete set of manuscript experiments.

## Installation

**Python 3.12 is recommended** for the classical and deep-learning methods tested with this package. A GPU is optional; TensorFlow support and acceleration depend on the platform and installed environment.

From the main repository:

```bash
cd benchmarking
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[classical,deep]'
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell.

Install only the components you need:

```bash
python -m pip install -e .               # Core package, data processing and evaluation
python -m pip install -e '.[classical]'  # Random Forest, SVM and SLIC-RF
python -m pip install -e '.[deep]'       # BSA and ResNet-based methods
python -m pip install -e '.[dev]'        # Tests and linting
```

The ResNet-based configurations default to randomly initialised backbones to avoid an automatic weights download. Set `hyperparameters.weights` to `imagenet` in a method configuration when pre-trained weights are required and available.

## Configure your dataset

Start with [`configs/datasets/example.yaml`](configs/datasets/example.yaml). A simple image/mask layout is:

```text
my_dataset/
├── images/
│   ├── sample_001.png
│   └── sample_002.png
└── masks/
    ├── sample_001.png
    └── sample_002.png
```

The dataset YAML defines image and mask paths, size and channels, filename pairing, mask encoding, resizing, normalisation, augmentation, source grouping and split settings. Paths can point to a dataset outside this repository.

For images and masks with different names, configure `pairing.strategy: token_replacement` (for example, `_rgb` → `_label`). If multiple images originate from one parent image or experimental unit, configure `splits.group_by_source: true` and `source_group_regex` so related samples remain in the same split.

The supplied SWIR configurations use **rendered false-colour RGB images**, not raw spectral cubes. They support configurable image-space auxiliary transforms including ExG, ExM, NExM and strict-purple contrast.

Validate your dataset configuration and sample contents:

```bash
bsa-benchmark validate dataset configs/datasets/example.yaml --check-paths --check-contents
```

The example configurations for the manuscript datasets require local data that is **not distributed** in this repository. For your own experiment, copy and edit the example YAML to point to your images and masks.

## Run an experiment

Experiment YAML files under `configs/experiments/` select a dataset, method, protocol, seeds and output settings. Method parameters live under `configs/methods/`; image processing and augmentation settings live in the dataset YAML.

Preview the included Barley example without training:

```bash
bsa-benchmark dry-run --experiment barley_rf_cross_validation --check-paths
```

Run the configured experiment (only after supplying its required dataset):

```bash
bsa-benchmark run --experiment barley_rf_cross_validation
```

Run only the first, zero-indexed cross-validation fold:

```bash
bsa-benchmark run --experiment barley_rf_cross_validation --fold 0
```

Choose your own dataset and a different method:

```bash
bsa-benchmark run \
  --experiment barley_bsa_development \
  --dataset configs/datasets/my_dataset.yaml \
  --method deeplabv3plus
```

The `run` command performs dataset indexing, group-aware splitting, preprocessing, training, inference, evaluation and report generation. Every invocation creates a separate output directory:

```text
runs/<dataset>/<method>/<experiment>/<run-id>/
├── resolved_config.yaml
├── metadata.json
├── status.json
├── splits/
├── checkpoints/
├── predictions/
├── metrics/
├── reports/
└── logs/
```

Run outputs are excluded from version control; the published numerical records under `results/` are not overwritten.

## Evaluation and results

The shared evaluator reports pixel accuracy, mean IoU, Dice/F1, precision, recall, frequency-weighted IoU, per-class metrics and confusion matrices. Reports include per-image, per-fold and aggregate outputs as appropriate to the experiment.

The [`results/`](results/) directory contains numerical records associated with the BSA manuscript, organised by dataset:

```text
results/
├── barley/
├── barley_swir/
├── dicot/
└── wheat/
```

These records include available segmentation metrics, cross-validation summaries, training histories, model comparisons and computational-cost measurements. They are provided for inspection alongside the manuscript; **new run outputs are stored separately** under `runs/`.

## Testing

The test suite uses small synthetic fixtures and does not require the manuscript image datasets.

```bash
python -m pip install -e '.[classical,deep,dev]'
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
```

## Data and scope

Raw manuscript images and ground-truth masks are not distributed with this repository. Use your own dataset and the supplied configuration examples. Dataset contents, predictions, checkpoints, caches, local settings and generated runs are excluded from version control.

The SWIR examples operate on rendered image-space representations. Wheat's separately retained multi-level source masks are not used by the supplied binary-mask configuration. Full manuscript-result equivalence and GPU execution across every hardware configuration have not been established.

For the BSA GUI, publication details and licence, see the [main repository README](../README.md) and [LICENSE](../LICENSE).
