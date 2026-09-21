# 🌿 Botanic Spectrum Analyser (BSA)

<p align="center">
  <img src="BSA_logo.png" width="200" alt="Botanic Spectrum Analyser logo">
</p>

**A graphical application for plant image segmentation and hyperspectral analysis**

Botanic Spectrum Analyser (BSA) is a Python-based graphical user interface for plant scientists, breeders and biologists. It combines deep learning segmentation with hyperspectral image processing, vegetation index calculation, spectral visualisation and data export.

This repository contains two separate components:

- **BSA application:** the graphical program launched from `BSA_v0.5.2.py`.
- **[BSA Benchmark](benchmarking/):** a standalone, configuration-driven package for training and evaluating segmentation methods. Its [README](benchmarking/README.md) contains installation and usage instructions.

The benchmarking package has its own dependencies and **does not require** the GUI's packaged runtime resources.

## BSA application features

- **Mask Creator:** segment RGB images and RGB representations of hyperspectral images, process batches and export segmentation masks.
- **HDR Creator:** generate ENVI-compatible `.hdr` metadata for hyperspectral data, including band, wavelength and interleave information.
- **Hypercube analysis:** apply dark/white reference calibration, extract mask-based regions of interest and export spectral measurements.
- **Vegetation indices:** calculate built-in indices, including NDVI, PRI, PSRI, SIPI, NDRE and WP1, or define custom indices.
- **Spectral Visualiser:** compare mean spectral curves, group samples using labels and export plots and spectra.

## Application resources

The BSA application uses pre-trained models and, for the packaged Windows executable, supporting runtime files. These large resources are distributed separately from the source code.

Visit **[CSI-Dublin](https://csi-dublin.ie/)**, navigate to **Resources**, and download the BSA resource package. Extract it and place the supplied resources alongside the GUI script as described in the package instructions:

```text
Botanic-Spectrum-Analyser/
├── BSA_v0.5.2.py
├── Models/             # Pre-trained application models
└── BSA-internal/       # Packaged executable runtime resources
```

`Models/` is needed for the GUI's pre-trained segmentation functions. `BSA-internal/` is part of the packaged application distribution; it is **not** a dependency of the separate benchmarking package.

## Running the BSA GUI

### From source

Clone the repository (you will need access if the repository is private):

```bash
git clone https://github.com/Walshj73/Botanic-Spectrum-Analyser.git
cd Botanic-Spectrum-Analyser
```

Create a virtual environment and install the GUI dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install tensorflow opencv-python numpy pandas matplotlib seaborn spectral pillow tqdm cryptography PyPDF2
```

Place the required model files in `Models/`, then launch:

```bash
python BSA_v0.5.2.py
```

Dependency compatibility for the GUI may depend on your operating system and Python version. The GUI installation above is separate from the tested Python 3.12 benchmarking environment.

### Windows executable

Download and extract the packaged BSA release from the CSI-Dublin Resources page, keeping its models and runtime directories alongside the executable. Launch the supplied `.exe`; a separate Python installation is not required for the packaged application.

## 🔬 Segmentation benchmarking

The **[benchmarking/](benchmarking/)** directory contains a standalone Python package for user-selected segmentation experiments. It supports BSA/SDA-UNet, Random Forest, SVM, SLIC-RF, FCN-ResNet50, DeepLabV3+ and PSPNet.

Researchers can configure datasets, training parameters and cross-validation runs, then generate predictions and evaluation reports. The numerical records associated with the BSA manuscript are available under **[benchmarking/results/](benchmarking/results/)**. New experiment output is kept separate from those records.

Start with the **[benchmarking README](benchmarking/README.md)** for installation instructions, dataset configuration and example commands. The raw manuscript image datasets are not included in this repository.

## Citation

If you use BSA or its benchmarking framework, please cite the associated manuscript. The preprint record is:

Walsh, J., et al. (2025). *Botanic Spectrum Analyser: A Deep Learning GUI for Plant Image Segmentation in Hyperspectral and RGB Phenotyping.* bioRxiv. [https://doi.org/10.1101/2025.09.14.676080](https://doi.org/10.1101/2025.09.14.676080)

If citing the final journal publication, use its definitive bibliographic record in place of the preprint reference above.

## License

See [LICENSE](LICENSE) for the repository's licence terms.

## Acknowledgements

University College Dublin · CRRBM — University of Picardie Jules Verne · CSI-Dublin
