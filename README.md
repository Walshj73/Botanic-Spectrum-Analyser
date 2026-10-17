# Botanic Spectrum Analyser

<p align="center">
  <img src="src/bsa/assets/BSA_logo.png" width="200" alt="Botanic Spectrum Analyser logo">
</p>

Botanic Spectrum Analyser (BSA) is a desktop application for plant image
segmentation and hyperspectral analysis. It is designed for plant scientists,
breeders, and biologists who want one tool to create plant masks, calibrate
hyperspectral images, calculate vegetation indices, compare spectra, and
export results. No programming is required.

BSA comes with a companion tool, **BSA Label Creator**, which helps you assign
experimental groups (for example *Control* and *Drought*) to your samples so
BSA can compare them.

## Download and run

Ready-to-use packages are on the
**[Releases page](https://github.com/Walshj73/Botanic-Spectrum-Analyser/releases/latest)**.
You do not need to install Python or anything else. The pre-trained
segmentation models are a separate download; see
[Segmentation models](#segmentation-models).

### Linux

1. Download `Botanic-Spectrum-Analyser-Linux-x86_64.AppImage` from the
   Releases page.
2. Make it runnable: right-click the file, choose **Properties → Permissions**,
   and tick **Allow executing file as program**. Or, in a terminal:

   ```bash
   chmod +x Botanic-Spectrum-Analyser-Linux-x86_64.AppImage
   ```

3. Double-click the file to start **BSA**.
4. To start **Label Creator**, run this in a terminal from the download folder:

   ```bash
   ./Botanic-Spectrum-Analyser-Linux-x86_64.AppImage --label-creator
   ```

**Prefer a normal folder?** Download
`Botanic-Spectrum-Analyser-Linux-x86_64.tar.gz` instead and extract it. The
folder contains two programs: `botanic-spectrum-analyser` (BSA) and
`bsa-label-creator` (Label Creator). Double-click either one. Keep the
folder together. Model files placed in its `Models/` folder appear in the
Mask Creator's model list automatically.

**Requirements:** a 64-bit (x86_64) Linux desktop from roughly 2020 or later,
such as Ubuntu 20.04+, Debian 11+, or Fedora 32+. About 2 GB of free disk
space. A graphics card is not needed.

<details>
<summary>The AppImage will not start</summary>

Some systems need FUSE to run AppImages. On Ubuntu 22.04 and later, install
it with `sudo apt install libfuse2`. Or start the AppImage without FUSE:

```bash
APPIMAGE_EXTRACT_AND_RUN=1 ./Botanic-Spectrum-Analyser-Linux-x86_64.AppImage
```

The `.tar.gz` download does not need FUSE.
</details>

<details>
<summary>Checking your download (optional)</summary>

The Releases page includes `SHA256SUMS.txt`. Put it in the same folder as
your download and run `sha256sum -c --ignore-missing SHA256SUMS.txt`. It
should report `OK`.
</details>

### Segmentation models

Mask Creator needs a pre-trained model to find the plants in your images. The
other tabs work without one. The models (Barley SWIR, Barley VNIR, Dicot, and
Gymnosperm), example datasets, and example header files are published on
Zenodo and linked from the [CSI-Dublin website](https://csi-dublin.ie/).

1. Download the model you need (a `.h5` file).
2. In BSA's **Mask Creator** tab, open the model list and choose
   **IMPORT MY MODEL**, then select the downloaded file.

### Windows

A Windows version is in preparation and will be published on the Releases
page. Until then, Windows users can run BSA from source (see
[For developers](#for-developers)).

## What you can do with BSA

The BSA window has five tabs, used roughly in order:

1. **Mask Creator** finds the plants in RGB images, or in RGB views of
   hyperspectral images, using pre-trained models, and saves
   them as masks.
2. **Define Camera Settings** creates the ENVI `.hdr` header file that
   describes your camera's image size, data format, and wavelengths.
3. **Upload Files** selects your dark, data, and white reference headers, your
   hyperspectral images, your masks, and an optional label file from Label
   Creator.
4. **Analyse** calibrates the images, extracts plant pixels using the masks,
   and calculates built-in or custom vegetation indices.
5. **Visualise Spectra** plots and compares spectra by sample or group, and
   saves plots and spectral tables.

Results can be saved as CSV or Excel (XLSX). Plots can be saved as PNG or
JPEG. BSA understands the PlantScreen BIL and VNIR RAW file-naming
conventions.

### Label Creator

Label Creator lists the acquisitions in your dataset folders so you can
assign each one to an experimental group, individually or in bulk. You can
use wildcards, filters, sorting, and undo. It saves a label file that you
load into BSA's **Upload Files** tab. It does not open or change your
hyperspectral images.

The label file format is described in [LABEL_MANIFEST.md](LABEL_MANIFEST.md).

### Your data

You supply your own images, header files, and masks. Example datasets are
not included. BSA asks where to save results and never writes into the
application folder.

Only load segmentation models from sources you trust. BSA checks models
before loading them, but this check is not a full security sandbox.

## For developers

### Running from source

Python 3.12 is recommended. From the repository root:

```bash
python3.12 -m venv .venv-gui
source .venv-gui/bin/activate          # Windows: .venv-gui\Scripts\activate
python -m pip install -r requirements-gui.txt
```

Tkinter comes with Python on Windows. On Linux, install your distribution's Tk
package if it is missing, for example `sudo apt install python3-tk` or
`sudo pacman -S --needed tk`.

Start BSA and Label Creator with:

```bash
python botanic_spectrum_analyser.py
python bsa_label_creator.py
```

The repository does not contain the pre-trained models. To use Mask Creator
from source, place compatible `.h5` models in a `Models/` folder at the
repository root, or use **IMPORT MY MODEL** in the application. See
[Segmentation models](#segmentation-models) for where to download them.

### Tests

```bash
python -m unittest discover -s tests -t . -v
```

Tests that need a display are skipped when none is available.

### Building release packages

Instructions for building the Linux AppImage and tarball are in
[packaging/linux/README.md](packaging/linux/README.md).

## Benchmarking

The separate [BSA Benchmark package](benchmarking/) trains and evaluates
the segmentation methods used in the BSA manuscript. See its
[README](benchmarking/README.md) for installation and usage. The published
numerical results are in [benchmarking/results/](benchmarking/results/).
It has its own dependencies and does not need the BSA application.

## Citation

If you use BSA or its benchmarking framework, please cite:

Walsh, J., et al. (2025). *Botanic Spectrum Analyser: A Deep Learning GUI for
Plant Image Segmentation in Hyperspectral and RGB Phenotyping.* bioRxiv.
[https://doi.org/10.1101/2025.09.14.676080](https://doi.org/10.1101/2025.09.14.676080)

If a journal version has been published, please cite that instead.

## Licence

BSA is released under the [MIT License](LICENSE).

## Acknowledgements

University College Dublin · CRRBM — University of Picardie Jules Verne · CSI-Dublin
