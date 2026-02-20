🌿 Botanic Spectrum Analyser (BSA)

<p align="center"> <img src="BSA_logo.png" width="200" alt="BSA Logo"> </p>

A Deep Learning GUI for Plant Image Segmentation and Hyperspectral
Analysis

Botanic Spectrum Analyser (BSA) is a free and open-source graphical user
interface (GUI) built in Python that enables plant scientists, breeders,
and biologists to perform:

-   🌱 Deep learning–based plant image segmentation
-   🌈 Hyperspectral image calibration and analysis
-   📊 Vegetation index extraction (NDVI, PRI, PSRI, SIPI, NDRE, WP1,
    etc.)
-   📈 Mean spectral curve visualization
-   🧠 Custom vegetation index computation

BSA bridges the gap between advanced AI segmentation and practical plant
phenotyping workflows — without requiring machine learning expertise.

------------------------------------------------------------------------

📦 Required Resource Download (Important)

Due to GitHub file size limitations, the Models/ and BSA-internal/
directories are not included in this repository.

To run BSA correctly, you must download the full resource package:

🔗 Download from:
https://csi-dublin.ie/

Navigate to the Resources section and download the full BSA RAR package.

After downloading:

1.  Extract the RAR archive
2.  Copy the following folders into the root of this repository:

    ```text
    Botanic-Spectrum-Analyser/
    │
    ├── BSA_v0.5.2.py
    ├── Models/              ← Copy from RAR
    ├── BSA-internal/        ← Copy from RAR
    ```

⚠️ These folders are required for:

-   Pre-trained U-Net models (.h5 files)
-   TensorFlow runtime binaries
-   Internal dependencies used by the compiled executable

Without these folders, segmentation and hyperspectral analysis will not
function.

------------------------------------------------------------------------

🚀 Features

🧠 1. Deep Learning Segmentation (Mask Creator)

-   Pre-trained U-Net models (.h5)
-   RGB image segmentation
-   Hyperspectral RGB-slice segmentation
-   Batch processing
-   Automatic binary mask export

🌗 2. HDR Creator (ENVI Header Generator)

-   Create ENVI .hdr files
-   Define bands, wavelengths, interleave, metadata
-   Supports BIL / RAW hyperspectral formats

📦 3. Hyperspectral Hypercube Analysis

-   Dark / White calibration
-   Pixel-wise calibration
-   Mask-based ROI extraction
-   Sequential memory-safe file processing

📊 4. Built-in Vegetation Indices

For VNIR:

-   NDVI
-   PRI
-   PSRI
-   SIPI
-   NDRE
-   WP1

For SWIR:

-   Water indices

📈 5. Spectral Visualiser

-   Mean spectral plots
-   Label-based grouping (via Labels.csv)
-   Export plots
-   Export spectra data

------------------------------------------------------------------------

🖥️ Software Architecture

From BSA_v0.5.2.py:

-   GUI: Tkinter
-   Deep Learning: TensorFlow / Keras
-   Image Processing: OpenCV
-   Hyperspectral Processing: spectral (ENVI support)
-   Plotting: Matplotlib / Seaborn
-   Data handling: NumPy / Pandas

------------------------------------------------------------------------

📂 Repository Structure

    Botanic-Spectrum-Analyser/
    │
    ├── BSA_v0.5.2.py           # Main GUI application
    ├── Models/                 # Pre-trained models (download separately - see above)
    ├── BSA-internal/           # Runtime binaries (download separately - see above)
    ├── BSA_logo.png
    ├── BSA_logo.ico
    └── README.md

------------------------------------------------------------------------

⚙️ How to Run BSA

Option 1 — Run from Source

1.  Clone the repository
    ```r
    git clone https://github.com/YOUR-USERNAME/Botanic-Spectrum-Analyser.git
    cd Botanic-Spectrum-Analyser
    ```

2.  Create Virtual Environment (Optional)
    ```r
    python -m venv venv
    Windows: venv\Scripts\activate
    ```

3.  Install Dependencies
    ```r
    pip install tensorflow opencv-python numpy pandas matplotlib seaborn spectral pillow tqdm cryptography PyPDF2
    ```

4.  Ensure Models/ and BSA-internal/ folders are copied from the downloaded RAR package.

5.  Run the application
    ```r
    python BSA_v0.5.2.py
    ```
------------------------------------------------------------------------

Option 2 — Run Standalone Executable

-   Extract the compiled ZIP package
-   Ensure BSA-internal/ and Models/ folders are present and populated
-   Double-click the .exe file

No Python installation required.

------------------------------------------------------------------------

💡 System Requirements

Recommended:

-   Python 3.9+
-   8GB+ RAM
-   GPU optional (TensorFlow supports CPU execution)

------------------------------------------------------------------------

🔓 License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files…

(Include full MIT license text in LICENSE file)

------------------------------------------------------------------------

🤝 Contributing

Contributions are welcome!

-   Fork the repo
-   Create a feature branch
-   Submit a pull request

------------------------------------------------------------------------

🌿 Acknowledgements

University College Dublin
CRRBM – University of Picardie Jules Verne
