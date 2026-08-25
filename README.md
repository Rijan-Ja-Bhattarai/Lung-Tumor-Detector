# Lung Tumor Segmentation with U-Net

A lightweight deep-learning project for segmenting suspected tumor regions in axial lung CT slices. The project uses a custom PyTorch U-Net, a Flask inference API, and a browser interface that displays the original scan, predicted binary mask, and a highlighted overlay.

> [!WARNING]
> This project is intended for education and research only. It is not a medical device and must not be used to diagnose, rule out, or treat disease. Predictions require review by qualified medical professionals.

![Lung CT segmentation result](images/Screenshot%202026-08-25%20113039.png)

## Features

- Accepts a processed `.png` CT slice or a single-frame axial `.dcm` image.
- Performs binary semantic segmentation with a compact U-Net.
- Supports CUDA automatically when a compatible GPU is available and otherwise uses the CPU.
- Displays the original CT image, predicted mask, and red prediction overlay.
- Uses patient-level training and validation splits to reduce data leakage.
- Includes class-weighted binary cross-entropy and Dice loss for imbalanced masks.

## Application preview

### Upload interface

![CT image upload interface](images/Screenshot%202026-08-25%20113011.png)

### Example with no pixels above the prediction threshold

![Negative segmentation example](images/Screenshot%202026-08-25%20113022.png)

### Example with a predicted region

![Positive segmentation example](images/Screenshot%202026-08-25%20113039.png)

## Technology

- Python
- PyTorch
- Flask
- NumPy
- Pillow
- pydicom
- Matplotlib
- ITK-SNAP and 3D Slicer for annotation/data preparation

## Project structure

```text
Lung-Tumor detection/
├── app.py                         # Flask server and inference pipeline
├── index.html                     # Browser interface
├── train.py                       # Dataset, U-Net, training, and evaluation
├── lung_tumor_unet_best.pt        # Trained model weights used by the app
├── predictions.png                # Training/validation prediction grid
├── images/                        # README application screenshots
├── Lung Tumor/
│   ├── RIDER-Clean-Slices/        # Prepared images and aligned masks
│   └── RIDER-Lung-CT-Processed/   # Processed RIDER CT scans
└── dataset.csv                    # Dataset metadata
```

Large dataset folders may be excluded from version control. Obtain the dataset separately and configure `DATA_ROOT` and `LOCAL_ROOT` in `train.py` if their locations differ.

## Getting started

### 1. Create a Conda environment

```powershell
conda create -n Tumor-Segmentation python=3.11 -y
conda activate Tumor-Segmentation
```

### 2. Install dependencies

Install PyTorch for your CPU or CUDA configuration by following the command recommended on the [PyTorch installation page](https://pytorch.org/get-started/locally/). Then install the remaining packages:

```powershell
pip install flask matplotlib numpy pillow pydicom
```

### 3. Check the model checkpoint

The application expects this file in the project root:

```text
lung_tumor_unet_best.pt
```

### 4. Run the web application

From the project root:

```powershell
conda run --no-capture-output -n Tumor-Segmentation python .\app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. Keep the terminal open while using the application and press `Ctrl+C` to stop it.

## Using the application

1. Select one processed PNG CT slice or one axial DICOM file.
2. Click **Predict segmentation**.
3. Review the original image, binary mask, and red overlay.

The upload limit is 32 MB. DICOM input must be a single-frame, monochrome, axial CT image containing pixel data and HU rescale metadata. Scout/localizer images are rejected.

## Training

Before training, update these paths near the top of `train.py` if necessary:

```python
DATA_ROOT = Path("path/to/RIDER-Clean-Slices")
LOCAL_ROOT = Path("path/to/RIDER-Lung-CT-Processed")
```

The prepared positive dataset must contain matching images and masks:

```text
RIDER-Clean-Slices/
├── images/
└── aligned_masks/
```

Run training with:

```powershell
conda run --no-capture-output -n Tumor-Segmentation python .\train.py
```

Training uses 256 × 256 grayscale inputs, a patient-level 80/20 split, horizontal-flip augmentation, and five epochs by default. It saves the best checkpoint as `best_model.pt` and creates `predictions.png`. These settings can be changed using the constants at the top of `train.py`.

To serve a newly trained checkpoint with the current application, replace or rename it as `lung_tumor_unet_best.pt` in the project root before restarting the server.

## Model overview

The custom `SmallUNet` contains two encoder stages, a bottleneck, two decoder stages with skip connections, and a one-channel segmentation head. Inference applies a sigmoid function and converts probabilities at or above `0.5` into the final binary mask.

## Project information

- **Project lead:** Rijan Bhattarai
- **Dataset:** RIDER Lung CT data with manually prepared segmentation masks

## Limitations

- Performance depends on the training data, annotation quality, preprocessing, and scanner characteristics.
- A segmentation can contain false positives or miss real abnormalities.
- The model evaluates one two-dimensional CT slice at a time and does not use full 3D scan context.
- The interface does not provide a clinical diagnosis or measure tumor stage.
