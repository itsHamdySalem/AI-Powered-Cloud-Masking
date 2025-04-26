# AI-Powered Cloud Masking

This project develops an AI-powered cloud masking system capable of automatically identifying clouds from satellite images. The system takes a satellite image as input and generates a corresponding cloud mask as output.

## Project Overview

The system uses a combination of deep learning (U-Net) and classical machine learning (Random Forest) approaches to perform cloud segmentation on satellite imagery containing four spectral bands (Red, Green, Blue, and Infrared).

## Features

- Data exploration and preprocessing capabilities
- U-Net deep learning model for cloud segmentation
- Random Forest classifier as a classical ML approach
- Model evaluation using Dice coefficient
- Inference script for generating predictions on new data

## Requirements

```txt
segmentation-models-pytorch
albumentations
rasterio
torch
numpy
pandas
scikit-learn
tqdm
matplotlib
tensorboard
opencv-python
pathlib
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/itsHamdySalem/AI-Powered-Cloud-Masking.git
   cd AI-Powered-Cloud-Masking
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset consists of satellite images with four spectral bands (Red, Green, Blue, and Infrared) and corresponding cloud masks. The images are in TIFF format.

## Usage

### Data Exploration

To analyze the dataset distribution and characteristics:

```bash
python data_exploration.py --images_dir path/to/images --masks_dir path/to/masks
```

This generates visualizations and statistics about cloud coverage, band distributions, and potential data quality issues.

### Model Profiling

To profile the model and get information about its parameters and operations:

```bash
python profile.py --model_path models/cloud_mask_unet.pkl --log_path model_logs.txt
```

### Model Evaluation

To evaluate the model performance on validation data:

```bash
python evaluate_model.py --model_path models/cloud_mask_unet.pkl --images_dir path/to/val/images --masks_dir path/to/val/masks --output_dir evaluation_results
```

This generates detailed metrics, visualizations of best/worst predictions, and error pattern analysis.

### Inference

To run inference on new data:

```bash
python run_inference.py --test_dir path/to/test/images --model_path models/cloud_mask_unet.pkl --output submission.csv
```

## Models

The project includes two trained models:

1. **U-Net (Primary Model)**: A deep learning model based on the U-Net architecture, optimized for cloud segmentation.
   - Located at: `models/cloud_mask_unet.pkl`
   - Parameters: See `model_logs.txt`

2. **Random Forest**: A classical machine learning approach using pixel-level features.
   - Located at: `models/cloud_mask_rf.pkl`

## Evaluation

The models are evaluated using the Dice coefficient, which measures the overlap between the predicted mask and the ground truth.

## Development

The models were developed and trained using Kaggle notebooks:

- `cloud-masking-using-u-net.ipynb`: Development of the U-Net model
- `cloud-masking-using-random-forest-classifier.ipynb`: Development of the Random Forest classifier
