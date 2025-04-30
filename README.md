# AI-Powered Cloud Masking

This project develops an AI-powered cloud masking system capable of automatically identifying clouds from satellite images. The system takes a satellite image as input and generates a corresponding cloud mask as output.

## Project Overview

The system uses a combination of deep learning (U-Net and Deep Lab v3) and classical machine learning (Random Forest) approaches to perform cloud segmentation on satellite imagery containing four spectral bands (Red, Green, Blue, and Infrared).

## Features

- Data exploration and preprocessing capabilities
- U-Net and Deep Lab v3 deep learning model3 for cloud segmentation
- Random Forest classifier as a classical ML approach
- Model evaluation using Dice coefficient
- Inference script for generating predictions on new data

## Requirements

```txt
albumentations
opencv-python
matplotlib
numpy
pandas
rasterio
scikit-image
scikit-learn
segmentation-models-pytorch
torch
tqdm
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

## Evaluation

The models are evaluated using the Dice coefficient, which measures the overlap between the predicted mask and the ground truth.

## Development

The models were developed and trained using Kaggle notebooks:

- `cloud-masking-using-u-net.ipynb`: Development of the U-Net model.
- `cloud-masking-using-deep-lab-v3.ipynb`: Development of the Deep Lab v3 model.
- `cloud-masking-using-random-forest-classifier.ipynb`: Development of the Random Forest classifier.
