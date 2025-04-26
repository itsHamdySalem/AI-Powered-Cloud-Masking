import os
import cv2
import torch
import pickle
import argparse
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient
    
    Args:
        y_true: binary ground truth mask
        y_pred: binary predicted mask
        smooth: smoothing factor to prevent division by zero
    
    Returns:
        dice_coef: Dice coefficient value
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def load_model(model_path):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_mask(model, image_path, device, transform=None):
    """Generate a prediction mask for the given image."""
    if transform is None:
        transform = A.Compose([
            A.Resize(512, 512),
        ])
    
    try:
        with rasterio.open(image_path) as img_file:
            img = np.stack([img_file.read(i+1) for i in range(4)], axis=2)
        
        img = img.astype(np.float32)
        for i in range(img.shape[2]):
            if np.max(img[:,:,i]) > 0:
                img[:,:,i] = img[:,:,i] / np.max(img[:,:,i])
        
        img = transform(image=img)['image']
        
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(img)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > 0.5).float()
        
        mask = prediction.squeeze().cpu().numpy()
        
        return mask
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def evaluate_model(model, images_dir, masks_dir, output_dir, device, sample_count=None):
    """
    Evaluate model on validation data and analyze results
    
    Args:
        model: trained model
        images_dir: directory containing validation images
        masks_dir: directory containing validation masks
        output_dir: directory to save evaluation results
        device: torch device
        sample_count: number of samples to evaluate (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(list(Path(images_dir).glob('*.tif*')))
    mask_paths = sorted(list(Path(masks_dir).glob('*.tif*')))
    
    if len(image_paths) != len(mask_paths):
        print(f"Error: Number of images ({len(image_paths)}) doesn't match number of masks ({len(mask_paths)})")
        return
    
    if sample_count and sample_count < len(image_paths):
        indices = np.random.choice(len(image_paths), sample_count, replace=False)
        image_paths = [image_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]
    
    worst_dir = os.path.join(output_dir, 'worst_predictions')
    best_dir = os.path.join(output_dir, 'best_predictions')
    os.makedirs(worst_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    
    dice_scores = []
    result_data = []
    
    transform = A.Compose([A.Resize(512, 512)])
    
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Evaluating"):
        with rasterio.open(mask_path) as mask_file:
            gt_mask = mask_file.read(1)
            gt_mask = (gt_mask > 0).astype(np.uint8)
        
        gt_mask_resized = transform(image=gt_mask)['image']
        
        pred_mask = predict_mask(model, img_path, device, transform)
        if pred_mask is None:
            continue
        
        dice = dice_coefficient(gt_mask_resized, pred_mask)
        dice_scores.append(dice)
        
        gt_coverage = np.mean(gt_mask) * 100
        
        result_data.append({
            'image_path': str(img_path),
            'dice_score': dice,
            'cloud_coverage': gt_coverage
        })
    
    results_df = pd.DataFrame(result_data)
    
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    
    mean_dice = np.mean(dice_scores)
    median_dice = np.median(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Median Dice Coefficient: {median_dice:.4f}")
    print(f"Standard Deviation: {std_dice:.4f}")
    
    results_df = results_df.sort_values('dice_score')
    
    worst_samples = results_df.head(10)
    best_samples = results_df.tail(10)
    
    for i, row in tqdm(worst_samples.iterrows(), total=len(worst_samples), desc="Visualizing worst predictions"):
        visualize_prediction(row['image_path'], model, device, os.path.join(worst_dir, f"worst_{i+1}.png"), transform)
    
    for i, row in tqdm(best_samples.iterrows(), total=len(best_samples), desc="Visualizing best predictions"):
        visualize_prediction(row['image_path'], model, device, os.path.join(best_dir, f"best_{i+1}.png"), transform)
    
    plt.figure(figsize=(10, 6))
    plt.hist(dice_scores, bins=20, alpha=0.7, color='steelblue')
    plt.axvline(mean_dice, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dice:.4f}')
    plt.xlabel('Dice Coefficient')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Dice Coefficients')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'dice_distribution.png'))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['cloud_coverage'], results_df['dice_score'], alpha=0.5)
    plt.xlabel('Cloud Coverage (%)')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient vs. Cloud Coverage')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'dice_vs_coverage.png'))
    
    analyze_error_patterns(worst_samples, model, device, transform, os.path.join(output_dir, 'error_analysis.txt'))
    
    print(f"\nEvaluation results saved to {output_dir}")

def visualize_prediction(image_path, model, device, output_path, transform):
    """Visualize the image, ground truth, and prediction side by side."""
    mask_path = str(image_path).replace('/data/', '/masks/').replace('\\data\\', '\\masks\\')
    
    with rasterio.open(image_path) as img_file:
        r = img_file.read(1).astype(np.float32)
        g = img_file.read(2).astype(np.float32)
        b = img_file.read(3).astype(np.float32)
        
        for band in [r, g, b]:
            if np.max(band) > 0:
                band /= np.max(band)
        
        rgb = np.dstack((r, g, b))
        rgb = (rgb * 255).astype(np.uint8)
    
    with rasterio.open(mask_path) as mask_file:
        gt_mask = mask_file.read(1)
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
    
    pred_mask = predict_mask(model, image_path, device, transform)
    pred_mask = (pred_mask * 255).astype(np.uint8)
    
    rgb_resized = cv2.resize(rgb, (512, 512))
    gt_mask_resized = cv2.resize(gt_mask, (512, 512))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(rgb_resized)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask_resized, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_error_patterns(worst_samples, model, device, transform, output_path):
    """Analyze common error patterns in the worst performing samples."""
    error_types = {
        'false_positives': 0,
        'false_negatives': 0,
        'boundary_errors': 0,
        'small_objects_missed': 0,
        'noisy_predictions': 0
    }
    
    for _, row in tqdm(worst_samples.iterrows(), total=len(worst_samples), desc="Analyzing error patterns"):
        image_path = row['image_path']
        mask_path = str(image_path).replace('/data/', '/masks/').replace('\\data\\', '\\masks\\')
        
        with rasterio.open(mask_path) as mask_file:
            gt_mask = mask_file.read(1)
            gt_mask = (gt_mask > 0).astype(np.uint8)
        
        pred_mask = predict_mask(model, image_path, device, transform)
        pred_mask = pred_mask.astype(np.uint8)
        
        gt_mask_resized = transform(image=gt_mask)['image']
        
        false_pos = (pred_mask > 0) & (gt_mask_resized == 0)
        false_neg = (pred_mask == 0) & (gt_mask_resized > 0)
        
        dilated_gt = binary_dilation(gt_mask_resized, iterations=3)
        eroded_gt = binary_erosion(gt_mask_resized, iterations=3)
        boundary_region = dilated_gt & ~eroded_gt
        boundary_errors = np.sum((false_pos | false_neg) & boundary_region) > 0.3 * np.sum(boundary_region)
        
        from skimage.measure import label
        gt_labeled = label(gt_mask_resized)
        pred_labeled = label(pred_mask)
        
        gt_regions = np.bincount(gt_labeled.flatten())
        if len(gt_regions) > 1:  # Skip background
            gt_regions = gt_regions[1:]
            small_regions_missed = False
            for i, region_size in enumerate(gt_regions):
                if region_size < 100:  # Small object threshold
                    region_mask = (gt_labeled == i+1)
                    if np.sum(region_mask & (pred_mask > 0)) == 0:
                        small_regions_missed = True
                        break
        else:
            small_regions_missed = False
        
        pred_labeled = label(pred_mask)
        pred_regions = np.bincount(pred_labeled.flatten())
        noisy_pred = False
        if len(pred_regions) > 10:
            noisy_pred = True
        
        if np.sum(false_pos) > 0.1 * np.sum(pred_mask):
            error_types['false_positives'] += 1
        if np.sum(false_neg) > 0.1 * np.sum(gt_mask_resized):
            error_types['false_negatives'] += 1
        if boundary_errors:
            error_types['boundary_errors'] += 1
        if small_regions_missed:
            error_types['small_objects_missed'] += 1
        if noisy_pred:
            error_types['noisy_predictions'] += 1
    
    with open(output_path, 'w') as f:
        f.write("Error Pattern Analysis\n")
        f.write("=====================\n\n")
        f.write(f"Total samples analyzed: {len(worst_samples)}\n\n")
        
        for error_type, count in error_types.items():
            percentage = (count / len(worst_samples)) * 100
            f.write(f"{error_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nRecommendations for improvement:\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate cloud masking model')
    parser.add_argument('--model_path', type=str, default='models/cloud_mask_unet.pkl', help='Path to model file')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory containing validation images')
    parser.add_argument('--masks_dir', type=str, required=True, help='Directory containing validation masks')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--sample_count', type=int, default=None, help='Number of samples to evaluate (default: all)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    if model is None:
        return
    
    model = model.to(device)
    
    evaluate_model(
        model=model,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        device=device,
        sample_count=args.sample_count
    )

if __name__ == "__main__":
    main()
