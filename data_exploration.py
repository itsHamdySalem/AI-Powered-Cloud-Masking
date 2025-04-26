import os
import argparse
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_cloud_coverage(masks_dir):
    print("\nAnalyzing cloud coverage distribution...")
    mask_paths = sorted(list(Path(masks_dir).glob('*.tif*')))
    
    if not mask_paths:
        print(f"No mask files found in {masks_dir}")
        return
    
    coverage_percentages = []
    full_cloud = 0
    partial_cloud = 0
    no_cloud = 0
    
    for path in tqdm(mask_paths, desc="Analyzing masks"):
        with rasterio.open(path) as mask_file:
            mask = mask_file.read(1)
            mask = (mask > 0).astype(np.uint8)
            
            coverage = np.mean(mask) * 100
            coverage_percentages.append(coverage)
            
            if coverage >= 99:
                full_cloud += 1
            elif coverage <= 1:
                no_cloud += 1
            else:
                partial_cloud += 1
    
    print(f"\nTotal masks analyzed: {len(mask_paths)}")
    print(f"Fully clouded images (>=99%): {full_cloud} ({full_cloud/len(mask_paths)*100:.2f}%)")
    print(f"Partially cloudy images (1-99%): {partial_cloud} ({partial_cloud/len(mask_paths)*100:.2f}%)")
    print(f"Cloud-free images (<=1%): {no_cloud} ({no_cloud/len(mask_paths)*100:.2f}%)")
    
    plt.figure(figsize=(10, 6))
    plt.hist(coverage_percentages, bins=20, alpha=0.7, color='steelblue')
    plt.xlabel('Cloud Coverage Percentage')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Cloud Coverage in the Dataset')
    plt.grid(alpha=0.3)
    plt.savefig('cloud_coverage_distribution.png')
    print("Distribution plot saved as 'cloud_coverage_distribution.png'")
    
    coverage_df = pd.DataFrame({
        'cloud_coverage_percentage': coverage_percentages,
        'mask_path': [str(p) for p in mask_paths]
    })
    coverage_df.to_csv('cloud_coverage_analysis.csv', index=False)
    print("Detailed coverage data saved to 'cloud_coverage_analysis.csv'")

def analyze_band_statistics(images_dir):
    """Analyze the statistics of each spectral band."""
    print("\nAnalyzing spectral band statistics...")
    image_paths = sorted(list(Path(images_dir).glob('*.tif*')))
    
    if not image_paths:
        print(f"No image files found in {images_dir}")
        return
    
    sample_size = min(100, len(image_paths))
    sampled_paths = np.random.choice(image_paths, sample_size, replace=False)
    
    band_names = ['Red', 'Green', 'Blue', 'Infrared']
    band_means = [[] for _ in range(4)]
    band_stds = [[] for _ in range(4)]
    band_mins = [[] for _ in range(4)]
    band_maxs = [[] for _ in range(4)]
    
    for path in tqdm(sampled_paths, desc="Analyzing bands"):
        with rasterio.open(path) as img_file:
            for i in range(4):
                band = img_file.read(i+1).astype(np.float32)
                band_means[i].append(np.mean(band))
                band_stds[i].append(np.std(band))
                band_mins[i].append(np.min(band))
                band_maxs[i].append(np.max(band))
    
    print("\nBand Statistics (based on sampled images):")
    for i, name in enumerate(band_names):
        print(f"\n{name} Band:")
        print(f"  Mean: {np.mean(band_means[i]):.2f}")
        print(f"  Std Dev: {np.mean(band_stds[i]):.2f}")
        print(f"  Min: {np.mean(band_mins[i]):.2f}")
        print(f"  Max: {np.mean(band_maxs[i]):.2f}")
    
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(band_names):
        plt.boxplot(band_means[i], positions=[i], labels=[name])
    plt.ylabel('Mean Value')
    plt.title('Distribution of Band Mean Values')
    plt.grid(alpha=0.3)
    plt.savefig('band_mean_distribution.png')
    print("\nBand mean distribution plot saved as 'band_mean_distribution.png'")

def analyze_image_quality(images_dir, masks_dir, sample_size=10):
    """Analyze image quality and potential mislabeled data."""
    print("\nAnalyzing image quality and potential mislabeling...")
    image_paths = sorted(list(Path(images_dir).glob('*.tif*')))
    mask_paths = sorted(list(Path(masks_dir).glob('*.tif*')))
    
    if not image_paths or not mask_paths:
        print("Image or mask files not found")
        return
    
    if len(image_paths) > sample_size:
        indices = np.random.choice(len(image_paths), sample_size, replace=False)
        sampled_images = [image_paths[i] for i in indices]
        sampled_masks = [mask_paths[i] for i in indices]
    else:
        sampled_images = image_paths
        sampled_masks = mask_paths
    
    os.makedirs('sample_visualizations', exist_ok=True)
    
    for i, (img_path, mask_path) in enumerate(zip(sampled_images, sampled_masks)):
        with rasterio.open(img_path) as img_file, rasterio.open(mask_path) as mask_file:
            r = img_file.read(1).astype(np.float32)
            g = img_file.read(2).astype(np.float32)
            b = img_file.read(3).astype(np.float32)
            
            for band in [r, g, b]:
                if np.max(band) > 0:
                    band /= np.max(band)
            
            rgb = np.dstack((r, g, b))
            rgb = (rgb * 255).astype(np.uint8)
            
            mask = mask_file.read(1)
            mask = (mask > 0).astype(np.uint8) * 255
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(rgb)
            axes[0].set_title('RGB Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Cloud Mask')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'sample_visualizations/sample_{i+1}.png')
            plt.close()
    
    print(f"Sample visualizations saved to 'sample_visualizations' directory")

def main():
    parser = argparse.ArgumentParser(description='Explore cloud masking dataset')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory containing image files')
    parser.add_argument('--masks_dir', type=str, required=True, help='Directory containing mask files')
    args = parser.parse_args()
    
    print("Cloud Masking Dataset Exploration")
    print("=================================")
    
    analyze_cloud_coverage(args.masks_dir)
    
    analyze_band_statistics(args.images_dir)
    
    analyze_image_quality(args.images_dir, args.masks_dir)

if __name__ == "__main__":
    main()
