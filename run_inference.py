import sys
import torch
import pickle
import rasterio
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import albumentations as A

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    
    Returns:
        str: RLE-encoded string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs[::2] -= 1

    return " ".join(map(str, runs))

def load_model(model_path):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def process_tiff(file_path, model, device, img_size=512):
    """Process a TIFF file and return the mask prediction."""
    transform = A.Compose([
        A.Resize(img_size, img_size),
    ])
    
    try:
        with rasterio.open(file_path) as img_file:
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
        
        if mask.shape != (img_size, img_size):
            mask = A.Resize(img_size, img_size)(image=mask)['image']
        
        return mask.astype(np.uint8)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((img_size, img_size), dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Run inference on test TIFF files')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test TIFF files')
    parser.add_argument('--model_path', type=str, default='models/cloud_mask_unet.pkl', help='Path to model file')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output CSV file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path).to(device)
    
    test_dir = Path(args.test_dir)
    test_files = sorted(list(test_dir.glob('*.tiff')))
    if not test_files:
        print(f"No TIFF files found in {args.test_dir}")
        sys.exit(1)
    
    print(f"Found {len(test_files)} test files")
    
    results = []
    for file_path in tqdm(test_files, desc="Processing"):
        file_id = file_path.stem
        mask = process_tiff(file_path, model, device)
        rle = rle_encode(mask)
        results.append((file_id, rle))
    
    submission_df = pd.DataFrame(results, columns=['id', 'segmentation'])
    submission_df.to_csv(args.output, index=False)
    print(f"Submission saved to {args.output}")

if __name__ == "__main__":
    main()
