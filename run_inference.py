import pickle
from pathlib import Path
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from skimage.transform import resize
from tqdm import tqdm

def load_model(path):
    """
    Loads a saved PyTorch model from the specified file path using pickle.

    Args:
        path (str): Path to the pickled model file.

    Returns:
        torch.nn.Module: Loaded PyTorch model moved to the configured device and set to evaluation mode.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()
    return model

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).

    Returns:
        str: RLE-encoded string, or a single space " " if mask is all zeros.
    """
    if np.sum(mask) == 0:
        return " "
    
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs[::2] -= 1

    return " ".join(map(str, runs))

def run_inference_to_csv( tiff_dir, model, csv_path, img_size, 
                         rle_size, device, threshold, show, max_show,
                        ) -> None:
    """
    Runs model inference on a directory of TIFF images, saves RLE-encoded masks to CSV,
    and optionally visualizes a subset of predictions.

    Args:
        tiff_dir (str or Path): Directory containing TIFF image files for inference.
        model (torch.nn.Module): Trained model for segmentation.
        csv_path (str or Path): Output path for the CSV file.
        img_size (int): Size to which images will be resized for inference.
        rle_size (int): Size for resizing predictions before RLE encoding.
        device (torch.device): Device for model inference.
        threshold (float): Probability threshold for binarizing model outputs.
        show (bool): Whether to display visualization of predictions.
        max_show (int): Maximum number of samples to visualize.

    Returns:
        None
    """
    tiff_paths: List[Path] = sorted(Path(tiff_dir).glob("*.tif*"))
    assert tiff_paths, f"No TIFF images found in '{tiff_dir}'."

    rows, visuals = [], []

    for p in tqdm(tiff_paths, desc="Inference"):
        with rasterio.open(p) as src:
            img4 = np.stack([src.read(i + 1) for i in range(4)], axis=2).astype(np.float32)

        for c in range(4):
            m = img4[..., c].max()
            if m > 0:
                img4[..., c] /= m

        img_r = cv2.resize(img4, (img_size, img_size), cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_r.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            prob   = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask_pred = (prob > threshold).astype(np.uint8)

        mask_rle = resize(mask_pred, (rle_size, rle_size), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        rows.append(
            {
                "id": p.stem,
                "segmentation": rle_encode(mask_rle),
            }
        )

        if show and len(visuals) < max_show:
            rgb_disp = img_r[..., :3]
            rgb_disp = (rgb_disp - rgb_disp.min()) / (rgb_disp.ptp() + 1e-6)
            visuals.append((rgb_disp, mask_pred))

    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(rows)} predictions → {csv_path}")

    if show and visuals:
        n_vis       = len(visuals)
        dpi         = plt.rcParams.get("figure.dpi", 100)
        row_height  = 3.0
        max_inches  = 65500 / dpi
        rows_per_fig_max = int(max_inches // row_height)
    
        for chunk_idx in range(0, n_vis, rows_per_fig_max):
            chunk = visuals[chunk_idx : chunk_idx + rows_per_fig_max]
            rows  = len(chunk)
            fig, axes = plt.subplots(rows, 3,
                                     figsize=(12, row_height * rows),
                                     squeeze=False)
    
            for r, (rgb, msk) in enumerate(chunk):
                axes[r, 0].imshow(rgb)
                axes[r, 0].set_title("Resized RGB")
                axes[r, 0].axis("off")
    
                axes[r, 1].imshow(rgb)
                axes[r, 1].imshow(msk, alpha=0.5, cmap="cool")
                axes[r, 1].set_title("Mask overlay")
                axes[r, 1].axis("off")
    
                axes[r, 2].imshow(msk, cmap="gray")
                axes[r, 2].set_title("Binary mask")
                axes[r, 2].axis("off")
    
            fig.tight_layout()
            plt.show()



if __name__ == "__main__":
    TEST_SET = "test_set/"
    MODEL_PATH = "models/cloud_mask_deeplab.pkl"

    model = load_model(MODEL_PATH)

    run_inference_to_csv(
        tiff_dir=Path(TEST_SET),
        model=model,
        csv_path="test_results.csv",
        img_size=512,
        rle_size=256,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        threshold=0.5,
        show=True,
        max_show=10,
    )
