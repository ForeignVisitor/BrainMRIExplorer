import os
import h5py
import numpy as np
from PIL import Image
from src.h5_helper import load_h5_file, get_slice_image
from src.segment import load_unet_model, predict_mask


VOLUME_DIR = "./testing"         # Folder with demo .h5 volumes
OUT_DIR = "./precomputed"        
MODEL_PATH = "./models/unet_brats_h5_best.pth"

os.makedirs(OUT_DIR, exist_ok=True)

# Load segmentation model
seg_model = load_unet_model(MODEL_PATH)

for fname in os.listdir(VOLUME_DIR):
    if not fname.endswith('.h5'):
        continue
    file_id = os.path.splitext(fname)[0]
    print(f"Processing {fname} ...")
    volume = load_h5_file(os.path.join(VOLUME_DIR, fname))
    for idx in range(volume.shape[0]):
        #Save MRI slice
        img = get_slice_image(volume, idx)
        slice_png_path = os.path.join(OUT_DIR, f"{file_id}_{idx}.png")
        img.save(slice_png_path)

        #Save mask
        slice_np = volume[idx]  # (4, 240, 240)
        mask = predict_mask(seg_model, slice_np)  # (240, 240)
        mask_img = Image.fromarray((mask * 255).astype("uint8"))
        mask_png_path = os.path.join(OUT_DIR, f"{file_id}_{idx}_mask.png")
        mask_img.save(mask_png_path)
    print(f"Done {fname}")

print("All demo slices and masks precomputed!")